import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import json

class ChatbotInclusifGemini:
    def __init__(self, dataset_path, api_key):
        """
        Initialise le chatbot inclusif avec le dataset des établissements publics
        et l'API Gemini 2.0 Flash.
        
        Args:
            dataset_path (str): Chemin vers le fichier CSV du dataset
            api_key (str): Clé API pour Gemini
        """
        self.dataset_path = dataset_path
        self.model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        
        # Configuration de Gemini
        genai.configure(api_key=api_key)
        
        # Modèle Gemini pour la génération de documents hypothétiques
        generation_config = {
            "temperature": 0.2,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 1024,
        }
        
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        self.gemini_model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        # Configuration pour les réponses naturelles (température plus élevée)
        natural_generation_config = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "max_output_tokens": 1024,
        }
        
        self.gemini_natural_model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config=natural_generation_config,
            safety_settings=safety_settings
        )
        
        # Base de connaissances pour les questions générales
        # Température plus élevée pour les réponses informatives
        knowledge_generation_config = {
            "temperature": 0.5,
            "top_p": 0.9,
            "top_k": 40,
            "max_output_tokens": 2048,  # Plus long pour les réponses informatives
        }
        
        self.gemini_knowledge_model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config=knowledge_generation_config,
            safety_settings=safety_settings
        )
        
        # Chargement des données
        self.load_data()
        
        # Initialisation de la base de connaissances sur le handicap
        self.initialize_knowledge_base()
        
    def load_data(self):
        """
        Charge le dataset et prépare les données
        """
        print("Chargement du dataset...")
        self.df = pd.read_csv(self.dataset_path, low_memory=False)
        
        # Conversion des colonnes booléennes
        bool_columns = [col for col in self.df.columns if any(x in col for x in ['presence', 'pmr', 'stable', 'plain_pied', 'adaptes'])]
        for col in bool_columns:
            self.df[col] = self.df[col].astype(str).str.lower().map({'true': True, 'false': False, 'nan': np.nan, 'none': np.nan})
        
        print(f"Dataset chargé avec succès. {len(self.df)} établissements disponibles.")
    
    def initialize_knowledge_base(self):
        """
        Initialise la base de connaissances sur le handicap et l'accessibilité
        """
        # Dictionnaire contenant des informations essentielles sur les différents types de handicap
        # et les aides disponibles en France
        self.knowledge_base = {
            "types_handicap": [
                "moteur", "visuel", "auditif", "mental", "psychique", "cognitif", "polyhandicap"
            ],
            "aides_financieres": [
                "AAH (Allocation aux Adultes Handicapés)",
                "PCH (Prestation de Compensation du Handicap)",
                "AEEH (Allocation d'Éducation de l'Enfant Handicapé)",
                "MVA (Majoration pour la Vie Autonome)",
                "ACTP (Allocation Compensatrice pour Tierce Personne)"
            ],
            "organismes": [
                "MDPH (Maison Départementale des Personnes Handicapées)",
                "CAF (Caisse d'Allocations Familiales)",
                "CCAS (Centre Communal d'Action Sociale)",
                "CPAM (Caisse Primaire d'Assurance Maladie)",
                "APF France Handicap",
                "UNAPEI"
            ],
            "droits_accessibilite": [
                "Loi de 2005 pour l'égalité des droits et des chances",
                "Obligation d'accessibilité des ERP (Établissements Recevant du Public)",
                "Carte Mobilité Inclusion (CMI)",
                "RQTH (Reconnaissance de la Qualité de Travailleur Handicapé)"
            ]
        }
        
    def classify_query_type(self, query):
        """
        Classifie le type de requête de l'utilisateur pour déterminer
        s'il s'agit d'une recherche d'établissement ou d'une question générale.
        
        Args:
            query (str): La requête de l'utilisateur
            
        Returns:
            str: Type de requête ('establishment_search', 'general_info', 'off_topic')
        """
        prompt = f"""
        Analyse la requête suivante et détermine son type:
        
        "{query}"
        
        Classifie la requête dans l'une des catégories suivantes:
        1. "establishment_search" - Une recherche d'établissement public ou d'un lieu avec ou sans critères d'accessibilité
        2. "general_info" - Une question générale sur le handicap, l'accessibilité, les droits, les aides, etc.
        3. "off_topic" - Une question qui n'est pas liée au handicap ou à l'accessibilité
        
        Réponds uniquement par le type de requête (un seul mot parmi les trois proposés), sans aucune explication.
        """
        
        try:
            response = self.gemini_model.generate_content(prompt)
            response_text = response.text.strip().lower()
            
            if "establishment_search" in response_text:
                return "establishment_search"
            elif "general_info" in response_text:
                return "general_info"
            else:
                return "off_topic"
        except Exception as e:
            print(f"Erreur lors de la classification de la requête: {e}")
            # Par défaut, on suppose que c'est une recherche d'établissement
            return "establishment_search"
        
    def generate_hypothetical_document_with_gemini(self, query):
        """
        Génère un document hypothétique basé sur la requête de l'utilisateur en utilisant Gemini.
        
        Args:
            query (str): La requête de l'utilisateur
            
        Returns:
            str: Document hypothétique généré
            dict: Critères extraits de la requête
        """
        # Construire le prompt pour Gemini
        prompt = f"""
        Analyse la requête utilisateur suivante et extrait les critères de recherche pertinents pour un établissement public en France.
        
        Requête: "{query}"
        
        Crée un document hypothétique au format JSON avec uniquement les champs suivants si pertinents:
        - commune: le nom de la ville ou commune
        - activite: le type d'établissement (restaurant, musée, etc.)
        - entree_pmr: true si l'accessibilité PMR est mentionnée
        - stationnement_pmr: true si le stationnement PMR est mentionné
        - stationnement_presence: true si le stationnement est mentionné
        - sanitaires_presence: true si des sanitaires sont mentionnés
        - sanitaires_adaptes: true si des sanitaires adaptés sont mentionnés
        - accueil_equipements_malentendants_presence: true si des équipements pour malentendants sont mentionnés
        - accueil_audiodescription_presence: true si l'audiodescription est mentionnée
        - cheminement_ext_bande_guidage: true si des bandes de guidage sont mentionnées
        - cheminement_ext_plain_pied: true si l'accessibilité de plain-pied est mentionnée
        - transport_station_presence: true si la proximité des transports est mentionnée
        
        Retourne uniquement le JSON, sans aucun autre texte.
        Si la requête ne concerne pas la recherche d'un établissement public, retourne un JSON vide {{}}.
        """
        
        try:
            # Appel à l'API Gemini
            response = self.gemini_model.generate_content(prompt)
            
            # Récupération du texte
            response_text = response.text
            
            # Extraction du JSON
            # Nettoyage du texte pour s'assurer qu'il est au format JSON
            json_text = response_text.strip()
            if json_text.startswith('```json'):
                json_text = json_text[7:]
            if json_text.endswith('```'):
                json_text = json_text[:-3]
            json_text = json_text.strip()
            
            # Conversion en dictionnaire
            criteria = json.loads(json_text)
            
            # Création du document texte
            doc_text = ""
            for key, value in criteria.items():
                doc_text += f"{key} = {value}\n"
                
            return doc_text, criteria
            
        except Exception as e:
            print(f"Erreur lors de l'appel à Gemini: {e}")
            return "", {}
        
    def search_establishments(self, criteria, top_n=5):
        """
        Recherche les établissements correspondant aux critères extraits.
        
        Args:
            criteria (dict): Critères de recherche
            top_n (int): Nombre maximum de résultats à retourner
            
        Returns:
            list: Liste des établissements correspondants
        """
        # Filtre initial
        filtered_df = self.df.copy()
        
        # Traitement spécial pour le type d'établissement (activité)
        has_activite_filter = False
        if 'activite' in criteria:
            has_activite_filter = True
            activite_value = criteria['activite']
            # Recherche insensible à la casse pour l'activité
            filtered_df = filtered_df[filtered_df['activite'].str.lower().str.contains(activite_value.lower(), na=False)]
            
        # Traitement spécial pour la commune
        has_commune_filter = False
        if 'commune' in criteria:
            has_commune_filter = True
            commune_value = criteria['commune']
            # Recherche insensible à la casse pour la commune
            filtered_df = filtered_df[filtered_df['commune'].str.lower().str.contains(commune_value.lower(), na=False)]
        
        # Autres filtres
        for key, value in criteria.items():
            if key not in ['activite', 'commune'] and key in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[key] == value]
        
        # Si aucun résultat avec tous les critères, on essaie une recherche plus souple
        if len(filtered_df) == 0 and has_activite_filter and has_commune_filter:
            print("Aucun résultat exact trouvé, essai d'une recherche plus souple...")
            
            # On essaie d'abord avec juste l'activité et la commune
            filtered_df = self.df.copy()
            
            if has_activite_filter:
                activite_value = criteria['activite']
                filtered_df = filtered_df[filtered_df['activite'].str.lower().str.contains(activite_value.lower(), na=False)]
            
            if has_commune_filter and len(filtered_df) > 0:
                commune_value = criteria['commune']
                filtered_df = filtered_df[filtered_df['commune'].str.lower().str.contains(commune_value.lower(), na=False)]
        
        # Si toujours aucun résultat, on essaie avec une recherche encore plus large
        if len(filtered_df) == 0 and has_commune_filter:
            print("Toujours aucun résultat, essai avec seulement la commune...")
            filtered_df = self.df.copy()
            commune_value = criteria['commune']
            filtered_df = filtered_df[filtered_df['commune'].str.lower().str.contains(commune_value.lower(), na=False)]
            
            # On limite alors à quelques résultats aléatoires
            if len(filtered_df) > top_n:
                filtered_df = filtered_df.sample(top_n)
        
        # Limitation du nombre de résultats
        if len(filtered_df) > top_n:
            filtered_df = filtered_df.head(top_n)
            
        return filtered_df.to_dict('records')
    
    def generate_natural_response(self, establishments, query, criteria):
        """
        Génère une réponse naturelle aux résultats trouvés en utilisant Gemini.
        
        Args:
            establishments (list): Liste des établissements trouvés
            query (str): La requête originale de l'utilisateur
            criteria (dict): Les critères extraits de la requête
            
        Returns:
            str: Réponse naturelle générée
        """
        if not establishments:
            # Si aucun établissement trouvé
            prompt = f"""
            Tu es un assistant spécialisé dans la recherche d'établissements publics accessibles en France.
            
            L'utilisateur a demandé: "{query}"
            
            J'ai cherché dans ma base de données avec les critères suivants:
            {json.dumps(criteria, ensure_ascii=False, indent=2)}
            
            Mais je n'ai trouvé aucun établissement correspondant. 
            
            Générez une réponse empathique, naturelle et en français qui:
            1. Informe l'utilisateur qu'aucun établissement n'a été trouvé
            2. Suggère de reformuler sa demande ou d'élargir ses critères
            3. Si une ville est mentionnée, réutilise-la dans la réponse
            4. Si un type d'établissement est mentionné, réutilise-le dans la réponse
            
            Écrivez votre réponse comme si vous vous adressiez directement à l'utilisateur, sans introduction ni conclusion artificielle.
            """
        else:
            # Préparation des données des établissements pour le prompt
            establishments_data = []
            for i, estab in enumerate(establishments, 1):
                estab_info = {
                    "nom": estab.get('name', 'Établissement sans nom'),
                    "activite": estab.get('activite', 'Non spécifié'),
                }
                
                # Adresse
                address_parts = []
                if estab.get('numero'):
                    address_parts.append(str(estab['numero']))
                if estab.get('voie'):
                    address_parts.append(estab['voie'])
                if estab.get('commune'):
                    address_parts.append(estab['commune'])
                if estab.get('postal_code'):
                    address_parts.append(str(estab['postal_code']))
                    
                estab_info["adresse"] = ' '.join(str(part) for part in address_parts if part)
                
                # Accessibilité
                access_features = []
                if estab.get('entree_pmr') == True:
                    access_features.append("Entrée accessible PMR")
                if estab.get('stationnement_pmr') == True:
                    access_features.append("Stationnement PMR")
                if estab.get('sanitaires_adaptes') == True:
                    access_features.append("Sanitaires adaptés")
                if estab.get('accueil_equipements_malentendants_presence') == True:
                    access_features.append("Équipements pour malentendants")
                if estab.get('accueil_audiodescription_presence') == True:
                    access_features.append("Audiodescription disponible")
                    
                estab_info["accessibilite"] = access_features
                
                # Contact
                contact_info = {}
                if estab.get('contact_url'):
                    contact_info["contact"] = estab['contact_url']
                if estab.get('site_internet'):
                    contact_info["site_web"] = estab['site_internet']
                if estab.get('web_url'):
                    contact_info["page_web"] = estab['web_url']
                if estab.get('latitude') and estab.get('longitude'):
                    contact_info["coordonnees_gps"] = f"{estab['latitude']}, {estab['longitude']}"
                    
                estab_info["contact"] = contact_info
                
                establishments_data.append(estab_info)
                
            # Prompt pour Gemini
            prompt = f"""
            Tu es un assistant spécialisé dans la recherche d'établissements publics accessibles en France.
            
            L'utilisateur a demandé: "{query}"
            
            J'ai trouvé {len(establishments_data)} établissement(s) qui correspondent aux critères suivants:
            {json.dumps(criteria, ensure_ascii=False, indent=2)}
            
            Voici les informations sur ces établissements:
            {json.dumps(establishments_data, ensure_ascii=False, indent=2)}
            
            Générez une réponse naturelle, conversationnelle et en français qui:
            1. Commence par confirmer que vous avez trouvé des établissements correspondant à la demande
            2. Présentez brièvement chaque établissement en mentionnant:
               - Son nom
               - Son adresse
               - Ses caractéristiques d'accessibilité
               - Les informations de contact si disponibles
            3. Mettez en avant les aspects d'accessibilité qui correspondent spécifiquement à la demande de l'utilisateur
            4. Si les établissements ont des caractéristiques communes, regroupez-les pour éviter la répétition
            
            Important:
            - Utilisez un ton naturel, conversationnel et empathique
            - Évitez les formulations trop formelles ou robotiques
            - Incluez des phrases de transition naturelles entre les établissements
            - Adaptez votre réponse pour qu'elle réponde directement aux préoccupations d'accessibilité mentionnées dans la requête
            - Ne mentionnez pas explicitement que vous êtes un assistant IA dans votre réponse
            
            Écrivez votre réponse comme si vous vous adressiez directement à l'utilisateur, sans introduction ni conclusion artificielle.
            """
            
        try:
            # Appel à l'API Gemini avec une température plus élevée pour des réponses naturelles
            response = self.gemini_natural_model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Erreur lors de la génération de la réponse naturelle: {e}")
            # En cas d'erreur, on revient à la méthode de formatage classique
            return self.format_result(establishments)

    def generate_knowledge_response(self, query):
        """
        Génère une réponse pour les questions générales sur le handicap et l'accessibilité.
        
        Args:
            query (str): La requête de l'utilisateur
            
        Returns:
            str: Réponse informative sur le sujet demandé
        """
        # Extraction des mots-clés potentiels
        query_lower = query.lower()
        relevant_keywords = []
        
        # Vérification des types de handicap mentionnés
        for handicap_type in self.knowledge_base["types_handicap"]:
            if handicap_type in query_lower:
                relevant_keywords.append(handicap_type)
        
        # Vérification des aides financières mentionnées
        for aide in self.knowledge_base["aides_financieres"]:
            aide_lower = aide.lower()
            if aide_lower in query_lower or aide.split('(')[0].strip().lower() in query_lower:
                relevant_keywords.append(aide)
        
        # Vérification des organismes mentionnés
        for organisme in self.knowledge_base["organismes"]:
            organisme_lower = organisme.lower()
            if organisme_lower in query_lower or organisme.split('(')[0].strip().lower() in query_lower:
                relevant_keywords.append(organisme)
        
        # Vérification des droits mentionnés
        for droit in self.knowledge_base["droits_accessibilite"]:
            if droit.lower() in query_lower:
                relevant_keywords.append(droit)
        
        # Création du prompt pour Gemini
        prompt = f"""
        Tu es un assistant spécialisé dans l'information sur le handicap et l'accessibilité en France.
        
        L'utilisateur a posé la question suivante:
        "{query}"
        
        Voici quelques informations pertinentes sur ce sujet:
        
        Types de handicap: {', '.join(self.knowledge_base["types_handicap"])}
        
        Aides financières principales:
        {', '.join(self.knowledge_base["aides_financieres"])}
        
        Organismes importants:
        {', '.join(self.knowledge_base["organismes"])}
        
        Droits et législation:
        {', '.join(self.knowledge_base["droits_accessibilite"])}
        
        Mots-clés particulièrement pertinents dans cette question:
        {', '.join(relevant_keywords) if relevant_keywords else "Aucun mot-clé spécifique identifié"}
        
        Génère une réponse en français qui:
        1. Répond directement à la question de l'utilisateur
        2. Fournit des informations précises et à jour sur le sujet du handicap et de l'accessibilité
        3. Mentionne les aides ou organismes pertinents quand c'est approprié
        4. Utilise un ton bienveillant, empathique et inclusif
        5. Évite le jargon technique quand c'est possible
        6. Reste factuel et précis
        
        Si tu ne connais pas la réponse exacte à une question très spécifique (comme les montants exacts des aides), précise que ces informations peuvent varier et qu'il est préférable de contacter directement les organismes concernés.
        
        Écris ta réponse comme si tu t'adressais directement à l'utilisateur, sans introduction ni conclusion artificielle.
        """
        
        try:
            # Appel à l'API Gemini avec une température moyenne pour des réponses informatives
            response = self.gemini_knowledge_model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Erreur lors de la génération de la réponse sur les connaissances: {e}")
            return "Je suis désolé, mais je ne peux pas répondre à cette question pour le moment. Veuillez contacter directement la MDPH de votre département pour obtenir des informations précises sur les aides et services disponibles."
    
    def format_result(self, establishments):
        """
        Méthode de secours pour formater les résultats si la génération naturelle échoue.
        
        Args:
            establishments (list): Liste des établissements trouvés
            
        Returns:
            str: Réponse formatée
        """
        if not establishments:
            return "Je n'ai pas trouvé d'établissement correspondant à votre demande."
            
        response = f"J'ai trouvé {len(establishments)} établissement(s) correspondant à votre demande :\n\n"
        
        for i, estab in enumerate(establishments, 1):
            response += f"{i}. {estab.get('name', 'Établissement sans nom')}\n"
            
            # Type d'activité
            if estab.get('activite'):
                response += f"   Activité: {estab['activite']}\n"
            
            # Adresse
            address_parts = []
            if estab.get('numero'):
                address_parts.append(str(estab['numero']))
            if estab.get('voie'):
                address_parts.append(estab['voie'])
            if estab.get('commune'):
                address_parts.append(estab['commune'])
            if estab.get('postal_code'):
                address_parts.append(str(estab['postal_code']))
                
            if address_parts:
                response += f"   Adresse: {' '.join(str(part) for part in address_parts if part)}\n"
            
            # Accessibilité
            accessibility = []
            if estab.get('entree_pmr') == True:
                accessibility.append("Entrée accessible PMR")
            if estab.get('stationnement_pmr') == True:
                accessibility.append("Stationnement PMR")
            if estab.get('sanitaires_adaptes') == True:
                accessibility.append("Sanitaires adaptés")
            if estab.get('accueil_equipements_malentendants_presence') == True:
                accessibility.append("Équipements pour malentendants")
            if estab.get('accueil_audiodescription_presence') == True:
                accessibility.append("Audiodescription disponible")
                
            if accessibility:
                response += f"   Accessibilité: {', '.join(accessibility)}\n"
            
            # Coordonnées et sites web
            if estab.get('contact_url'):
                response += f"   Contact: {estab['contact_url']}\n"
            if estab.get('site_internet'):
                response += f"   Site web: {estab['site_internet']}\n"
            if estab.get('web_url'):
                response += f"   Page web: {estab['web_url']}\n"
                
            # Coordonnées géographiques
            if estab.get('latitude') and estab.get('longitude'):
                response += f"   Coordonnées GPS: {estab['latitude']}, {estab['longitude']}\n"
                
            response += "\n"
            
        return response
    
    def process_query(self, query):
        """
        Traite la requête de l'utilisateur et renvoie une réponse.
        
        Args:
            query (str): Requête de l'utilisateur
            
        Returns:
            str: Réponse du chatbot
        """
        # Classification de la requête
        query_type = self.classify_query_type(query)
        print(f"Type de requête détecté: {query_type}")
        
        # Traitement selon le type de requête
        if query_type == "off_topic":
            return "Je suis désolé, je ne peux vous aider que pour des recherches d'établissements publics en France et des informations sur le handicap et l'accessibilité."
        
        elif query_type == "general_info":
            # Traitement des questions générales sur le handicap
            return self.generate_knowledge_response(query)
        
        else:  # establishment_search
            # Génération du document hypothétique avec Gemini
            hypothetical_doc, criteria = self.generate_hypothetical_document_with_gemini(query)
            
            # Affichage du document hypothétique pour le débogage
            print("Document hypothétique généré:")
            print(hypothetical_doc)
            print("Critères extraits:", criteria)
            
            # Vérification des critères suffisants
            if not criteria:
                return "Je ne comprends pas votre demande concernant un établissement. Pourriez-vous préciser quel type d'établissement vous recherchez et dans quelle ville ?"
            
            # Si pas de critères de localisation, demander plus d'informations
            if 'commune' not in criteria:
                return "Pourriez-vous préciser dans quelle commune ou ville vous souhaitez effectuer votre recherche ?"
            
            # Recherche des établissements
            establishments = self.search_establishments(criteria)
            
            # Générer une réponse naturelle avec le LLM
            response = self.generate_natural_response(establishments, query, criteria)
            
            return response

# Exemple d'utilisation
if __name__ == "__main__":
    dataset_path = "/Users/romain/Desktop/HyDE/acceslibre-with-web-url.csv"
    api_key = ""  # Remplacez par votre clé API
    
    chatbot = ChatbotInclusifGemini(dataset_path, api_key)
    
    
    print("Chatbot Inclusif avec Gemini initialisé. Posez votre question (ou tapez 'quit' pour quitter):")
    
    while True:
        user_query = input("\nVotre question: ")
        
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("Au revoir !")
            break
            
        response = chatbot.process_query(user_query)
        print("\nRéponse du chatbot:")
        print(response)