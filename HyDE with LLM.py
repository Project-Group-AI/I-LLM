import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

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
        
        # Modèle Gemini
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
        
        # Chargement des données
        self.load_data()
        
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
            import json
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
    
    def format_result(self, establishments):
        """
        Formate les résultats pour l'affichage.
        
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
    
    def is_off_topic(self, query):
        """
        Vérifie si la requête est hors sujet en utilisant Gemini.
        
        Args:
            query (str): Requête de l'utilisateur
            
        Returns:
            bool: True si la requête est hors sujet, False sinon
        """
        prompt = f"""
        Détermine si la requête suivante est une demande de recherche d'établissement public en France ou une question sur l'accessibilité d'un lieu:
        
        "{query}"
        
        Réponds uniquement par "Pertinent" si c'est une question sur un établissement public ou l'accessibilité,
        ou "Hors sujet" si c'est une question sur un autre domaine (météo, sport, calcul, etc.).
        """
        
        try:
            response = self.gemini_model.generate_content(prompt)
            return "Hors sujet" in response.text
        except Exception as e:
            print(f"Erreur lors de la vérification du sujet: {e}")
            return False
    
    def process_query(self, query):
        """
        Traite la requête de l'utilisateur et renvoie une réponse.
        
        Args:
            query (str): Requête de l'utilisateur
            
        Returns:
            str: Réponse du chatbot
        """
        # Vérification si la requête est hors sujet avec Gemini
        if self.is_off_topic(query):
            return "Je suis désolé, je ne peux vous aider que pour des recherches d'établissements publics en France et leur accessibilité."
        
        # Génération du document hypothétique avec Gemini
        hypothetical_doc, criteria = self.generate_hypothetical_document_with_gemini(query)
        
        # Affichage du document hypothétique pour le débogage
        print("Document hypothétique généré:")
        print(hypothetical_doc)
        print("Critères extraits:", criteria)
        
        # Vérification des critères suffisants
        if not criteria:
            return "Je ne comprends pas votre demande. Pourriez-vous préciser quel type d'établissement vous recherchez et dans quelle ville ?"
        
        # Si pas de critères de localisation, demander plus d'informations
        if 'commune' not in criteria:
            return "Pourriez-vous préciser dans quelle commune ou ville vous souhaitez effectuer votre recherche ?"
        
        # Recherche des établissements
        establishments = self.search_establishments(criteria)
        
        # Vérification des résultats
        if not establishments:
            # Si pas de résultats et qu'on a spécifié un type d'établissement
            if 'activite' in criteria:
                return f"Je n'ai pas trouvé de {criteria['activite']} à {criteria['commune']}. Essayez peut-être avec un autre type d'établissement ou une autre commune."
            else:
                return f"Je n'ai pas trouvé d'établissement correspondant à vos critères à {criteria['commune']}."
        
        # Formatage de la réponse
        response = self.format_result(establishments)
        
        return response

# Exemple d'utilisation
if __name__ == "__main__":
    dataset_path = "/Users/romain/Desktop/HyDE/acceslibre-with-web-url.csv"
    api_key = "AIzaSyDbziFf_7_kDv0uFnv4hvIUrITCr1QIZzo"  # Remplacez par votre clé API
    
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