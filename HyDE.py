import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os

class ChatbotInclusif:
    def __init__(self, dataset_path):
        """
        Initialise le chatbot inclusif avec le dataset des établissements publics.
        
        Args:
            dataset_path (str): Chemin vers le fichier CSV du dataset
        """
        self.dataset_path = dataset_path
        self.model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
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
        
    def generate_hypothetical_document(self, query):
        """
        Génère un document hypothétique basé sur la requête de l'utilisateur.
        
        Args:
            query (str): La requête de l'utilisateur
            
        Returns:
            str: Document hypothétique généré
            dict: Critères extraits de la requête
        """
        # Définition des patterns pour extraire les informations
        patterns = {
            'localisation': r'(à|a|sur|dans|près de|proche de|autour de)\s+([A-Za-zÀ-ÿ\-\s]+)',
            'type_etablissement': r'(restaurant|café|hôtel|musée|bibliothèque|mairie|école|hôpital|cinéma|théâtre|piscine|gymnase|stade|parc|jardin|office de tourisme|gare|station|aéroport|supermarché|magasin|boutique|pharmacie|banque|poste|commissariat|coiffeur|salon de coiffure|boulangerie|épicerie|librairie|médiathèque|centre commercial|brasserie|cabinet médical|dentiste|opticien)',
            'pmr': r'(pmr|personne à mobilité réduite|fauteuil roulant|handicap moteur)',
            'malentendant': r'(malentendant|sourd|handicap auditif)',
            'malvoyant': r'(malvoyant|aveugle|handicap visuel)',
            'stationnement': r'(parking|stationnement|place de parking)',
            'transport': r'(transport|bus|métro|tram|tramway|train)',
            'sanitaires': r'(toilette|wc|sanitaire)'
        }
        
        # Extraction des informations
        extracted_info = {}
        
        # Extraction de la localisation
        loc_matches = re.findall(patterns['localisation'], query.lower())
        if loc_matches:
            for match in loc_matches:
                if len(match) > 1:
                    extracted_info['commune'] = match[1].strip()
        
        # Extraction du type d'établissement
        type_matches = re.findall(patterns['type_etablissement'], query.lower())
        if type_matches:
            extracted_info['type_etablissement'] = type_matches[0]
        else:
            # Recherche de types d'établissements spécifiques non listés dans le pattern général
            specific_types = {
                'coiffeur': r'(coiffeur|salon de coiffure|coiffure)',
                'boulangerie': r'(boulangerie|boulanger|pain)',
                'pharmacie': r'(pharmacie|pharmacien)',
                'bar': r'(bar|café|bistrot)',
                'médecin': r'(médecin|docteur|cabinet médical)',
                'école': r'(école|établissement scolaire|collège|lycée)'
            }
            
            for type_name, type_pattern in specific_types.items():
                if re.search(type_pattern, query.lower()):
                    extracted_info['type_etablissement'] = type_name
                    break
        
        # Extraction des autres critères
        for key, pattern in patterns.items():
            if key not in ['localisation', 'type_etablissement']:
                if re.search(pattern, query.lower()):
                    extracted_info[key] = True
        
        # Création du document hypothétique
        hypothetical_doc = {}
        
        # Critères de localisation
        if 'commune' in extracted_info:
            hypothetical_doc['commune'] = extracted_info['commune']
        
        # Critères d'accessibilité PMR
        if 'pmr' in extracted_info:
            hypothetical_doc['entree_pmr'] = True
            hypothetical_doc['cheminement_ext_plain_pied'] = True
            hypothetical_doc['stationnement_pmr'] = True
        
        # Critères pour malentendants
        if 'malentendant' in extracted_info:
            hypothetical_doc['accueil_equipements_malentendants_presence'] = True
        
        # Critères pour malvoyants
        if 'malvoyant' in extracted_info:
            hypothetical_doc['accueil_audiodescription_presence'] = True
            hypothetical_doc['cheminement_ext_bande_guidage'] = True
            hypothetical_doc['entree_reperage'] = True
        
        # Critères de stationnement
        if 'stationnement' in extracted_info:
            hypothetical_doc['stationnement_presence'] = True
            
        # Critères de transport
        if 'transport' in extracted_info:
            hypothetical_doc['transport_station_presence'] = True
            
        # Critères de sanitaires
        if 'sanitaires' in extracted_info:
            hypothetical_doc['sanitaires_presence'] = True
            
        # Critères de type d'établissement
        if 'type_etablissement' in extracted_info:
            hypothetical_doc['activite'] = extracted_info['type_etablissement']
            
        # Création du document texte
        doc_text = ""
        for key, value in hypothetical_doc.items():
            doc_text += f"{key} = {value}\n"
            
        return doc_text, hypothetical_doc
    
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
    
    def process_query(self, query):
        """
        Traite la requête de l'utilisateur et renvoie une réponse.
        
        Args:
            query (str): Requête de l'utilisateur
            
        Returns:
            str: Réponse du chatbot
        """
        # Vérification si la requête est hors sujet
        hors_sujet_keywords = ['météo', 'calcul', 'mathématiques', 'sport', 'résultat', 'actualité', 'recette', 'chanson']
        if any(keyword in query.lower() for keyword in hors_sujet_keywords):
            return "Je suis désolé, je ne peux vous aider que pour des recherches d'établissements publics en France et leur accessibilité."
        
        # Génération du document hypothétique
        hypothetical_doc, criteria = self.generate_hypothetical_document(query)
        
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

# Test du chatbot
if __name__ == "__main__":
    dataset_path = "/Users/romain/Desktop/HyDE/acceslibre-with-web-url.csv"
    chatbot = ChatbotInclusif(dataset_path)
    
    print("Chatbot Inclusif initialisé. Posez votre question (ou tapez 'quit' pour quitter):")
    
    while True:
        user_query = input("\nVotre question: ")
        
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("Au revoir !")
            break
            
        response = chatbot.process_query(user_query)
        print("\nRéponse du chatbot:")
        print(response)