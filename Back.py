# --- START OF FILE Back.py ---

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import json
import requests
from urllib.parse import quote
import math # Added for distance calculation

class ChatbotInclusifGemini:
    def __init__(self, api_base_url, gemini_api_key):
        """
        Initialise le chatbot inclusif avec l'API des établissements publics
        et l'API Gemini 2.0 Flash.

        Args:
            api_base_url (str): URL de base de l'API des établissements
            gemini_api_key (str): Clé API pour Gemini
        """
        self.api_base_url = api_base_url
        self.pmr_parking_api_url = "https://data.seineouest.fr/api/explore/v2.1/catalog/datasets/places-de-stationnement-pmr/records"
        self.model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

        self.seine_ouest_cities = {
            "Boulogne-Billancourt", "Chaville", "Issy-les-Moulineaux",
            "Marnes-la-Coquette", "Meudon", "Sèvres", "Vanves", "Ville-d'Avray"
        }

        genai.configure(api_key=gemini_api_key)

        # Modèles Gemini (inchangés)
        generation_config = { "temperature": 0.2, "top_p": 0.8, "top_k": 40, "max_output_tokens": 4096 }
        safety_settings = { HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE, HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE, HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE, HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE }
        self.gemini_model = genai.GenerativeModel("gemini-1.5-flash", generation_config=generation_config, safety_settings=safety_settings)
        natural_generation_config = { "temperature": 0.7, "top_p": 0.9, "top_k": 40, "max_output_tokens": 4096 }
        self.gemini_natural_model = genai.GenerativeModel("gemini-1.5-flash", generation_config=natural_generation_config, safety_settings=safety_settings)
        knowledge_generation_config = { "temperature": 0.5, "top_p": 0.9, "top_k": 40, "max_output_tokens": 4096 }
        self.gemini_knowledge_model = genai.GenerativeModel("gemini-1.5-flash", generation_config=knowledge_generation_config, safety_settings=safety_settings)

        self.check_api_connection()
        self.initialize_knowledge_base()

    # --- (check_api_connection unchanged) ---
    def check_api_connection(self):
        """
        Vérifie la connexion à l'API et récupère des informations sur la structure des données
        """

        try:
            # URL de l'API (assure-toi que self.api_base_url est bien défini)
            api_url = f"{self.api_base_url}?page=1&page_size=1"  # Ajout de params pour limiter la charge

            # Requête avec timeout pour éviter de bloquer l'exécution
            response = requests.get(api_url, timeout=10)
            response.raise_for_status()  # Lève une erreur si le statut HTTP est mauvais (ex: 404, 500)

            # Vérification du format JSON
            try:
                data = response.json()
            except requests.exceptions.JSONDecodeError:
                raise ValueError("La réponse de l'API n'est pas un JSON valide.")

            # Vérification que la réponse contient bien des données
            if "data" in data and isinstance(data["data"], list): # Removed len > 0 check to handle empty results gracefully
                print("✅ Connexion à l'API Établissements réussie.")
                if len(data["data"]) > 0 and isinstance(data["data"][0], dict):
                     print(f"🔹 Structure des données Établissements : {list(data['data'][0].keys())}")
                else:
                    print("🔹 Connexion réussie, mais la première page de données est vide ou mal formée.")
            else:
                print("⚠️ Connexion à l'API Établissements réussie, mais structure de réponse inattendue (pas de clé 'data' ou pas une liste).")


        except requests.exceptions.RequestException as e:
            print(f"❌ Erreur de connexion à l'API Établissements : {e}")
            raise
        except Exception as e:
            print(f"❌ Une erreur inattendue s'est produite lors de la vérification de l'API Établissements : {e}")
            raise

    # --- (initialize_knowledge_base unchanged) ---
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

    # --- (classify_query_type unchanged) ---
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
            # Safer text extraction
            response_text = ""
            if response.parts:
                response_text = "".join(part.text for part in response.parts if hasattr(part, 'text')).strip().lower()
            elif hasattr(response, 'text'):
                 response_text = response.text.strip().lower()


            if "establishment_search" in response_text:
                return "establishment_search"
            elif "general_info" in response_text:
                return "general_info"
            else:
                # If Gemini doesn't classify clearly, check for keywords as a fallback
                keywords = ["restaurant", "musée", "mairie", "bibliothèque", "piscine", "magasin", "où", "trouver", "chercher", "adresse", "lieu", "place"]
                # Check query case-insensitively for keywords
                query_lower = query.lower()
                if any(keyword in query_lower for keyword in keywords):
                    print("Fallback: Classified as establishment_search based on keywords.")
                    return "establishment_search"
                print(f"Warning: Could not classify query type reliably. Gemini response: '{response_text}'. Defaulting to off_topic.")
                return "off_topic" # More conservative default
        except Exception as e:
            print(f"Erreur lors de la classification de la requête: {e}")
            return "off_topic" # Safer default if classification fails

    # --- (rank_establishments_by_embedding unchanged) ---
    def rank_establishments_by_embedding(self, establishments, query, top_n=5):
        """
        Classe les établissements en fonction de la similarité sémantique avec la requête utilisateur.

        Args:
            establishments (list): Liste des établissements retournés par l'API.
            query (str): Requête utilisateur originale.
            top_n (int): Nombre maximal d'établissements à retourner.

        Returns:
            list: Établissements classés par ordre de similarité sémantique.
        """
        if not establishments:
            return []

        # Ensure establishments is a list of dicts
        if not isinstance(establishments, list) or not all(isinstance(e, dict) for e in establishments):
             print("Warning: rank_establishments_by_embedding received non-list or non-dict elements.")
             return []


        # Embedding requête utilisateur
        try:
            query_embedding = self.model.encode(query, convert_to_tensor=True)
        except Exception as e:
            print(f"Error encoding query: {e}")
            return [] # Cannot rank without query embedding


        # Embedding enrichi des établissements
        establishment_texts = []
        valid_establishments_indices = []
        for i, e in enumerate(establishments):
            # Basic check for dict type again, just in case
            if not isinstance(e, dict):
                print(f"Skipping non-dict element at index {i} in establishments list.")
                continue

            text_parts = [
                e.get('name', ''),
                e.get('activite', ''),
                e.get('commentaire', ''),
                e.get('categorie', ''),
                # Adresse fields might vary based on API response structure
                e.get('adresse', ''), # Assuming a combined 'adresse' field exists
                e.get('voie', ''),
                e.get('commune', ''),
                e.get('code_postal', '') # Check if this field exists in your API response
            ]
            # Ensure all parts are strings before joining
            establishment_text = ' '.join([str(part) for part in text_parts if part is not None and str(part).strip()])

            if establishment_text: # Only include if there's some text
                 establishment_texts.append(establishment_text)
                 valid_establishments_indices.append(i)
            else:
                 print(f"Skipping establishment at index {i} due to empty text representation.")


        if not establishment_texts:
            print("No valid text representations found for establishments.")
            return []

        try:
            establishment_embeddings = self.model.encode(establishment_texts, convert_to_tensor=True)
        except Exception as e:
            print(f"Error encoding establishments: {e}")
            return [] # Cannot rank without establishment embeddings


        # Calcul similarité cosinus
        try:
            # Ensure embeddings are on CPU and correctly shaped
            query_emb_cpu = query_embedding.cpu().numpy().reshape(1, -1)
            estab_emb_cpu = establishment_embeddings.cpu().numpy()

            if query_emb_cpu.shape[1] != estab_emb_cpu.shape[1]:
                 print(f"Embedding dimension mismatch: Query {query_emb_cpu.shape}, Establishments {estab_emb_cpu.shape}")
                 return []


            similarities = cosine_similarity(query_emb_cpu, estab_emb_cpu)[0]

        except Exception as e:
             print(f"Error calculating cosine similarity: {e}")
             return []


        # Trier selon similarité cosinus
        # argsort returns indices of the *original* `similarities` array
        sorted_similarity_indices = np.argsort(similarities)[::-1]

        # Map these sorted indices back to the *original* `establishments` list indices
        ranked_original_indices = [valid_establishments_indices[i] for i in sorted_similarity_indices]


        # Sélectionner les meilleurs résultats selon embeddings
        ranked_establishments = [establishments[i] for i in ranked_original_indices[:top_n]]

        return ranked_establishments

    # --- (generate_hypothetical_document_with_gemini unchanged) ---
    def generate_hypothetical_document_with_gemini(self, query):
        """
        Génère un document hypothétique basé sur la requête de l'utilisateur en utilisant Gemini.
        Args:
            query (str): La requête de l'utilisateur
        Returns:
            str: Document hypothétique généré (texte descriptif)
            dict: Critères extraits de la requête (JSON/dictionnaire)
        """
        prompt = f"""
        Analyse la requête utilisateur suivante et extrait les critères de recherche pertinents pour un établissement public en France. IMPORTANT: Conserve la capitalisation exacte pour les noms de communes.

        Requête: "{query}"

        Crée un document hypothétique au format JSON avec uniquement les champs suivants si pertinents:
        - commune: le nom EXACT de la ville ou commune (ex: "Boulogne-Billancourt", "Issy-les-Moulineaux"). Si plusieurs communes sont mentionnées, choisis la plus probable ou la première. Ne modifie PAS la capitalisation. Si la requête mentionne "Paris 15" ou similaire, extraire "Paris".
        - activite: le type d'établissement (restaurant, musée, mairie, etc.). Si rien n'est mentionné, laisse ce champ vide. Si un type lié à la restauration est mentionné (pizzeria, fast-food, brasserie), utilise "restaurant".
        - entree_pmr: true si l'accessibilité PMR générale ou l'accès en fauteuil roulant est mentionné.
        - stationnement_pmr: true si le stationnement PMR ou parking adapté est mentionné.
        - stationnement_presence: true si le stationnement ou parking est mentionné (même sans mention PMR).
        - sanitaires_presence: true si des sanitaires ou toilettes sont mentionnés.
        - sanitaires_adaptes: true si des sanitaires adaptés ou toilettes PMR sont mentionnés.
        - accueil_equipements_malentendants_presence: true si des équipements pour malentendants (boucle magnétique, etc.) sont mentionnés.
        - accueil_audiodescription_presence: true si l'audiodescription est mentionnée.
        - cheminement_ext_bande_guidage: true si des bandes de guidage podotactiles sont mentionnées.
        - cheminement_ext_plain_pied: true si l'accès de plain-pied est mentionné.
        - transport_station_presence: true si la proximité des transports en commun (métro, bus, gare) est mentionnée.
        - google_maps_link_needed: true si la requête suggère un besoin de localisation précise (ex: "près de la gare", "comment y aller"). Mets à false par défaut.

        Retourne uniquement le JSON valide, sans aucun autre texte avant ou après.
        Si la requête ne semble pas concerner la recherche d'un établissement spécifique (par exemple, une question générale), retourne un JSON vide {{}}.
        Assure-toi que le JSON est correctement formaté.
        """

        try:
            response = self.gemini_model.generate_content(prompt)

            # Robust extraction of JSON block
            response_text = ""
            if response.parts:
                response_text = "".join(part.text for part in response.parts if hasattr(part, 'text'))
            elif hasattr(response, 'text'):
                 response_text = response.text

            match = re.search(r'```json\s*(\{.*?\})\s*```|\{.*?\}', response_text, re.DOTALL)
            if not match:
                 print(f"Warning: Could not extract JSON from Gemini response: {response_text}")
                 return "", {}

            json_text = match.group(1) if match.group(1) else match.group(0)
            json_text = json_text.strip()

            criteria = json.loads(json_text)

            # --- Data Cleaning and Normalization ---
            # Handle "Paris X" -> "Paris" but keep other capitalizations
            if 'commune' in criteria and isinstance(criteria['commune'], str):
                 commune_val = criteria['commune'].strip()
                 # Normalize Paris variants specifically
                 if commune_val.startswith("Paris"):
                     criteria['commune'] = "Paris"
                 else:
                      # Keep original capitalization for other cities
                      criteria['commune'] = commune_val
            elif 'commune' in criteria and criteria['commune'] is None:
                 del criteria['commune']

            # Ensure boolean fields are actual booleans
            boolean_fields = [
                 'entree_pmr', 'stationnement_pmr', 'stationnement_presence',
                 'sanitaires_presence', 'sanitaires_adaptes',
                 'accueil_equipements_malentendants_presence', 'accueil_audiodescription_presence',
                 'cheminement_ext_bande_guidage', 'cheminement_ext_plain_pied',
                 'transport_station_presence', 'google_maps_link_needed'
            ]
            for field in boolean_fields:
                 if field in criteria:
                     if isinstance(criteria[field], str):
                         criteria[field] = criteria[field].lower() == 'true'
                     elif not isinstance(criteria[field], bool):
                          print(f"Warning: Unexpected type for boolean field '{field}': {type(criteria[field])}. Setting to False.")
                          criteria[field] = False

            # Clean up criteria (remove false/empty values, keep commune/activite)
            criteria_cleaned = {}
            for key, value in criteria.items():
                 is_bool_field = key in boolean_fields
                 if key in ['commune', 'activite']:
                      if value is not None: # Keep even if empty string
                          criteria_cleaned[key] = value
                 elif is_bool_field and value is True:
                     criteria_cleaned[key] = value
                 elif not is_bool_field and value:
                     criteria_cleaned[key] = value

            # Create descriptive text (for debugging)
            doc_text = "Critères extraits pour la recherche :\n"
            if criteria_cleaned:
                 for key, value in criteria_cleaned.items():
                    doc_text += f"- {key}: {value}\n"
            else:
                 doc_text += "(Aucun critère spécifique identifié)\n"

            return doc_text.strip(), criteria_cleaned

        except json.JSONDecodeError as e:
             print(f"Erreur de décodage JSON depuis Gemini: {e}")
             print(f"Réponse brute de Gemini: {response_text}")
             return "Erreur: Impossible d'analyser les critères.", {}
        except Exception as e:
            print(f"Erreur lors de l'appel à Gemini pour l'extraction de critères: {e}")
            error_details = ""
            try:
                if response and response.prompt_feedback:
                    error_details = f" Prompt Feedback: {response.prompt_feedback}"
            except Exception: pass
            print(f"Réponse brute (si disponible): {response_text if 'response_text' in locals() else 'N/A'}{error_details}")
            return "Erreur lors de l'analyse de la requête.", {}


    # --- (search_all_establishments unchanged) ---
    def search_all_establishments(self, criteria, max_pages=10):
        """
        Recherche TOUS les établissements correspondant aux critères extraits via l'API en gérant la pagination.

        Args:
            criteria (dict): Critères de recherche.
            max_pages (int): Nombre maximal de pages à parcourir.

        Returns:
            list: Liste complète des établissements correspondants.
        """
        all_establishments = []
        page = 1
        # Use a reasonable page size supported by the API, default 100 if unknown
        page_size = criteria.get('page_size', 100)


        while page <= max_pages:
            query_params = []

            # Add mandatory pagination parameters first
            query_params.append(f"page={page}")
            query_params.append(f"page_size={page_size}")


            # Add filters based on criteria (ensure keys match API expectations)
            # Commune: Use exact match if specified - Use quote for URL encoding
            if criteria.get('commune'):
                 query_params.append(f"commune__exact={quote(str(criteria['commune']))}")


            # Activite: Use contains match if specified - Use quote
            if criteria.get('activite'):
                 query_params.append(f"activite__contains={quote(str(criteria['activite']))}")


            # Boolean fields: Add only if True
            boolean_fields_api = { # Map criteria keys to potential API field names
                 'entree_pmr': 'entree_pmr',
                 'stationnement_pmr': 'stationnement_pmr',
                 'stationnement_presence': 'stationnement_presence',
                 'sanitaires_presence': 'sanitaires_presence',
                 'sanitaires_adaptes': 'sanitaires_adaptes',
                 'accueil_equipements_malentendants_presence': 'accueil_equipements_malentendants_presence',
                 'accueil_audiodescription_presence': 'accueil_audiodescription_presence',
                 'cheminement_ext_bande_guidage': 'cheminement_ext_bande_guidage',
                 'cheminement_ext_plain_pied': 'cheminement_ext_plain_pied',
                 'transport_station_presence': 'transport_station_presence'
            }


            for criteria_key, api_field in boolean_fields_api.items():
                 if criteria.get(criteria_key) is True:
                     query_params.append(f"{api_field}__exact=true")


            # Construct the final URL
            api_url = f"{self.api_base_url}?{'&'.join(query_params)}"
            print(f"Appel API Établissements (page {page}): {api_url}")


            try:
                response = requests.get(api_url, timeout=20)
                response.raise_for_status()

                if 'application/json' not in response.headers.get('Content-Type', ''):
                     print(f"Erreur: Réponse de l'API Établissements n'est pas du JSON. Status: {response.status_code}. Contenu: {response.text[:200]}")
                     break

                data = response.json()
                establishments_page = data.get('data', [])

                if not isinstance(establishments_page, list):
                     print(f"Erreur: La clé 'data' de l'API Établissements ne contient pas une liste. Reçu: {type(establishments_page)}")
                     break

                if not establishments_page:
                     print(f"Aucun établissement trouvé sur la page {page}. Fin de la recherche.")
                     break

                all_establishments.extend(establishments_page)

                if len(establishments_page) < page_size:
                     print("Dernière page atteinte (moins de résultats que page_size).")
                     break

                page += 1

            except requests.exceptions.HTTPError as http_err:
                print(f"Erreur HTTP lors de la recherche API Établissements (page {page}): {http_err}")
                break
            except requests.exceptions.ConnectionError as conn_err:
                print(f"Erreur de Connexion lors de la recherche API Établissements (page {page}): {conn_err}")
                break
            except requests.exceptions.Timeout as timeout_err:
                print(f"Timeout lors de la recherche API Établissements (page {page}): {timeout_err}")
                break
            except requests.exceptions.RequestException as req_err:
                print(f"Erreur Requête lors de la recherche API Établissements (page {page}): {req_err}")
                break
            except json.JSONDecodeError as json_err:
                print(f"Erreur de décodage JSON de l'API Établissements (page {page}): {json_err}")
                print(f"Contenu brut reçu (début): {response.text[:200]}")
                break
            except Exception as e:
                 print(f"Erreur inattendue lors du traitement de la page {page} de l'API Établissements: {e}")
                 break

        print(f"Nombre total d'établissements trouvés après pagination : {len(all_establishments)}")
        return all_establishments

    # --- (_calculate_distance unchanged) ---
    def _calculate_distance(self, lat1, lon1, lat2, lon2):
        """
        Calcule la distance approximative en kilomètres entre deux points GPS
        en utilisant la formule de Haversine.
        """
        R = 6371  # Rayon de la Terre en km
        dLat = math.radians(lat2 - lat1)
        dLon = math.radians(lon2 - lon1)
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)

        a = math.sin(dLat / 2) * math.sin(dLat / 2) + \
            math.cos(lat1_rad) * math.cos(lat2_rad) * \
            math.sin(dLon / 2) * math.sin(dLon / 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = R * c
        return distance

    # --- (search_pmr_parking unchanged) ---
    def search_pmr_parking(self, latitude, longitude, max_distance_km=2.0, limit=100):
        """
        Recherche les places de parking PMR à proximité d'un point donné
        en utilisant l'API Seine Ouest et filtre par distance.
        Returns: list of dicts with 'lat', 'lon', 'localisation', etc.
        """
        nearby_parking_spots = []
        params = {'limit': limit}
        api_url = f"{self.pmr_parking_api_url}"

        print(f"Appel API Parking PMR (Seine Ouest): {api_url} avec limit={limit}")

        try:
            response = requests.get(api_url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            if 'results' in data and isinstance(data['results'], list):
                all_spots = data['results']
                print(f"Reçu {len(all_spots)} places PMR de l'API Seine Ouest. Filtrage par distance...")

                for spot in all_spots:
                    geo_point = spot.get('geo_point_2d')
                    # Ensure coordinates are present and valid
                    if isinstance(geo_point, dict) and 'lat' in geo_point and 'lon' in geo_point:
                        try:
                            spot_lat = float(geo_point['lat'])
                            spot_lon = float(geo_point['lon'])
                            distance = self._calculate_distance(latitude, longitude, spot_lat, spot_lon)

                            if distance <= max_distance_km:
                                spot_info = {
                                    'localisation': spot.get('localisation', 'Adresse inconnue'),
                                    'commune': spot.get('commune', ''),
                                    'num_loc': spot.get('num_loc', ''),
                                    'comp_loc': spot.get('comp_loc', ''),
                                    'distance_km': round(distance, 2),
                                    'lat': spot_lat, # Keep lat
                                    'lon': spot_lon  # Keep lon
                                }
                                nearby_parking_spots.append(spot_info)
                        except (ValueError, TypeError) as e:
                            print(f"Warning: Impossible de traiter les coordonnées pour la place PMR ID {spot.get('id_pmr', 'inconnu')}: {e} - Data: {geo_point}")
                            continue # Skip this spot if coords are bad

                nearby_parking_spots.sort(key=lambda x: x['distance_km'])
                print(f"Trouvé {len(nearby_parking_spots)} places PMR à moins de {max_distance_km} km.")
            else:
                print("Warning: La réponse de l'API Parking PMR n'a pas la structure attendue.")

        except requests.exceptions.RequestException as e:
            print(f"Erreur lors de l'appel à l'API Parking PMR: {e}")
        except json.JSONDecodeError as e:
            print(f"Erreur de décodage JSON de l'API Parking PMR: {e}")
            print(f"Réponse brute: {response.text[:200]}")
        except Exception as e:
            print(f"Erreur inattendue lors de la recherche de parking PMR: {e}")

        return nearby_parking_spots


    # --- *** MODIFIED Function: generate_natural_response *** ---
    def generate_natural_response(self, establishments, query, criteria):
        """
        Génère une réponse naturelle aux résultats trouvés en utilisant Gemini,
        en incluant potentiellement les informations de parking PMR (avec liens Maps)
        pour les communes de Seine Ouest.
        """
        # Input validation (unchanged)
        if not isinstance(establishments, list) or not all(isinstance(e, dict) for e in establishments):
             if isinstance(establishments, dict) and 'data' in establishments and isinstance(establishments['data'], list):
                 establishments = establishments['data']
             else:
                 print("Warning: generate_natural_response received invalid 'establishments' format.")
                 establishments = [] if not isinstance(establishments, list) else [e for e in establishments if isinstance(e, dict)]

        # Handle no results found (unchanged)
        if not establishments:
            prompt = f"""
            Tu es un assistant spécialisé dans la recherche d'établissements publics accessibles en France.
            L'utilisateur a demandé: "{query}"
            J'ai cherché dans ma base de données avec les critères suivants: {json.dumps(criteria, ensure_ascii=False, indent=2)}
            Mais je n'ai trouvé aucun établissement correspondant.
            Génère une réponse empathique, naturelle et en français qui:
            1. Informe l'utilisateur qu'aucun établissement n'a été trouvé pour sa recherche '{query}'.
            2. Mentionne spécifiquement la commune ({criteria.get('commune', 'lieu non précisé')}) et le type d'activité ({criteria.get('activite', 'non spécifié')}) si disponibles.
            3. Suggère de reformuler, d'élargir les critères ou de vérifier l'orthographe.
            4. Garde un ton conversationnel et serviable.
            Écris ta réponse directement à l'utilisateur, sans introduction/conclusion artificielle.
            """
        # Handle results found
        else:
            establishments_data = []
            commune_recherchee = criteria.get('commune', '')
            is_seine_ouest_query = commune_recherchee in self.seine_ouest_cities

            for i, estab in enumerate(establishments, 1):
                 if not isinstance(estab, dict): continue
                 estab_info = { "id": f"etab_{i}", "nom": estab.get('name', 'N/A'), "activite": estab.get('activite', 'N/S'), }

                 # Address processing (unchanged)
                 address_parts = []
                 if estab.get('adresse'): address_parts.append(estab['adresse'])
                 else:
                    if estab.get('numero'): address_parts.append(str(estab['numero']))
                    if estab.get('voie'): address_parts.append(estab['voie'])
                    if estab.get('code_postal'): address_parts.append(str(estab['code_postal']))
                    if estab.get('commune'): address_parts.append(estab['commune'])
                 estab_info["adresse"] = ' '.join(filter(None, address_parts)).strip() or "N/A"

                 # Accessibility processing (unchanged)
                 access_features = []
                 access_map = { 'entree_pmr': "Entrée PMR", 'stationnement_pmr': "Stationnement PMR", 'sanitaires_adaptes': "Sanitaires adaptés", 'accueil_equipements_malentendants_presence': "Équipements malentendants", 'accueil_audiodescription_presence': "Audiodescription", 'cheminement_ext_bande_guidage': "Bandes guidage ext.", 'cheminement_ext_plain_pied': "Plain-pied ext." }
                 for key, text in access_map.items():
                      if estab.get(key) is True: access_features.append(text)
                 if estab.get('stationnement_presence') is True and not estab.get('stationnement_pmr'): access_features.append("Stationnement présent")
                 if estab.get('sanitaires_presence') is True and not estab.get('sanitaires_adaptes'): access_features.append("Sanitaires présents")
                 estab_info["accessibilite"] = access_features if access_features else ["Infos accessibilité N/S"]

                 # Contact processing (unchanged)
                 contact_info = {}
                 if estab.get('contact_url'): contact_info["contact_principal"] = estab['contact_url']
                 if estab.get('site_internet'): contact_info["site_web"] = estab['site_internet']
                 if estab.get('web_url'): contact_info["page_web"] = estab['web_url']
                 latitude, longitude = estab.get('latitude'), estab.get('longitude')
                 if latitude and longitude and criteria.get('google_maps_link_needed'):
                     try:
                         lat_f, lon_f = float(latitude), float(longitude)
                         link = f"https://www.google.com/maps/search/?api=1&query={lat_f},{lon_f}"
                         contact_info["google_maps_link"] = f"[Voir sur Google Maps]({link})" # Markdown link
                     except (ValueError, TypeError): pass
                 estab_info["contact"] = contact_info if contact_info else None

                 # --- *** MODIFIED: Add PMR Parking Info with Clickable Links *** ---
                 if 'nearby_parking_pmr' in estab and estab['nearby_parking_pmr']:
                    parking_spots = estab['nearby_parking_pmr']
                    parking_texts_md = [] # Store Markdown formatted links
                    for spot in parking_spots[:3]: # Show top 3
                        loc = spot.get('localisation','Adresse inconnue')
                        num = spot.get('num_loc','')
                        dist = spot.get('distance_km','?')
                        commune_spot = spot.get('commune','')
                        lat_spot = spot.get('lat') # Get parking lat
                        lon_spot = spot.get('lon') # Get parking lon

                        # Construct address string for display
                        parking_addr_text = f"{num} {loc}".strip()
                        if commune_spot and commune_spot != estab.get('commune'):
                            parking_addr_text += f" ({commune_spot})"

                        # Construct Google Maps Link if coordinates are valid
                        maps_link = None
                        if lat_spot is not None and lon_spot is not None:
                            try:
                                # Ensure they are floats before formatting
                                lat_f_spot = float(lat_spot)
                                lon_f_spot = float(lon_spot)
                                maps_link = f"https://www.google.com/maps/search/?api=1&query={lat_f_spot},{lon_f_spot}"
                            except (ValueError, TypeError):
                                print(f"Warning: Invalid coordinates for parking spot link: {lat_spot}, {lon_spot}")
                                maps_link = None # Invalidate link if coords bad

                        # Create Markdown link if possible, otherwise just text
                        if maps_link:
                            parking_md = f"- [{parking_addr_text}](<{maps_link}>) (à {dist} km)" # Use <URL> for robustness
                        else:
                            parking_md = f"- {parking_addr_text} (à {dist} km, lien Maps indisponible)"

                        parking_texts_md.append(parking_md)

                    if parking_texts_md:
                        estab_info["parking_pmr_proximite_md"] = parking_texts_md # Use a distinct key for MD links
                        if len(parking_spots) > 3:
                            estab_info["parking_pmr_proximite_md"].append(f"- et {len(parking_spots)-3} autre(s) plus loin.")
                 # --- *** End MODIFIED Parking Info *** ---
                 establishments_data.append(estab_info)

            # --- *** MODIFIED Prompt *** ---
            prompt = f"""
            Tu es un assistant conversationnel spécialisé dans l'accessibilité des lieux publics en France. Tu génères des réponses en Markdown.

            L'utilisateur a posé la question : "{query}"
            Critères identifiés : {json.dumps(criteria, ensure_ascii=False, indent=2)}
            Résultats trouvés ({len(establishments_data)}) dans la commune '{commune_recherchee}'.

            Détails des établissements (format JSON):
            {json.dumps(establishments_data, ensure_ascii=False, indent=2)}

            Génère une réponse naturelle, conversationnelle et bienveillante en français (format Markdown) qui :
            1. Confirme la trouvaille de résultats pour la commune ({commune_recherchee}).
            2. Présente chaque établissement clairement :
                - Nom et adresse complète.
                - Caractéristiques d'accessibilité (champ 'accessibilite').
                - **Parking PMR (IMPORTANT)**: Si le champ **'parking_pmr_proximite_md'** existe et contient des éléments, affiche chaque élément de cette liste. Ces éléments sont déjà formatés en Markdown avec des liens cliquables vers Google Maps. **Préserve IMPÉRATIVEMENT ce format Markdown `[Texte](<URL>)` pour les liens.** Précise que ce sont des places à proximité gérées par Seine Ouest.
                - Informations de contact (champ 'contact'). Si 'google_maps_link' est présent dans contact, affiche-le (il est déjà en Markdown).
            3. Utilise des transitions fluides.
            4. Ton empathique, utile, non robotique.
            5. Ne te présente PAS comme une IA.

            Rédige la réponse finale directement pour l'utilisateur en **Markdown**.
            """

        # Generate response using Gemini
        try:
            response = self.gemini_natural_model.generate_content(prompt)
            # Gemini 1.5 Flash generally outputs Markdown well when instructed.
            final_response_text = "".join(part.text for part in response.parts if hasattr(part, 'text')) if response.parts else (response.text if hasattr(response, 'text') else "")

            # --- Add Disclaimer Logic (using the new key) ---
            parking_data_present_md = any(
                isinstance(e, dict) and 'parking_pmr_proximite_md' in e and e['parking_pmr_proximite_md']
                for e in establishments_data # Check the data prepared for the prompt
            )

            # --- End Disclaimer ---

            return final_response_text.strip()

        # Handle Gemini errors (unchanged)
        except Exception as e:
            print(f"Erreur lors de la génération de la réponse naturelle: {e}")
            error_details = ""
            try:
                if response and response.prompt_feedback: error_details = f" Prompt Feedback: {response.prompt_feedback}"
            except Exception: pass
            print(f"Réponse brute (si disponible): {response.text if 'response' in locals() and hasattr(response, 'text') else 'N/A'}{error_details}")

            if establishments:
                names = [e.get('name', f'E{i+1}') for i, e in enumerate(establishments) if isinstance(e, dict)]
                if names: return f"J'ai trouvé {len(names)} établissement(s) ({', '.join(names)}), mais je n'arrive pas à formater les détails pour le moment."
                else: return "J'ai trouvé des infos, mais je n'arrive pas à les présenter. Veuillez réessayer."
            else:
                return "Désolé, je n'ai pas trouvé d'établissements pour votre demande. Essayez de reformuler ?"

    # --- (generate_knowledge_response unchanged) ---
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
            acronym_match = re.search(r'\(([A-Z]+)\)', aide)
            acronym = acronym_match.group(1) if acronym_match else None
            aide_base_name = aide.split('(')[0].strip().lower()
            if aide_lower in query_lower or aide_base_name in query_lower or (acronym and acronym.lower() in query_lower.split()):
                relevant_keywords.append(aide)

        # Vérification des organismes mentionnés
        for organisme in self.knowledge_base["organismes"]:
            organisme_lower = organisme.lower()
            acronym_match = re.search(r'\(([A-Z]+)\)', organisme)
            acronym = acronym_match.group(1) if acronym_match else None
            organisme_base_name = organisme.split('(')[0].strip().lower()
            if organisme_lower in query_lower or organisme_base_name in query_lower or (acronym and acronym.lower() in query_lower.split()):
                relevant_keywords.append(organisme)

        # Vérification des droits mentionnés
        for droit in self.knowledge_base["droits_accessibilite"]:
            droit_lower = droit.lower()
            acronym_match = re.search(r'\(([A-Z]+)\)', droit)
            acronym = acronym_match.group(1) if acronym_match else None
            droit_base_name = droit.split('(')[0].strip().lower()
            if droit_lower in query_lower or droit_base_name in query_lower or (acronym and acronym.lower() in query_lower.split()):
                relevant_keywords.append(droit)

        relevant_keywords = sorted(list(set(relevant_keywords)))

        # Création du prompt pour Gemini
        prompt = f"""
        Tu es un assistant spécialisé dans l'information sur le handicap et l'accessibilité en France. Ton rôle est de fournir des informations claires, précises et empathiques.
        L'utilisateur a posé la question suivante: "{query}"
        Contexte général (pour toi):
        - Types handicap: {', '.join(self.knowledge_base["types_handicap"])}
        - Aides financières: {', '.join(self.knowledge_base["aides_financieres"])}
        - Organismes clés: {', '.join(self.knowledge_base["organismes"])} (MDPH est central).
        - Droits: {', '.join(self.knowledge_base["droits_accessibilite"])}
        Mots-clés pertinents identifiés: {', '.join(relevant_keywords) if relevant_keywords else "Aucun"}

        Génère une réponse en français qui:
        1. Répond directement et précisément à la question.
        2. Fournit des infos factuelles et utiles.
        3. Mentionne aides/organismes/droits si pertinent. Oriente vers organismes officiels (MDPH) pour détails/démarches.
        4. Utilise un ton bienveillant, respectueux, inclusif, non technique/infantilisant.
        5. Si question très spécifique (montants, procédures), explique variabilité et conseille sources officielles/organismes compétents (nomme-les si possible).
        6. Reste objectif/informatif. Pas de conseils médicaux/juridiques perso.
        7. Structure clairement si plusieurs points.
        Écris directement la réponse à l'utilisateur, sans salutations/conclusions génériques.
        """

        try:
            response = self.gemini_knowledge_model.generate_content(prompt)
            response_text = "".join(part.text for part in response.parts if hasattr(part, 'text')) if response.parts else (response.text if hasattr(response, 'text') else "")
            return response_text.strip()
        except Exception as e:
            print(f"Erreur lors de la génération de la réponse sur les connaissances: {e}")
            error_details = ""
            try:
                 if response and response.prompt_feedback: error_details = f" Prompt Feedback: {response.prompt_feedback}"
            except Exception: pass
            print(f"Réponse brute (si disponible): {response.text if 'response' in locals() and hasattr(response, 'text') else 'N/A'}{error_details}")
            return "Je suis désolé, je rencontre un problème pour répondre. Pour des informations fiables sur le handicap en France, contactez la MDPH de votre département."


    # --- (process_query unchanged) ---
    def process_query(self, query):
        """
        Traite la requête de l'utilisateur, effectue les recherches nécessaires
        (y compris le parking PMR pour les communes de Seine Ouest) et renvoie une réponse.
        """
        # 1. Classification
        query_type = self.classify_query_type(query)
        print(f"Type de requête détecté: {query_type}")

        # 2. Traitement par type
        if query_type == "off_topic":
            return "Je suis là pour vous aider à trouver des lieux accessibles en France et répondre à des questions sur le handicap. Comment puis-je vous assister dans ce cadre ?"

        elif query_type == "general_info":
            return self.generate_knowledge_response(query)

        elif query_type == "establishment_search":
            # 3. Extraction Critères
            hypothetical_doc_desc, criteria = self.generate_hypothetical_document_with_gemini(query)
            print("Document hypothétique (description générée):\n", hypothetical_doc_desc)
            print("Critères extraits (dictionnaire):", criteria)

            # Vérification Critères
            if not criteria or not criteria.get('commune'):
                clarification_prompt = f"""
                L'utilisateur a demandé "{query}", mais je n'ai pas pu identifier clairement la ville ou le type de lieu.
                Demande poliment à l'utilisateur de préciser la commune (ville) et si possible le type de lieu recherché.
                Réponds directement à l'utilisateur.
                """
                try:
                    clarification_response = self.gemini_natural_model.generate_content(clarification_prompt)
                    clarification_text = "".join(part.text for part in clarification_response.parts if hasattr(part, 'text')) if clarification_response.parts else (clarification_response.text if hasattr(clarification_response, 'text') else "")
                    return clarification_text.strip() or "Pourriez-vous préciser la ville et le type de lieu svp ?"
                except Exception as e_clarify:
                     print(f"Erreur lors de la demande de clarification: {e_clarify}")
                     return "Je ne suis pas sûr d'avoir bien compris. Pourriez-vous préciser la ville et le type de lieu ?"

            # 4. Recherche Établissements
            establishments = self.search_all_establishments(criteria, max_pages=5)
            if not establishments:
                 return self.generate_natural_response([], query, criteria)

            # 5. Classement Sémantique
            ranked_establishments = self.rank_establishments_by_embedding(establishments, query, top_n=3)
            if not ranked_establishments:
                 print("Aucun établissement pertinent trouvé après classement sémantique.")
                 criteria['ranking_failed'] = True
                 return self.generate_natural_response([], query, criteria)

            # 6. TOOL ACTIVATION: Recherche Parking PMR
            commune_recherchee = criteria.get('commune', '')
            is_seine_ouest_query = commune_recherchee in self.seine_ouest_cities

            if is_seine_ouest_query:
                print(f"\nINFO: Recherche de parking PMR activée (commune: {commune_recherchee}).")
                establishments_with_parking = []
                for estab in ranked_establishments:
                     if isinstance(estab, dict) and 'latitude' in estab and 'longitude' in estab:
                         try:
                             lat, lon = float(estab['latitude']), float(estab['longitude'])
                             print(f"  -> Recherche parking PMR près de '{estab.get('name', 'N/A')}' ({lat}, {lon})")
                             nearby_parking = self.search_pmr_parking(lat, lon, max_distance_km=2.0, limit=50)
                             estab['nearby_parking_pmr'] = nearby_parking # Store raw parking data
                             if nearby_parking: print(f"     Trouvé {len(nearby_parking)} places PMR.")
                             else: print("     Aucune place PMR trouvée.")
                             establishments_with_parking.append(estab)
                         except (ValueError, TypeError) as e:
                             print(f"  -> Skipping parking search for '{estab.get('name', 'N/A')}' (coords invalides): {e}")
                             estab['nearby_parking_pmr'] = []
                             establishments_with_parking.append(estab)
                     else:
                          print(f"  -> Skipping parking search for '{estab.get('name', 'N/A')}' (coords manquantes/invalides).")
                          if isinstance(estab, dict): estab['nearby_parking_pmr'] = []
                          establishments_with_parking.append(estab)
                ranked_establishments = establishments_with_parking
            else:
                 print(f"\nINFO: Recherche de parking PMR non activée (commune '{commune_recherchee}' hors Seine Ouest).")

            # 7. Génération Réponse Finale
            response = self.generate_natural_response(ranked_establishments, query, criteria)
            return response

        else:
            print(f"Warning: Type de requête inattendu '{query_type}'.")
            return "Je ne suis pas sûr de comprendre. Pouvez-vous reformuler ?"

# --- (Keep the __main__ block for testing) ---
if __name__ == "__main__":
    gemini_api_key = "" # Replace placeholder
    if not gemini_api_key or gemini_api_key == "YOUR_FALLBACK_API_KEY_HERE":
         # Use a real key or raise error
         # raise ValueError("Erreur: Clé API Gemini non configurée. Voir variable d'env GEMINI_API_KEY.")
         gemini_api_key = "" # Using the key provided earlier for now
         print("Warning: Using hardcoded API Key for testing.")


    api_base_url = "https://tabular-api.data.gouv.fr/api/resources/93ae96a7-1db7-4cb4-a9f1-6d778370b640/data/"

    try:
        print("Initialisation du Chatbot...")
        chatbot = ChatbotInclusifGemini(api_base_url, gemini_api_key)
        print("\n✅ Chatbot Inclusif initialisé.")
        print("   - API Établissements:", api_base_url)
        print("   - API Parking PMR (Seine Ouest):", chatbot.pmr_parking_api_url)
        print(f"   - Communes Seine Ouest pour parking PMR: {chatbot.seine_ouest_cities}")
        print("\nPosez votre question (ou tapez 'quit' pour quitter):")

        while True:
            user_query = input("\nVotre question: ")
            if user_query.lower() in ['quit', 'exit', 'q']:
                print("Au revoir !")
                break
            if not user_query.strip(): continue

            print("\nTraitement de la requête...")
            response = chatbot.process_query(user_query)
            print("\nRéponse du chatbot:")
            # Ensure the output handles Markdown (relevant for Streamlit or similar UIs)
            print(response)

    except ValueError as ve:
         print(f"\nErreur d'initialisation ou de configuration: {ve}")
    except requests.exceptions.RequestException as req_err:
         print(f"\nErreur réseau lors de l'initialisation : {req_err}")
         print("Vérifiez votre connexion et les URLs des API.")
    except Exception as e:
        print(f"\n❌ Erreur critique : {e}")
        import traceback
        traceback.print_exc()

# --- END OF FILE Back.py ---
