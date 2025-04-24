# Projet Chatbot Inclusif "I-LLM"

## Lien
https://i-llm-chatbot.streamlit.app/

## Objectif du Projet

Ce projet vise à développer un chatbot conversationnel inclusif nommé "I-LLM". L'objectif principal est d'aider les utilisateurs à trouver des établissements publics accessibles en France et à obtenir des informations générales sur le handicap, les aides et les droits associés.


## Aperçu des Fichiers

*   **`front.py`**: Gère l'interface utilisateur web avec Streamlit. Il permet l'interaction avec l'utilisateur, affiche la conversation, gère les paramètres d'accessibilité (thèmes, polices) et communique avec le backend.
*   **`back.py`**: Contient la logique métier du chatbot (`ChatbotInclusifGemini`). Il traite les requêtes, interroge les API externes (données établissements, Gemini, parking PMR), classe les intentions, et génère les réponses appropriées.


---

## Détail des Fonctions - `front.py`

#### `load_conversations()`
Charge l'historique des conversations précédentes depuis le fichier `conversations.json`. Gère les cas où le fichier n'existe pas ou est corrompu.

```python
def load_conversations():
    if os.path.exists(CONVERSATIONS_FILE):
        # ... load logic ...
    else:
        return []
```


#### `save_conversations(conversations)`
Sauvegarde la liste actuelle des conversations dans le fichier conversations.json. Assure un formatage lisible et l'encodage UTF-8.

```python
def save_conversations(conversations):
    with open(CONVERSATIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(conversations, f, ensure_ascii=False, indent=4)
```


#### `initialize_chatbot(api_base_url, gemini_api_key)`
Initialise et met en cache l'instance du backend ChatbotInclusifGemini. Affiche un indicateur de chargement pendant l'initialisation.

```python
@st.cache_resource
def initialize_chatbot(api_base_url,  gemini_api_key):
    with st.spinner("Initialisation du chatbot en cours..."):
        return ChatbotInclusifGemini(api_base_url,  gemini_api_key)
```


#### `display_header()`
Affiche le titre principal "I-LLM" et une courte description du chatbot. Utilise les éléments d'interface Streamlit st.title et st.markdown.

```python
def display_header():
    st.title("I-LLM")
    st.markdown("""
    ### Trouvez des établissements publics accessibles...
    """)
```


#### `apply_accessibility_theme(theme_key)`
Applique un thème visuel spécifique (couleurs, police, taille) choisi par l'utilisateur. Gère le thème par défaut et les thèmes d'accessibilité en tenant compte du mode clair/sombre actuel.

```python
def apply_accessibility_theme(theme_key):
    # ... theme logic based on theme_key and light/dark mode ...
    st.session_state["current_accessibility_theme"] = theme_key
    st.session_state.run_rerun = True
```


#### `change_theme()`
Bascule entre le thème de base clair et sombre de l'application. Réapplique le thème d'accessibilité actif par-dessus le nouveau thème de base.

```python
def change_theme():
    # ... toggles st.session_state.themes["current_theme"] ...
    # ... reapplies accessibility theme if needed ...
    st.session_state.run_rerun = True
```


#### `apply_font_size()`
Met à jour la taille de police globale de l'application. Prend la valeur définie par le slider (font_size_draft) et l'applique.

```python
def apply_font_size():
    st.session_state["font_size"] = st.session_state["font_size_draft"]
```


#### `generate_title(messages)`
Génère automatiquement un titre court pour une conversation. Extrait des mots-clés pertinents de la première question de l'utilisateur en préservant la casse.

```python
def generate_title(messages):
    # ... logic to extract keywords from messages ...
    title = " ".join(summary_keywords)
    # ...
    return title
```


#### `main()`
Fonction principale qui exécute l'application Streamlit. Gère l'état de la session, l'affichage des composants (sidebar, chat, settings), l'injection CSS, l'interaction utilisateur et l'appel au backend.

```python
def main():
    # ... setup themes, fonts, session state ...
    display_header()
    # ... CSS injection ...
    # ... sidebar logic (history, theme, font settings) ...
    # ... main chat interface logic ...
    if prompt := st.chat_input(...):
        process_question(prompt)
    # ...
```

#### `process_question(question)` (Fonction imbriquée dans main)
Gère la soumission d'une nouvelle question par l'utilisateur. Ajoute la question à l'historique, l'affiche, appelle le backend pour obtenir une réponse et affiche la réponse (avec effet de streaming).

```python
# (Inside main function)
            def process_question(question):
                st.session_state.messages.append({"role": "user", "content": question})
                # ... display user message ...
                with st.chat_message("assistant"):
                    # ... call chatbot.process_query ...
                    # ... display assistant response with streaming effect ...
```


---

## Détail des Fonctions - `back.py` (Classe ChatbotInclusifGemini)

#### `__init__(self, api_base_url, gemini_api_key)`
Initialise la classe du chatbot. Configure les API (établissements, parking PMR, Gemini), charge les modèles (SentenceTransformer, classificateur local, Gemini) et la base de connaissances.

```python
class ChatbotInclusifGemini:
    def __init__(self, api_base_url, gemini_api_key):
        self.api_base_url = api_base_url
        self.pmr_parking_api_url = "..."
        self.model = SentenceTransformer(...)
        # ... configure genai ...
        # ... load local classifier ...
        self.check_api_connection()
        self.initialize_knowledge_base()
```


#### `check_api_connection(self)`
Vérifie si la connexion à l'API des données d'établissements est fonctionnelle. Effectue une requête test et affiche des informations sur la structure des données si réussie.

```python
def check_api_connection(self):
        try:
            api_url = f"{self.api_base_url}?page=1&page_size=1"
            response = requests.get(api_url, timeout=10)
            # ... check response status and format ...
        except requests.exceptions.RequestException as e:
            # ... handle errors ...
```


#### `initialize_knowledge_base(self)`
Crée une base de connaissances interne simple. Contient des listes prédéfinies de types de handicap, d'aides financières, d'organismes et de droits pour aider à générer des réponses informatives.

```python
def initialize_knowledge_base(self):
        self.knowledge_base = {
            "types_handicap": [...],
            "aides_financieres": [...],
            # ... more categories ...
        }
```


#### `classify_query_type(self, query)`
Utilise le modèle Gemini pour classifier l'intention de la requête utilisateur. Détermine s'il s'agit d'une recherche de lieu (establishment_search), d'une question générale (general_info) ou hors sujet (off_topic).

```python
def classify_query_type(self, query):
        prompt = f"""Analyse la requête suivante..."""
        try:
            response = self.gemini_model.generate_content(prompt)
            # ... parse response to return type string ...
        except Exception as e:
            # ... handle errors ...
```


#### `early_classification(self, query)`
Effectue une classification rapide de la requête à l'aide d'un modèle local (DistilCamembert fine-tuné). Offre une alternative plus rapide à l'appel API Gemini pour la classification initiale.

```python
def early_classification(self, query):
        if self.early_classifier is None:
            return self.classify_query_type(query) # Fallback
        result = self.early_classifier(query)
        # ... map label to 'establishment_search', 'general_info', 'off_topic' ...
```


#### `rank_establishments_by_embedding(self, establishments, query, top_n=5)`
Classe une liste d'établissements en fonction de leur pertinence sémantique par rapport à la requête utilisateur. Utilise les embeddings de SentenceTransformer et la similarité cosinus.

```python
def rank_establishments_by_embedding(self, establishments, query, top_n=5):
        # ... generate embeddings for query and establishments ...
        # ... calculate cosine similarity ...
        similarities = cosine_similarity(...)
        # ... sort establishments based on similarities ...
        return ranked_establishments
```


#### `generate_hypothetical_document_with_gemini(self, query)`
Extrait des critères de recherche structurés (commune, activité, besoins d'accessibilité) à partir de la requête utilisateur via Gemini. Retourne ces critères sous forme de dictionnaire pour l'appel API.

```python
def generate_hypothetical_document_with_gemini(self, query):
        prompt = f"""Analyse la requête utilisateur suivante et extrait les critères..."""
        try:
            response = self.gemini_model.generate_content(prompt)
            # ... extract JSON from response ...
            criteria = json.loads(json_text)
            # ... clean and normalize criteria ...
            return doc_text, criteria_cleaned
        except Exception as e:
            # ... handle errors ...
```


#### `search_all_establishments(self, criteria, max_pages=10)`
Interroge l'API des établissements pour récupérer les lieux correspondant aux critères extraits. Gère la pagination pour obtenir potentiellement plusieurs pages de résultats.

```python
def search_all_establishments(self, criteria, max_pages=10):
        all_establishments = []
        page = 1
        while page <= max_pages:
            # ... build API query parameters from criteria ...
            api_url = f"{self.api_base_url}?{'&'.join(query_params)}"
            # ... make API request and handle pagination ...
        return all_establishments
```


#### `_calculate_distance(self, lat1, lon1, lat2, lon2)`
Fonction utilitaire privée pour calculer la distance (en km) entre deux points géographiques. Utilise la formule de Haversine.

```python
def _calculate_distance(self, lat1, lon1, lat2, lon2):
        R = 6371 # Earth radius
        # ... Haversine formula implementation ...
        return distance
```


#### `search_pmr_parking(self, latitude, longitude, max_distance_km=2.0, limit=100)`
Recherche des places de stationnement PMR à proximité d'un point GPS donné. Utilise l'API spécifique de Seine Ouest et filtre les résultats par distance.

```python
def search_pmr_parking(self, latitude, longitude, max_distance_km=2.0, limit=100):
        nearby_parking_spots = []
        api_url = f"{self.pmr_parking_api_url}"
        try:
            response = requests.get(api_url, params=params, timeout=15)
            # ... process results and filter by distance using _calculate_distance ...
        except Exception as e:
            # ... handle errors ...
        return nearby_parking_spots
```


#### `generate_natural_response(self, establishments, query, criteria)`
Génère une réponse textuelle conviviale à partir des résultats de recherche d'établissements. Met en forme les informations (nom, adresse, accessibilité, parking PMR avec liens Maps) en utilisant Gemini pour un ton naturel.

```python
def generate_natural_response(self, establishments, query, criteria):
        # ... format establishment data (including nearby PMR parking with MD links) ...
        prompt = f"""Tu es un assistant conversationnel... Génère une réponse en Markdown..."""
        try:
            response = self.gemini_natural_model.generate_content(prompt)
            final_response_text = # ... extract text ...
            # ... add disclaimer if parking data was used ...
            return final_response_text.strip()
        except Exception as e:
            # ... handle errors ...
```


#### `generate_knowledge_response(self, query)`
Génère une réponse à une question générale sur le handicap ou l'accessibilité. Utilise Gemini en s'appuyant sur la base de connaissances interne et les mots-clés détectés pour fournir une réponse informative.

```python
def generate_knowledge_response(self, query):
        # ... extract keywords related to knowledge base ...
        prompt = f"""Tu es un assistant spécialisé dans l'information sur le handicap..."""
        try:
            response = self.gemini_knowledge_model.generate_content(prompt)
            # ... extract text ...
            return response_text.strip()
        except Exception as e:
            # ... handle errors ...
```


#### `process_query(self, query)`
Orchestre le traitement complet d'une requête utilisateur. Appelle la classification (locale d'abord), l'extraction de critères, la recherche d'établissements, le classement, la recherche de parking (si pertinent) et la génération de la réponse finale.

```python
def process_query(self, query):
        # 1. Classify query (using early_classification)
        query_type = self.early_classification(query)

        if query_type == "general_info":
            return self.generate_knowledge_response(query)
        elif query_type == "establishment_search":
            # 3. Extract criteria
            _, criteria = self.generate_hypothetical_document_with_gemini(query)
            # 4. Search establishments
            establishments = self.search_all_establishments(criteria, ...)
            # 5. Rank results
            ranked_establishments = self.rank_establishments_by_embedding(...)
            # 6. Search PMR parking (if applicable)
            # ... conditionally call search_pmr_parking and update ranked_establishments ...
            # 7. Generate final response
            return self.generate_natural_response(ranked_establishments, query, criteria)
        else: # off_topic or error
            # ... return appropriate message ...
```

---

## Maintenance et Évolution du Projet

Pour assurer la pérennité et l'amélioration continue de ce chatbot, voici quelques points clés à considérer :

1.  **Gestion des Dépendances :**
    *   Vérifiez régulièrement les mises à jour des bibliothèques Python (`streamlit`, `google-generativeai`, `sentence-transformers`, `transformers`, `requests`, etc.) via `pip list --outdated`.
    *   Mettez à jour les dépendances (`pip install -U <package>`) avec précaution, en testant l'application après chaque mise à jour majeure pour détecter d'éventuelles incompatibilités.

2.  **Suivi des API Externes :**
    *   **API Établissements (AccesLibre via data.gouv) :** Surveillez la disponibilité et d'éventuels changements dans la structure des données retournées par l'URL configurée dans `front.py` et utilisée dans `back.py` (`search_all_establishments`). La fonction `check_api_connection` peut aider au diagnostic initial.
    *   **API Parking PMR (Seine Ouest) :** Vérifiez périodiquement que l'API (`pmr_parking_api_url` dans `back.py`) est toujours active et que le format des réponses n'a pas changé, ce qui impacterait la fonction `search_pmr_parking`.
    *   **API Gemini :** Tenez-vous informé des éventuelles évolutions des modèles Gemini (ex: nouvelles versions de `gemini-1.5-flash`) et des changements dans l'API `google-generativeai`. Mettez à jour les noms des modèles ou les configurations (`generation_config`, `safety_settings`) dans `back.py` si nécessaire.

3.  **Gestion des Modèles :**
    *   **Sentence Transformer (`paraphrase-multilingual-mpnet-base-v2`) :** Vérifiez s'il existe des modèles multilingues plus performants ou mieux adaptés au contexte français pour l'encodage sémantique (`rank_establishments_by_embedding`).
    *   **Classificateur Local (`gabincharlemagne/finetuned-distilcamembert`) :** Évaluez périodiquement sa performance (`early_classification`). Si la précision diminue ou si les types de requêtes évoluent, envisagez un réentraînement ou l'utilisation d'un autre modèle de classification léger.
    *   **Modèles Gemini :** Expérimentez avec différents prompts dans les fonctions `classify_query_type`, `generate_hypothetical_document_with_gemini`, `generate_natural_response`, et `generate_knowledge_response` pour améliorer la précision, la pertinence et le naturel des réponses.

4.  **Base de Connaissances et Données :**
    *   **`knowledge_base` (dans `back.py`) :** Mettez à jour les informations (aides, organismes, droits) si la législation ou les dispositifs évoluent en France.
    *   **`conversations.json` :** Surveillez la taille de ce fichier. Si elle devient trop importante, envisagez des stratégies d'archivage ou de stockage plus robustes (ex: base de données). Assurez-vous que le formatage reste correct.

5.  **Qualité du Code et Interface :**
    *   **Refactoring :** Simplifiez les fonctions complexes dans `front.py` et `back.py` pour améliorer la lisibilité et la maintenabilité.
    *   **Tests :** Envisagez d'ajouter des tests unitaires ou d'intégration pour les fonctions critiques (ex: parsing d'API, logique de classification, formatage des réponses) afin de prévenir les régressions.
    *   **Interface (`front.py`) :** Améliorez l'ergonomie et l'accessibilité de l'interface Streamlit. Testez les thèmes d'accessibilité et les options de police/taille pour garantir une bonne expérience utilisateur.

6.  **Sécurité :**
    *   **Clé API Gemini :** Gérez la clé API de manière sécurisée. Évitez de la stocker directement dans le code source. Utilisez des variables d'environnement ou un système de gestion des secrets, surtout si le projet est déployé.
