import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import json
import time

# Import de la classe ChatbotInclusifGemini existante
# (assurez-vous que le fichier contenant cette classe est dans le même répertoire)
from Back import ChatbotInclusifGemini

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Chatbot Inclusif - Accessibilité",
    page_icon="♿",
    layout="wide",
    initial_sidebar_state="expanded"
)

CONVERSATIONS_FILE = "conversations.json"

def load_conversations():
    if os.path.exists(CONVERSATIONS_FILE):
        with open(CONVERSATIONS_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    else:
        return []

def save_conversations(conversations):
    with open(CONVERSATIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(conversations, f, ensure_ascii=False, indent=4)

# Fonction pour initialiser le chatbot
@st.cache_resource
def initialize_chatbot(api_base_url,  gemini_api_key):
    with st.spinner("Initialisation du chatbot en cours..."):
        return ChatbotInclusifGemini(api_base_url,  gemini_api_key)

# Fonction pour afficher l'en-tête de l'application
def display_header():
    st.title("I-LLM")
    st.markdown("""
    ### Trouvez des établissements publics accessibles et obtenez des informations sur le handicap
    Ce chatbot vous aide à trouver des établissements adaptés à vos besoins d'accessibilité et répond à vos questions sur le handicap.
    """)

# Fonction pour appliquer un thème d'accessibilité
def apply_accessibility_theme(theme_key):
    # Récupère le thème de base actuel (light ou dark)
    current_theme_mode = st.session_state.themes["current_theme"]  # "light" ou "dark"
    
    if theme_key == "default":
        # Revenir au thème de base (light/dark actuel)
        theme_dict = st.session_state.themes[current_theme_mode]
        
        # Appliquer les paramètres du thème de base
        for key, value in theme_dict.items():
            if key.startswith("theme"):
                st._config.set_option(key, value)
        
        # Réinitialiser la taille de police
        st.session_state["font_size"] = 16
        st.session_state["font_size_draft"] = 16
    else:
        # Récupérer le dictionnaire du thème d'accessibilité
        accessibility_theme = st.session_state.accessibility_themes[theme_key]
        
        # Appliquer d'abord les couleurs de base du thème actuel (light/dark)
        base_theme_dict = st.session_state.themes[current_theme_mode]
        for key, value in base_theme_dict.items():
            if key.startswith("theme"):
                st._config.set_option(key, value)
        
        # Appliquer ensuite les modifications spécifiques du thème d'accessibilité pour le mode actuel
        theme_mode_dict = accessibility_theme.get(current_theme_mode, {})
        for key, value in theme_mode_dict.items():
            if key.startswith("theme"):
                st._config.set_option(key, value)
                
        # Appliquer la taille de police si spécifiée
        if "font_size" in accessibility_theme:
            st.session_state["font_size"] = accessibility_theme["font_size"]
            st.session_state["font_size_draft"] = accessibility_theme["font_size"]
        
        # Appliquer la police si spécifiée dans le thème (mapping vers l'option du selectbox)
        if "fontFamily" in theme_mode_dict:
            ff = theme_mode_dict["fontFamily"]
            if "Comic Sans MS" in ff:
                st.session_state["selected_font"] = "Comic Sans MS"
            elif "Verdana" in ff:
                st.session_state["selected_font"] = "Verdana"
            else:
                st.session_state["selected_font"] = "police de base"
    
    # Stocker le thème d'accessibilité actuel
    st.session_state["current_accessibility_theme"] = theme_key
    
    # Forcer le rafraîchissement
    st.session_state.run_rerun = True
    
def change_theme():
    previous_theme = st.session_state.themes["current_theme"]
    new_theme = "dark" if previous_theme == "light" else "light"
    st.session_state.themes["current_theme"] = new_theme  # Mise à jour de l'état
    tdict = st.session_state.themes[new_theme]  # Récupérer le dictionnaire du nouveau thème
    
    # Sauvegarder le thème d'accessibilité actuel
    current_accessibility_theme = st.session_state.get("current_accessibility_theme", "default")
    
    # Appliquer le thème de base seulement si on n'utilise pas un thème d'accessibilité personnalisé
    if current_accessibility_theme == "default":
        for vkey, vval in tdict.items(): 
            if vkey.startswith("theme"): 
                st._config.set_option(vkey, vval)
    else:
        # Si un thème d'accessibilité est actif, le réappliquer avec les nouvelles couleurs de base
        # Cette ligne réapplique le thème d'accessibilité tout en tenant compte du changement light/dark
        apply_accessibility_theme(current_accessibility_theme)

    # Preserve the current font selection
    st.session_state["selected_font"] = st.session_state.get("selected_font", "police de base")

    # Ensure font selection is preserved
    st.session_state.themes["refreshed"] = True
    st.session_state.run_rerun = True  # Définir un flag pour le rerun

def apply_font_size():
    # Fonction séparée pour appliquer la taille de police
    st.session_state["font_size"] = st.session_state["font_size_draft"]

def generate_title(messages):
    """Génère un titre court résumant la demande générale de l'utilisateur en préservant la casse originale."""
    import re
    for m in messages:
        if m.get("role") == "user" and m.get("content"):
            text = m.get("content").strip()
            # Supprimer la ponctuation en conservant la casse originale
            text_no_punct = re.sub(r'[^\w\sÀ-ÿ]', '', text)
            original_words = text_no_punct.split()
            # Liste de stopwords en minuscules, incluant "y" et "atil" pour gérer "Y a-t-il"
            stopwords = {
                "quels", "quelles", "sont", "mes", "en", "tant", "que", "les", "des",
                "de", "le", "la", "et", "pour", "a", "un", "une", "ces", "ce", "est",
                "où", "puisje", "trouver", "je", "cherche", "chercher", "personne", "dont", "au", "aux",
                "y", "atil"
            }
            # Conserver la casse originale pour les mots non filtrés
            keywords = [word for word in original_words if word.lower() not in stopwords]
            
            # Extraction d'une localisation dans le texte original (exemple : "à Paris")
            loc_match = re.search(r' à ([\wÀ-ÿ]+)', text, re.IGNORECASE)
            location = loc_match.group(1).strip() if loc_match else ""
            # Retirer la localisation des mots clés si elle y figure
            keywords = [word for word in keywords if word.lower() != location.lower()]
            
            # Utiliser jusqu'à 3 mots clés pour le résumé
            summary_keywords = keywords[:3]
            title = " ".join(summary_keywords)
            if location:
                title += " à " + location
            # S'assurer que le titre débute par une majuscule sans modifier le reste
            if title:
                title = title[0].upper() + title[1:]
            return title
    import time
    return time.strftime("%d/%m/%Y %H:%M")

# Fonction principale
def main():
    if st.session_state.get("run_rerun", False):
        st.session_state.run_rerun = False  # Réinitialiser le flag
        st.rerun()

    if "selected_font" not in st.session_state:
        st.session_state["selected_font"] = "police de base"

    # Initialisation du thème si nécessaire
    ms = st.session_state
    if "themes" not in ms: 
        ms.themes = {
            "current_theme": "light",
            "refreshed": True,
            "light": {
                "theme.base": "light",
                "theme.backgroundColor": "white",
                "theme.primaryColor": "#3B82F6",  # Soft blue for primary elements in light mode
                "theme.secondaryBackgroundColor": "#f0f2f6",  # Light gray for secondary background
                "theme.textColor": "#1F2937",  # Dark charcoal for text in light mode
                "button_face": "🌜"
            },
            "dark": {
                "theme.base": "dark",
                "theme.backgroundColor": "#121212",  # Very dark background, almost black
                "theme.primaryColor": "#6366F1",  # Indigo for primary elements in dark mode
                "theme.secondaryBackgroundColor": "#1E1E1E",  # Dark gray for secondary background
                "theme.textColor": "#E5E7EB",  # Light gray for text in dark mode
                "button_face": "🌞"
            }
        }
    if "accessibility_themes" not in ms:
        ms.accessibility_themes = {
            "default": {
                "name": "Par défaut",
                "description": "Thème standard de l'application",
                "light": {},
                "dark": {}
            },
            "high_contrast": {
                "name": "Contraste élevé",
                "description": "Noir sur blanc ou blanc sur noir pour une lisibilité maximale",
                "light": {
                    "theme.backgroundColor": "#FFFFFF",
                    "theme.primaryColor": "#000000",
                    "theme.secondaryBackgroundColor": "#E0E0E0",
                    "theme.textColor": "#000000"
                },
                "dark": {
                    "theme.backgroundColor": "#000000",
                    "theme.primaryColor": "#FFFFFF",
                    "theme.secondaryBackgroundColor": "#222222",
                    "theme.textColor": "#FFFFFF"
                },
            },
            "deuteranopia": {
                "name": "Deutéranopie",
                "description": "Adapté pour les personnes avec difficulté à percevoir le vert",
                "light": {
                    "theme.backgroundColor": "#F8F8FF",
                    "theme.primaryColor": "#0000FF",
                    "theme.secondaryBackgroundColor": "#E6E6FA",
                    "theme.textColor": "#000080",
                    "fontFamily": "Verdana, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif"
                },
                "dark": {
                    "theme.backgroundColor": "#000033",
                    "theme.primaryColor": "#ADD8E6",
                    "theme.secondaryBackgroundColor": "#191970",
                    "theme.textColor": "#E6E6FA",
                    "fontFamily": "Verdana, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif"
                }
            },
            "protanopia": {
                "name": "Protanopie",
                "description": "Adapté pour les personnes avec difficulté à percevoir le rouge",
                "light": {
                    "theme.backgroundColor": "#F0FFFF",
                    "theme.primaryColor": "#00008B",
                    "theme.secondaryBackgroundColor": "#E0FFFF",
                    "theme.textColor": "#000080",
                    "fontFamily": "Verdana, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif"
                },
                "dark": {
                    "theme.backgroundColor": "#000033",
                    "theme.primaryColor": "#87CEEB",
                    "theme.secondaryBackgroundColor": "#191970",
                    "theme.textColor": "#ADD8E6",
                    "fontFamily": "Verdana, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif"
                }
            },
            "tritanopia": {
                "name": "Tritanopie",
                "description": "Adapté pour les personnes avec difficulté à percevoir le bleu",
                "light": {
                    "theme.backgroundColor": "#FFFAF0",
                    "theme.primaryColor": "#8B0000",
                    "theme.secondaryBackgroundColor": "#FFF5EE",
                    "theme.textColor": "#800000",
                    "fontFamily": "Verdana, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif"
                },
                "dark": {
                    "theme.backgroundColor": "#330000",
                    "theme.primaryColor": "#FFCCCC",
                    "theme.secondaryBackgroundColor": "#4D0000",
                    "theme.textColor": "#FFF0E0",
                    "fontFamily": "Verdana, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif"
                }
            },
            "dyslexia_friendly": {
                "name": "Adapté pour dyslexie",
                "description": "Police et couleurs adaptées pour la dyslexie",
                "light": {
                    "theme.backgroundColor": "#FFFDD0",
                    "theme.primaryColor": "#2E5090",
                    "theme.secondaryBackgroundColor": "#F5F5DC",
                    "theme.textColor": "#333333",
                    "fontFamily": "'Comic Sans MS', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif"
                },
                "dark": {
                    "theme.backgroundColor": "#2A2A2A",
                    "theme.primaryColor": "#82C3EC",
                    "theme.secondaryBackgroundColor": "#3A3A3A",
                    "theme.textColor": "#E6E6E6",
                    "fontFamily": "'Comic Sans MS', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif"
                }
            },
            "achromatopsia": {
                "name": "Achromatopsie",
                "description": "Thème en niveaux de gris pour les personnes ne percevant pas les couleurs",
                "light": {
                    "theme.backgroundColor": "#E8E8E8",
                    "theme.primaryColor": "#4D4D4D",
                    "theme.secondaryBackgroundColor": "#C0C0C0",
                    "theme.textColor": "#1A1A1A",
                    "fontFamily": "Verdana, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif"
                },
                "dark": {
                    "theme.backgroundColor": "#2B2B2B",
                    "theme.primaryColor": "#AAAAAA",
                    "theme.secondaryBackgroundColor": "#3D3D3D",
                    "theme.textColor": "#E0E0E0",
                    "fontFamily": "Verdana, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif"
                }
            }
        }

    # Initialisation de la taille de police si nécessaire
    if "font_size" not in st.session_state:
        st.session_state["font_size"] = 16  # Taille de police par défaut
    
    # Initialisation du brouillon de taille de police
    if "font_size_draft" not in st.session_state:
        st.session_state["font_size_draft"] = st.session_state["font_size"]

    display_header()

    # Détermine la police à utiliser (directement depuis le nom ou le thème)
    selected_font = st.session_state.get("selected_font", "police de base")

    if selected_font == "Comic Sans MS":
        css_font = "'Comic Sans MS', cursive, sans-serif"
    elif selected_font == "Verdana":
        css_font = "Verdana, sans-serif"
    elif selected_font == "police de base":
        css_font = "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
    else:
        # Si la valeur vient directement d’un thème, utilise-la telle quelle (ex. "'Comic Sans MS', Arial, sans-serif")
        css_font = selected_font

    # Injection of CSS to apply font and size on all elements
    st.markdown(
        f"""
        <style>
        * {{
            font-family: {css_font} !important;
            font-size: {st.session_state['font_size']}px !important;
        }}
        /* Adjustments for different element types */
        h1 {{ font-size: {st.session_state['font_size'] * 2}px !important; }}
        h2 {{ font-size: {st.session_state['font_size'] * 1.5}px !important; }}
        h3 {{ font-size: {st.session_state['font_size'] * 1.3}px !important; }}
        small {{ font-size: {st.session_state['font_size'] * 0.8}px !important; }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # Injection du CSS pour appliquer la police sur tous les éléments
    # Moved this to ensure it's always applied, regardless of theme change
    st.markdown(
        f"""
        <style>
        * {{
            font-family: {css_font} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


    with st.expander("⚙️ Paramètres de configuration", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            # Champ pour la clé API (masqué)
             gemini_api_key = st.text_input("Clé API Gemini", type="password", value=st.session_state.get("gemini_api_key", "AIzaSyDl3AzHPqefJLbYgDc_MywAbGtdEpqr4gE"), 
                                  help="Entrez votre clé API Gemini pour activer le chatbot")
        
        with col2:
            # Sélection du fichier de données
            api_base_url = st.text_input("Lien de l'API pour le dataset", 
                                       value=st.session_state.get("api_base_url", "https://tabular-api.data.gouv.fr/api/resources/93ae96a7-1db7-4cb4-a9f1-6d778370b640/data/"),
                                       help="Lien de l'API pour le dataset des établissements accessibles")
        
        # Bouton pour appliquer les paramètres
        if st.button("Appliquer les paramètres"):
            st.success("Paramètres appliqués avec succès!")
            # On met à jour une variable d'état plutôt que de redémarrer l'app
            st.session_state. gemini_api_key =  gemini_api_key
            st.session_state.api_base_url = api_base_url
            # Redirection conditionnelle ou gestion de l'état
            st.rerun()  # Cela peut être potentiellement supprimé si cela devient redondant.

    # ------------------------------
    # Chargement de l'historique depuis le fichier
    # ------------------------------
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = load_conversations()
    
    # Sidebar pour l'historique des conversations
    with st.sidebar:
        # Bouton de thème dans la sidebar
        btn_face = (
            ms.themes["light"]["button_face"] 
            if ms.themes["current_theme"] == "light" 
            else ms.themes["dark"]["button_face"]
        )
        st.button(f"{btn_face} Changer de thème", on_click=change_theme)

        # Initialiser le thème d'accessibilité actuel s'il n'existe pas
        if "current_accessibility_theme" not in st.session_state:
            st.session_state["current_accessibility_theme"] = "default"

        # Liste des thèmes avec descriptions pour l'affichage dans le selectbox
        theme_options = [(key, f"{value['name']} - {value['description']}") 
                        for key, value in st.session_state.accessibility_themes.items()]

        # Fonction callback pour appliquer le thème immédiatement après sélection
        def on_theme_change():
            # Récupérer directement la valeur du selectbox via la session_state
            selected_theme = st.session_state.accessibility_theme_selector
            # Appliquer le thème directement, sans condition supplémentaire
            apply_accessibility_theme(selected_theme)
            # Forcer le rechargement pour s'assurer que le thème est correctement appliqué
            st.session_state.run_rerun = True

        # Utiliser on_change pour déclencher l'application du thème dès la sélection
        st.sidebar.selectbox(
            "Thèmes d'accessibilité",
            options=[key for key, _ in theme_options],
            format_func=lambda x: next((st.session_state.accessibility_themes[k]["name"] for k, _ in theme_options if k == x), x),
            index=list(st.session_state.accessibility_themes.keys()).index(st.session_state["current_accessibility_theme"]),
            key="accessibility_theme_selector",
            on_change=on_theme_change
        )
        
        # --- Sélecteur de police ---
        # Le selectbox est associé à la clé "selected_font" pour sauvegarder le choix dans session_state
        st.selectbox(
            "Police du site",
            ["police de base", "Comic Sans MS", "Verdana"],
            key="selected_font"
        )

        # Slider pour la taille de police draft
        font_size_draft = st.slider(
            "Ajustez la taille de police", 
            min_value=12, 
            max_value=24, 
            value=st.session_state["font_size_draft"], 
            key="font_size_draft_slider"
        )
        st.session_state["font_size_draft"] = font_size_draft
        
        # Bouton Appliquer avec on_click
        st.button(
            "Appliquer la taille de police", 
            on_click=apply_font_size,
            key="apply_font_size_btn"
        )

        st.divider()
        
        st.header("📜 Historique des conversations")
        
        # Affichage de l'historique des conversations
        if "conversation_history" not in st.session_state:
            st.session_state.conversation_history = []
        
        if not st.session_state.conversation_history:
            st.info("Aucune conversation enregistrée.")
        else:
            for i, conversation in enumerate(st.session_state.conversation_history):
                with st.expander(f"{conversation.get('title', f'Conversation {i+1}')} - {conversation['date']}"):
                    for message in conversation["messages"]:
                        st.markdown(f"**{message['role'].capitalize()}**: {message['content'][:50]}...")
                    
                    # Option pour charger cette conversation
                    if st.button(f"Charger cette conversation", key=f"load_{i}"):
                        st.session_state.messages = conversation["messages"].copy()
                        st.rerun()
                    
                    # Option pour supprimer cette conversation
                    if st.button(f"Supprimer", key=f"delete_{i}"):
                        st.session_state.conversation_history.pop(i)
                        save_conversations(st.session_state.conversation_history)
                        st.rerun()
        
        # Bouton pour créer une nouvelle conversation
        if st.button("➕ Nouvelle conversation"):
            # Réinitialisation des messages
            st.session_state.messages = [
                {"role": "assistant", "content": "Bonjour ! Je suis votre assistant virtuel pour l'accessibilité. Comment puis-je vous aider aujourd'hui ?"}
            ]
            if "saved_conversation" in st.session_state:
                del st.session_state.saved_conversation
            st.rerun()

        if st.button("Supprimer toutes les conversations"):
            st.session_state.conversation_history = []
            save_conversations(st.session_state.conversation_history)
            st.rerun()
        
        st.divider()
        
        # Informations sur l'application
        st.markdown("""
        ### À propos
        Ce chatbot utilise:
        - Gemini 2.0 Flash pour la génération des réponses
        - Sentence Transformers pour l'analyse sémantique
        - Les données AccesLibre pour les établissements
        """)

    # Initialisation du chatbot si les paramètres sont fournis
    if st.session_state.get("gemini_api_key") and st.session_state.get("api_base_url"):
        try:
            chatbot = initialize_chatbot(st.session_state["api_base_url"], st.session_state["gemini_api_key"])
            st.success("✅ Chatbot initialisé avec succès!")
            
            # Initialisation de l'historique de conversation s'il n'existe pas déjà
            if "messages" not in st.session_state:
                st.session_state.messages = [
                    {"role": "assistant", "content": "Bonjour ! Je suis votre assistant virtuel pour l'accessibilité. Comment puis-je vous aider aujourd'hui ?"}
                ]
            
            # Affichage de l'historique des messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            def process_question(question):
                st.session_state.messages.append({"role": "user", "content": question})
                with st.chat_message("user"):
                    st.markdown(question)
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    message_placeholder.markdown("🔍 Recherche en cours...")
                    try:
                        response = chatbot.process_query(question)
                        with st.spinner("Génération de la réponse..."):
                            for i in range(len(response) // 10):
                                partial_response = response[:i*10]
                                message_placeholder.markdown(f"{partial_response}▌")
                                time.sleep(0.01)
                            message_placeholder.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        message_placeholder.markdown(f"⚠️ Désolé, une erreur s'est produite : {str(e)}")
            
            if prompt := st.chat_input("Posez votre question sur l'accessibilité ou recherchez un établissement..."):
                process_question(prompt)

            if "selected_question" in st.session_state:
                question = st.session_state.selected_question
                del st.session_state.selected_question
                process_question(question)
            
            # Bouton pour sauvegarder la conversation actuelle
            if len(st.session_state.messages) > 1 and "saved_conversation" not in st.session_state:
                if st.button("💾 Sauvegarder cette conversation"):
                    st.session_state.conversation_history.append({
                        "date": time.strftime("%d/%m/%Y %H:%M"),
                        "title": generate_title(st.session_state.messages),
                        "messages": st.session_state.messages.copy()
                    })
                    save_conversations(st.session_state.conversation_history)
                    st.session_state.saved_conversation = True
                    st.success("Conversation sauvegardée!")
                    st.rerun()
            
            with st.expander("📝 Exemples de questions"):
                st.write("Cliquez sur une question pour la poser automatiquement:")
                st.markdown("""**Recherche d'établissements:**""")
                if st.button("Où puis-je trouver un restaurant accessible en fauteuil roulant à Paris ?"):
                    st.session_state.selected_question = "Où puis-je trouver un restaurant accessible en fauteuil roulant à Paris ?"
                    st.rerun()
                if st.button("Y a-t-il un musée avec audiodescription à Lyon ?"):
                    st.session_state.selected_question = "Y a-t-il un musée avec audiodescription à Lyon ?"
                    st.rerun()
                if st.button("Je cherche une piscine avec stationnement PMR à Bordeaux"):
                    st.session_state.selected_question = "Je cherche une piscine avec stationnement PMR à Bordeaux"
                    st.rerun()
                st.markdown("""**Questions générales:**""")
                if st.button("Quelles sont les aides financières pour les personnes handicapées ?"):
                    st.session_state.selected_question = "Quelles sont les aides financières pour les personnes handicapées ?"
                    st.rerun()
                if st.button("Comment fonctionne la MDPH ?"):
                    st.session_state.selected_question = "Comment fonctionne la MDPH ?"
                    st.rerun()
                if st.button("Quels sont mes droits en tant que personne malvoyante ?"):
                    st.session_state.selected_question = "Quels sont mes droits en tant que personne malvoyante ?"
                    st.rerun()
                st.markdown("""**Outils disponibles:**""")
                st.markdown("""Recherche google maps:""")
                if st.button("Je cherche un restaurant accessible PMR à Lille, donne moi le liens google maps pour savoir ou il se situe."):
                    st.session_state.selected_question = "Je cherche un restaurant accessible PMR à Lille, génére moi une carte maps pour savoir ou il se situe."
                    st.rerun()

                    
        except Exception as e:
            st.error(f"Erreur lors de l'initialisation du chatbot: {str(e)}")
            st.info("Vérifiez que l'URL de l'API est correct et que votre clé API est valide.")
    else:
        st.warning("⚠️ Veuillez entrer votre clé API Gemini et l'URL de l'API pour initialiser le chatbot.")
        
        # Exemples de cas d'utilisation
        st.subheader("🚀 Ce que vous pourrez faire avec ce chatbot:")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 🔍 Recherche")
            st.markdown("Trouvez des établissements accessibles selon vos besoins spécifiques")
        
        with col2:
            st.markdown("### 💡 Information")
            st.markdown("Obtenez des réponses sur le handicap, les droits et les aides")
        
        with col3:
            st.markdown("### 🗺️ Localisation")
            st.markdown("Découvrez les lieux accessibles près de chez vous")

# Point d'entrée de l'application
if __name__ == "__main__":
    main()