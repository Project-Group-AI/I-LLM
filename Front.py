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

# Fonction pour gérer le changement de thème
def change_theme():
    previous_theme = st.session_state.themes["current_theme"]
    new_theme = "dark" if previous_theme == "light" else "light"
    st.session_state.themes["current_theme"] = new_theme  # Mise à jour de l'état
    tdict = st.session_state.themes[new_theme]  # Récupérer le dictionnaire du nouveau thème
    
    for vkey, vval in tdict.items(): 
        if vkey.startswith("theme"): 
            st._config.set_option(vkey, vval)

    # Preserve the current font selection
    st.session_state["selected_font"] = st.session_state.get("selected_font", "police de base")

    # Ensure font selection is preserved
    st.session_state.themes["refreshed"] = True
    st.session_state.run_rerun = True  # Définir un flag pour le rerun

def apply_font_size():
    # Fonction séparée pour appliquer la taille de police
    st.session_state["font_size"] = st.session_state["font_size_draft"]

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

    
    # Initialisation de la taille de police si nécessaire
    if "font_size" not in st.session_state:
        st.session_state["font_size"] = 16  # Taille de police par défaut
    
    # Initialisation du brouillon de taille de police
    if "font_size_draft" not in st.session_state:
        st.session_state["font_size_draft"] = st.session_state["font_size"]

    display_header()

    # Détermine la police à utiliser
    if st.session_state["selected_font"] == "Comic Sans MS":
        css_font = "'Comic Sans MS', cursive, sans-serif"
    else:
        # Default system font that adapts to the theme
        css_font = "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"

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



    with st.expander("⚙️ Paramètres de configuration", expanded=True):
        col1, col2 = st.columns(2)
            
        with col1:
            gemini_api_key = st.text_input("Clé API Gemini", type="password", 
                                      help="Entrez votre clé API Gemini pour activer le chatbot")
            
        with col2:
            api_base_url = st.text_input("Lien de l'API pour le dataset", 
                                        value="https://tabular-api.data.gouv.fr/api/resources/93ae96a7-1db7-4cb4-a9f1-6d778370b640/data/",
                                        help="Lien de l'API pour le dataset des établissements accessibles")
        
        # Check if manual inputs are provided
        if not (gemini_api_key and api_base_url):
            st.warning("⚠️ Veuillez entrer votre clé API Gemini et l'URL de l'API pour initialiser le chatbot.")
            return
        
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
        
        # --- Sélecteur de police ---
        # Le selectbox est associé à la clé "selected_font" pour sauvegarder le choix dans session_state
        st.selectbox("Police du site", ["police de base", "Comic Sans MS"], key="selected_font")

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
                with st.expander(f"Conversation {i+1} - {conversation['date']}"):
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
            if "messages" in st.session_state and len(st.session_state.messages) > 1:
                # Sauvegarde de la conversation actuelle si elle existe
                st.session_state.conversation_history.append({
                    "date": time.strftime("%d/%m/%Y %H:%M"),
                    "messages": st.session_state.messages.copy()
                })
                save_conversations(st.session_state.conversation_history)
            
            # Réinitialisation des messages
            st.session_state.messages = [
                {"role": "assistant", "content": "Bonjour ! Je suis votre assistant virtuel pour l'accessibilité. Comment puis-je vous aider aujourd'hui ?"}
            ]
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
    if  gemini_api_key and api_base_url:
        try:
            chatbot = initialize_chatbot(api_base_url,  gemini_api_key)
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
            if len(st.session_state.messages) > 1:
                if st.button("💾 Sauvegarder cette conversation"):
                    st.session_state.conversation_history.append({
                        "date": time.strftime("%d/%m/%Y %H:%M"),
                        "messages": st.session_state.messages.copy()
                    })
                    save_conversations(st.session_state.conversation_history)
                    st.success("Conversation sauvegardée!")
            
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
        
        # Affichage d'une démo visuelle en attendant
        st.image("https://via.placeholder.com/800x400?text=Chatbot+Inclusif+Demo", caption="Aperçu du chatbot")
        
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