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

# Fonction pour initialiser le chatbot
@st.cache_resource
def initialize_chatbot(api_base_url,  gemini_api_key):
    with st.spinner("Initialisation du chatbot en cours..."):
        return ChatbotInclusifGemini(api_base_url,  gemini_api_key)

# Fonction pour afficher l'en-tête de l'application
def display_header():
    # Récupérer les préférences de police de la session
    font_size = st.session_state.get("font_size", "16px")
    line_spacing = st.session_state.get("line_spacing", "1.5")
    
    # Calculer des tailles relatives
    title_size = f"calc({font_size} * 2.5)"  # Plus grand pour le titre principal
    subtitle_size = f"calc({font_size} * 1.17)"  # Pour le sous-titre h3
    
    # Utiliser du HTML personnalisé au lieu de st.title()
    st.markdown(f"""
    <h1 style="font-size: {title_size} !important; line-height: {line_spacing} !important; margin-bottom: 0.5em !important; font-weight: bold !important;">
        I-LLM
    </h1>
    <h3 style="font-size: {subtitle_size} !important; line-height: {line_spacing} !important; font-weight: bold !important;">
        Trouvez des établissements publics accessibles et obtenez des informations sur le handicap
    </h3>
    <p style="font-size: {font_size} !important; line-height: {line_spacing} !important;">
        Ce chatbot vous aide à trouver des établissements adaptés à vos besoins d'accessibilité et répond à vos questions sur le handicap.
    </p>
    """, unsafe_allow_html=True)

def apply_display_preferences():
    # Récupérer les préférences de la session
    font_size = st.session_state.get("font_size", "16px")
    line_spacing = st.session_state.get("line_spacing", "1.5")
    
    # Vérifier si le mode sombre est activé
    is_dark_mode = st.session_state.get("dark_mode", False)
    
    # Définir les couleurs en fonction du mode
    if is_dark_mode:
        background_color = "#121212"
        text_color = "#FFFFFF"
        secondary_bg = "#1E1E1E"
        border_color = "#333333"
        input_bg = "#2D2D2D"
        hover_bg = "#3D3D3D"
        link_color = "#4DA6FF"
        chat_input_bg = "#FFFFFF"
        header_bg = "#121212"
    else:
        background_color = "#FFFFFF"
        text_color = "#000000"
        secondary_bg = "#F0F2F6"
        border_color = "#CCCCCC"
        input_bg = "#F8F8F8"
        hover_bg = "#EFEFEF"
        link_color = "#1E88E5"
        chat_input_bg = "#F8F8F8"
        header_bg = "#FFFFFF"
    
    # Appliquer les styles CSS 
    css = f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Roboto&family=Open+Sans&family=Lato&family=Montserrat&display=swap');
            
            /* Définition locale pour Comic Sans MS */
            @font-face {{
                font-family: 'Comic Sans MS';
                src: local('Comic Sans MS');
            }}
            
            /* Style pour le corps principal */
            .main .block-container,
            .stApp {{
                background-color: {background_color} !important;
                color: {text_color} !important;
            }}
            
            /* Style pour le Header - partie supérieure */
            header[data-testid="stHeader"],
            .st-emotion-cache-1cypcdb.ea3mdgi2, 
            .st-emotion-cache-1avcm0n.ezrtsby2,
            .st-emotion-cache-10pw50.ea3mdgi1,
            .st-emotion-cache-1dp5vir {{
                background-color: {header_bg} !important;
                color: {text_color} !important;
            }}
            
            /* Styles pour la sidebar */
            section[data-testid="stSidebar"], 
            .st-emotion-cache-16txtl3.eczjsme4 {{
                background-color: {secondary_bg} !important;
                color: {text_color} !important;
            }}
            
            /* Styles pour les widgets */
            .stButton button, 
            .stTextInput input, 
            .stSelectbox, 
            .stTextArea textarea,
            .css-1n76uvr, 
            .css-qbe2hs, 
            .st-emotion-cache-7oyrr6,
            .stCheckbox label {{
                background-color: {secondary_bg} !important;
                color: {text_color} !important;
                border-color: {border_color} !important;
            }}
            
            /* Styles pour sélecteurs et dropdown */
            .stSelectbox > div, 
            .stMultiselect > div {{
                background-color: {secondary_bg} !important;
                color: {text_color} !important;
                border-color: {border_color} !important;
            }}
            
            /* Styles pour les options des sélecteurs */
            .stSelectbox > div > div > div + div > div > div, 
            .stMultiselect > div > div > div + div > div > div {{
                background-color: {input_bg} !important;
                color: {text_color} !important;
            }}
            
            .stButton button:hover {{
                background-color: {hover_bg} !important;
            }}

            /* Styles pour la zone de saisie du chat */
            .st-emotion-cache-1k3j2a5,
            [data-testid="stChatInput"],
            [data-testid="textInputRootElement"],
            .st-emotion-cache-yycrt6 {{
                background-color: {chat_input_bg} !important;
                color: {text_color} !important;
                border-color: {border_color} !important;
            }}
            
            /* Styles pour le placeholder du chat */
            .st-emotion-cache-30do4w.e1nzilvr2 {{
                color: {text_color} !important;
                opacity: 0.7;
            }}
            
            /* Styles de base (taille) */
            html, body, [class*="st-"], .stApp div, .stApp p, .stApp button,
            .stInput input, .stSelectbox, .stTextArea textarea {{
                font-size: {font_size} !important;
            }}
            
            /* Assurer que les titres ont une taille relative à la taille de base */
            h1 {{ font-size: calc({font_size} * 2) !important; }}
            h2 {{ font-size: calc({font_size} * 1.5) !important; }}
            h3 {{ font-size: calc({font_size} * 1.17) !important; }}
            h4 {{ font-size: calc({font_size} * 1.12) !important; }}
            h5 {{ font-size: calc({font_size} * 0.83) !important; }}
            h6 {{ font-size: calc({font_size} * 0.75) !important; }}
            
            /* Styles pour les messages de chat */
            .stChatMessage [data-testid="stChatMessageContent"],
            .st-emotion-cache-1lw0oie {{
                background-color: {secondary_bg} !important;
                color: {text_color} !important;
                border-color: {border_color} !important;
            }}
            
            /* Styles pour les messages de chat utilisateur */
            [data-testid="stChatMessage"] {{
                border-color: {border_color} !important;
            }}
            
            /* Styles pour les expanseurs */
            .streamlit-expanderHeader, 
            .streamlit-expanderContent,
            .st-emotion-cache-1qg05tj {{
                background-color: {secondary_bg} !important;
                color: {text_color} !important;
            }}
            
            /* Styles pour les liens */
            a, a:visited, .st-emotion-cache-q8sbsg {{
                color: {link_color} !important;
            }}
            
            /* Styles pour les textes grisés */
            .st-emotion-cache-92ybdk, 
            .st-emotion-cache-1uhzrxc {{
                color: {text_color} !important;
                opacity: 0.7;
            }}
            
            /* Corrections pour le bouton d'envoi du chat */
            button[data-testid="stChatInputSubmitButton"] {{
                background-color: {secondary_bg} !important;
                color: {text_color} !important;
                border-color: {border_color} !important;
            }}
            
            button[data-testid="stChatInputSubmitButton"]:hover {{
                background-color: {hover_bg} !important;
            }}
            
            /* Style pour les conteneurs des messages */
            .element-container, .stMarkdown {{
                color: {text_color} !important;
            }}
            
            /* Style pour les messages d'info, d'erreur, etc. */
            .stAlert, .stInfo, .stWarning, .stError, .stSuccess {{
                background-color: {secondary_bg} !important;
                color: {text_color} !important;
                border-color: {border_color} !important;
            }}
            
            /* Style pour les switch / toggles */
            .stCheckbox label span {{
                background-color: {input_bg} !important;
                color: {text_color} !important;
            }}
            
            /* Transition douce lors des changements */
            * {{
                transition: all 0.3s ease-in-out;
            }}
            /* Ajout de styles spécifiques pour les labels de widgets */
            .stSelectbox label p, 
            .stTextInput label p,
            .st-emotion-cache-183lzff,
            .st-emotion-cache-10oheav,
            .st-emotion-cache-a4xup4,
            .css-qrbaxs, 
            .st-emotion-cache-1whb5tz p:first-child,
            .st-emotion-cache-1offfbd {{
                color: {text_color} !important;
                opacity: 1 !important;
            }}
            
            /* Labels dans les expanders */
            .streamlit-expanderContent .stSelectbox label p,
            .streamlit-expanderContent .stTextInput label p {{
                color: {text_color} !important;
                opacity: 1 !important;
            }}
            
            /* Labels des sélecteurs et entrées */
            [data-baseweb="select"] + div p,
            [data-baseweb="input"] + div p {{
                color: {text_color} !important;
            }}
            
            /* Labels spécifiques au mode sombre */
            {f'.st-dark [class*="st-"] label, .st-dark [class*="stWidget"] label p' if is_dark_mode else ''} {{
                color: {text_color} !important;
            }}
            /* Correction pour la zone de saisie et son arrière-plan en mode sombre */
            [data-testid="stChatInput"] div,
            [data-testid="stChatInput"] div div,
            [data-testid="stChatInputContainer"],
            .st-emotion-cache-136bz9s,
            .st-emotion-cache-g8be54,
            .st-emotion-cache-90vs21,
            .st-emotion-cache-yf5hs1,
            .st-emotion-cache-7ym5gk,
            .st-emotion-cache-xse5my,
            .st-emotion-cache-9ycgxx,
            .st-emotion-cache-1erivf3 {{
                background-color: {input_bg} !important;
                color: {text_color} !important;
                border-color: {border_color} !important;
            }}
            
            /* Assurer que la zone autour du champ de saisie est correctement colorée */
            [data-testid="stChatMessageInput"],
            [data-testid="chatAvatarIcon-user"],
            .st-emotion-cache-1vbkxwb,
            .st-emotion-cache-z5fcl4 {{
                background-color: {background_color} !important;
            }}
            
            /* Cibler également les conteneurs parents */
            footer[data-testid="stFooter"],
            footer[data-testid="stFooter"] > div,
            .st-emotion-cache-1rs6os.edgvbvh3,
            .st-emotion-cache-cio0dv.ea3mdgi1 {{
                background-color: {background_color} !important;
            }}
            
            /* Styles spécifiques pour le placeholder du chat input */
            ::placeholder,
            [data-testid="stChatInput"] input::placeholder,
            [data-testid="stChatInput"] textarea::placeholder,
            .st-emoji-cache-1n76uvr::placeholder,
            .st-emotion-cache-1u3r8l4::placeholder,
            .st-emotion-cache-w1wt0n::placeholder,
            [data-testid="stChatInputTextArea"]::placeholder,
            textarea[data-testid="stChatInputTextArea"]::placeholder {{
                color: {text_color} !important;
                opacity: 0.7 !important;
            }}

            /* Ciblage supplémentaire pour le texte du placeholder */
            .stTextArea::placeholder, 
            .stTextInput::placeholder,
            [placeholder="Posez votre question sur l'accessibilité ou recherchez un établissement..."] {{
                color: {text_color} !important;
                opacity: 0.7 !important;
            }}
            /* Ciblage supplémentaire pour le texte du placeholder */
            .stTextArea::placeholder, 
            .stTextInput::placeholder,
            [placeholder="Posez votre question sur l'accessibilité ou recherchez un établissement..."] {{
                color: {text_color} !important;
                opacity: 0.7 !important;
            }}
            
            /* Style pour le footer et conteneur de bas de page */
            div[data-testid="stBottomBlockContainer"],
            footer[data-testid="stFooter"],
            .main footer {{
                background-color: {background_color} !important;
                color: {text_color} !important;
            }}
            
        </style>
    """
    
    # Utiliser st.markdown pour injecter le CSS
    st.markdown(css, unsafe_allow_html=True)

# Fonction pour appliquer les préférences d'affichage universellement
def apply_universal_font():
    font_family = st.session_state.get("font_family", "sans-serif")
    
    # CSS ultra-spécifique qui cible vraiment tout
    st.markdown(f"""
    <style>
        /* Force ALL elements to use the selected font */
        html *,
        body *,
        .stApp *,
        [class*="st"] *,
        .element-container *,
        header *,
        .main *,
        .stHeader *,
        .stMarkdown *,
        div *,
        span *,
        p *,
        h1 *, h1,
        h2 *, h2,
        h3 *, h3,
        h4 *, h4,
        h5 *, h5,
        h6 *, h6,
        button *,
        input *,
        select *,
        textarea *,
        .stButton *,
        label *,
        .stSidebar * {{
            font-family: {font_family}, sans-serif !important;
        }}
        
        /* Ciblage spécifique des headers streamlit */
        .css-10trblm, .css-1vbkxwb, .css-edivx2,
        .css-16idsys p, 
        .css-1aehpvj, .css-1v3fvcr {{
            font-family: {font_family}, sans-serif !important;
        }}
    </style>
    """, unsafe_allow_html=True)

# Fonction pour basculer le mode sombre
def toggle_dark_mode():
    # Basculer le mode sombre
    st.session_state.dark_mode = not st.session_state.get("dark_mode", False)
    # Indiquer qu'un rechargement est nécessaire
    st.session_state.need_rerun = True

# Fonction principale
def main():
    # Initialiser le mode sombre s'il n'est pas déjà défini
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False

    # Vérifier si un rechargement est nécessaire
    if st.session_state.get("need_rerun", False):
        st.session_state.need_rerun = False
        st.rerun()
        
    apply_display_preferences()
    apply_universal_font()
    display_header()
    
    # Paramètres en haut dans un header
    with st.expander("⚙️ Paramètres de configuration", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            # Champ pour la clé API (masqué) avec valeur par défaut de session_state
            gemini_api_key = st.text_input("Clé API Gemini", 
                                    value=st.session_state.get("gemini_api_key", ""),
                                    type="password", 
                                    help="Entrez votre clé API Gemini pour activer le chatbot")

        with col2:
            # Sélection du fichier de données avec valeur par défaut de session_state
            api_base_url = st.text_input("Lien de l'API pour le dataset", 
                                        value=st.session_state.get("api_base_url", "https://tabular-api.data.gouv.fr/api/resources/93ae96a7-1db7-4cb4-a9f1-6d778370b640/data/"),
                                        help="Lien de l'API pour le dataset des établissements accessibles")
        
        # Bouton pour appliquer les paramètres
        if st.button("Appliquer les paramètres"):
            st.success("Paramètres appliqués avec succès!")
            st.session_state.gemini_api_key = gemini_api_key
            st.session_state.api_base_url = api_base_url
            st.session_state.need_rerun = True

    # Dans la fonction main(), après la saisie des paramètres
    if gemini_api_key:
        # Stocker la clé API dans session_state si elle n'y est pas déjà
        if "gemini_api_key" not in st.session_state or not st.session_state.gemini_api_key:
            st.session_state.gemini_api_key = gemini_api_key

    if api_base_url:
        # Stocker l'URL de l'API dans session_state si elle n'y est pas déjà
        if "api_base_url" not in st.session_state or not st.session_state.api_base_url:
            st.session_state.api_base_url = api_base_url

    # Sidebar pour l'historique des conversations
    with st.sidebar:
        # Ajout du bouton de mode sombre en haut de la sidebar
        dark_mode_text = "🌙 Mode Sombre" if not st.session_state.get("dark_mode", False) else "☀️ Mode Clair"
        st.button(dark_mode_text, on_click=toggle_dark_mode, key="dark_mode_toggle")
        
        # Ajout des préférences d'affichage dans la sidebar
        with st.expander("👁️ Préférences d'affichage", expanded=False):
            st.markdown("### Personnalisez l'affichage")
            
            # Choix de la police
            font_options = ["Roboto", "Open Sans", "Lato", "Montserrat", "Comic Sans MS", "sans-serif"]
            selected_font = st.selectbox(
                "Police d'affichage",
                options=font_options,
                index=font_options.index(st.session_state.get("font_family", "sans-serif")) if st.session_state.get("font_family", "sans-serif") in font_options else 0,
                key="font_selector"
            )
            # Ajouter une info pour Comic Sans MS
            if selected_font == "Comic Sans MS":
                st.info("💡 La police Comic Sans MS peut faciliter la lecture pour les personnes dyslexiques grâce à ses caractères plus distincts.")
            
            # Taille de la police
            font_size_options = ["12px", "14px", "16px", "18px", "20px", "22px", "24px"]
            selected_size = st.selectbox(
                "Taille de police",
                options=font_size_options,
                index=font_size_options.index(st.session_state.get("font_size", "16px")) if st.session_state.get("font_size", "16px") in font_size_options else 2,
                key="size_selector"
            )

            # Bouton pour appliquer les changements
            if st.button("Appliquer les préférences", key="apply_preferences_btn"):
                # Mettre à jour les valeurs dans session_state
                st.session_state.font_family = selected_font
                st.session_state.font_size = selected_size
                # Appliquer les préférences
                st.success("Préférences d'affichage appliquées avec succès!")
                st.session_state.need_rerun = True
        
        
        # Le reste du code de la sidebar...
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
                        st.session_state.need_rerun = True
                    
                    # Option pour supprimer cette conversation
                    if st.button(f"Supprimer", key=f"delete_{i}"):
                        st.session_state.conversation_history.pop(i)
                        st.session_state.need_rerun = True
        
        # Bouton pour créer une nouvelle conversation
        if st.button("➕ Nouvelle conversation"):
            if "messages" in st.session_state and len(st.session_state.messages) > 1:
                # Sauvegarde de la conversation actuelle si elle existe
                st.session_state.conversation_history.append({
                    "date": time.strftime("%d/%m/%Y %H:%M"),
                    "messages": st.session_state.messages.copy()
                })
            
            # Réinitialisation des messages
            st.session_state.messages = [
                {"role": "assistant", "content": "Bonjour ! Je suis votre assistant virtuel pour l'accessibilité. Comment puis-je vous aider aujourd'hui ?"}
            ]
            st.session_state.need_rerun = True
        
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
    gemini_api_key = st.session_state.get("gemini_api_key", "")
    api_base_url = st.session_state.get("api_base_url", "")
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
            
            # Zone de saisie pour la question de l'utilisateur
            if prompt := st.chat_input("Posez votre question sur l'accessibilité ou recherchez un établissement..."):
                # Ajout du message de l'utilisateur à l'historique
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Affichage du message de l'utilisateur
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Affichage du message de l'assistant avec animation de chargement
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    message_placeholder.markdown("🔍 Recherche en cours...")
                    
                    # Obtention de la réponse du chatbot
                    try:
                        response = chatbot.process_query(prompt)
                        
                        # Animation de chargement (optionnel)
                        with st.spinner("Génération de la réponse..."):
                            # Création d'une animation de saisie
                            for i in range(len(response) // 10):
                                partial_response = response[:i*10]
                                message_placeholder.markdown(f"{partial_response}▌")
                                time.sleep(0.01)
                            
                            # Affichage de la réponse complète
                            message_placeholder.markdown(response)
                            
                        # Ajout de la réponse à l'historique
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                    except Exception as e:
                        message_placeholder.markdown(f"⚠️ Désolé, une erreur s'est produite : {str(e)}")
            
            # Bouton pour sauvegarder la conversation actuelle
            if len(st.session_state.messages) > 1:  # S'il y a plus que le message initial
                if st.button("💾 Sauvegarder cette conversation"):
                    if "conversation_history" not in st.session_state:
                        st.session_state.conversation_history = []
                    
                    st.session_state.conversation_history.append({
                        "date": time.strftime("%d/%m/%Y %H:%M"),
                        "messages": st.session_state.messages.copy()
                    })
                    
                    st.success("Conversation sauvegardée!")
            
            # Section d'exemples de questions
            with st.expander("📝 Exemples de questions"):
                st.markdown("""
                Voici quelques exemples de questions que vous pouvez poser:
                
                **Recherche d'établissements:**
                - "Où puis-je trouver un restaurant accessible en fauteuil roulant à Paris ?"
                - "Y a-t-il un musée avec audiodescription à Lyon ?"
                - "Je cherche une piscine avec stationnement PMR à Bordeaux"
                
                **Questions générales:**
                - "Quelles sont les aides financières pour les personnes handicapées ?"
                - "Comment fonctionne la MDPH ?"
                - "Quels sont mes droits en tant que personne malvoyante ?"
                """)
                
                # Boutons pour poser directement les questions d'exemple
                col1, col2 = st.columns(2)
                if col1.button("Restaurants accessibles à Paris"):
                    st.session_state.messages.append({"role": "user", "content": "Où puis-je trouver un restaurant accessible en fauteuil roulant à Paris ?"})
                    st.session_state.need_rerun = True
                
                if col2.button("Aides financières"):
                    st.session_state.messages.append({"role": "user", "content": "Quelles sont les aides financières pour les personnes handicapées ?"})
                    st.session_state.need_rerun = True
                    
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