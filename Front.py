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

# Fichiers de stockage
CONVERSATIONS_FILE = "conversations.json"
DATASETS_FILE = "datasets.json"       # Fichier de stockage des datasets
PROVIDERS_FILE = "providers.json"     # Fichier de stockage des providers

# --- Gestion des conversations ---
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

# --- Gestion des datasets ---
def load_datasets():
    if os.path.exists(DATASETS_FILE):
        with open(DATASETS_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    else:
        # Datasets par défaut
        return {
            "AccesLibre": {
                "base_url": "https://tabular-api.data.gouv.fr/api/resources/93ae96a7-1db7-4cb4-a9f1-6d778370b640/data/",
                "help_text": "Dataset officiel des établissements accessibles"
            }
        }

def save_datasets(datasets):
    with open(DATASETS_FILE, "w", encoding="utf-8") as f:
        json.dump(datasets, f, ensure_ascii=False, indent=4)

# --- Gestion des providers ---
def load_providers():
    if os.path.exists(PROVIDERS_FILE):
        with open(PROVIDERS_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    else:
        # Providers par défaut
        return {
            "Gemini 2.0 Flash": {
                "key_label": "Clé API Gemini",
                "help_text": "Entrez votre clé API Gemini pour activer le chatbot",
                "model_name": "gemini-2.0-flash"
            }
        }

def save_providers(providers):
    with open(PROVIDERS_FILE, "w", encoding="utf-8") as f:
        json.dump(providers, f, ensure_ascii=False, indent=4)

# --- Initialisation du chatbot ---
@st.cache_resource
def initialize_chatbot(api_base_url, api_key):
    with st.spinner("Initialisation du chatbot en cours..."):
        return ChatbotInclusifGemini(api_base_url, api_key)

# --- Affichage de l'en-tête ---
def display_header():
    st.title("I-LLM")
    st.markdown("""
    ### Trouvez des établissements publics accessibles et obtenez des informations sur le handicap
    Ce chatbot vous aide à trouver des établissements adaptés à vos besoins d'accessibilité et répond à vos questions sur le handicap.
    """)

# --- Gestion du thème ---
def change_theme():
    previous_theme = st.session_state.themes["current_theme"]
    new_theme = "dark" if previous_theme == "light" else "light"
    st.session_state.themes["current_theme"] = new_theme  # Mise à jour de l'état
    tdict = st.session_state.themes[new_theme]  # Récupérer le dictionnaire du nouveau thème
    for vkey, vval in tdict.items():
        if vkey.startswith("theme"):
            st._config.set_option(vkey, vval)
    st.session_state["selected_font"] = st.session_state.get("selected_font", "police de base")
    st.session_state.themes["refreshed"] = True
    st.session_state.run_rerun = True  # Flag pour forcer le rerun

def apply_font_size():
    st.session_state["font_size"] = st.session_state["font_size_draft"]

# --- Génération d'un titre pour la sauvegarde d'une conversation ---
def generate_title(messages):
    """Génère un titre court résumant la demande générale de l'utilisateur en préservant la casse originale."""
    import re
    for m in messages:
        if m.get("role") == "user" and m.get("content"):
            text = m.get("content").strip()
            text_no_punct = re.sub(r'[^\w\sÀ-ÿ]', '', text)
            original_words = text_no_punct.split()
            stopwords = {
                "quels", "quelles", "sont", "mes", "en", "tant", "que", "les", "des",
                "de", "le", "la", "et", "pour", "a", "un", "une", "ces", "ce", "est",
                "où", "puisje", "trouver", "je", "cherche", "chercher", "personne", "dont", "au", "aux",
                "y", "atil"
            }
            keywords = [word for word in original_words if word.lower() not in stopwords]
            loc_match = re.search(r' à ([\wÀ-ÿ]+)', text, re.IGNORECASE)
            location = loc_match.group(1).strip() if loc_match else ""
            keywords = [word for word in keywords if word.lower() != location.lower()]
            summary_keywords = keywords[:3]
            title = " ".join(summary_keywords)
            if location:
                title += " à " + location
            if title:
                title = title[0].upper() + title[1:]
            return title
    import time
    return time.strftime("%d/%m/%Y %H:%M")

# --- Fonction principale ---
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
    if "font_size" not in st.session_state:
        st.session_state["font_size"] = 16  # Taille de police par défaut
    if "font_size_draft" not in st.session_state:
        st.session_state["font_size_draft"] = st.session_state["font_size"]

    display_header()

    # Détermine la police à utiliser
    if st.session_state["selected_font"] == "Comic Sans MS":
        css_font = "'Comic Sans MS', cursive, sans-serif"
    else:
        css_font = "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"

    # Injection du CSS pour appliquer la police et la taille sur tous les éléments
    st.markdown(
        f"""
        <style>
        * {{
            font-family: {css_font} !important;
            font-size: {st.session_state['font_size']}px !important;
        }}
        h1 {{ font-size: {st.session_state['font_size'] * 2}px !important; }}
        h2 {{ font-size: {st.session_state['font_size'] * 1.5}px !important; }}
        h3 {{ font-size: {st.session_state['font_size'] * 1.3}px !important; }}
        small {{ font-size: {st.session_state['font_size'] * 0.8}px !important; }}
        </style>
        """,
        unsafe_allow_html=True
    )
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

    # --- Chargement des datasets et providers depuis leurs fichiers JSON ---
    datasets = load_datasets()
    providers = load_providers()

    # --- Expander pour la configuration des paramètres ---
    with st.expander("⚙️ Paramètres de configuration", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            # Sélection du provider
            selected_provider = st.selectbox(
                "Fournisseur de modèle",
                options=list(providers.keys()),
                index=0,
                key="provider_select",
                help="Sélectionnez le fournisseur de modèle IA à utiliser"
            )
            provider_config = providers[selected_provider]
            api_key = st.text_input(
                provider_config["key_label"],
                type="password",
                value=st.session_state.get("api_key", ""),
                help=provider_config["help_text"]
            )

        with col2:
            # Sélection du dataset
            selected_dataset = st.selectbox(
                "Source de données",
                options=list(datasets.keys()),
                index=0,
                key="dataset_select",
                help="Sélectionnez la base de données à interroger"
            )
            dataset_config = datasets[selected_dataset]
            api_base_url = st.text_input(
                "URL de l'API",
                value=st.session_state.get("api_base_url", dataset_config["base_url"]),
                help=dataset_config["help_text"]
            )

        if st.button("Appliquer les paramètres"):
            if api_key and api_base_url:
                st.session_state.update({
                    "api_key": api_key,
                    "api_base_url": api_base_url,
                    "selected_provider": selected_provider,
                    "selected_dataset": selected_dataset,
                    "provider_config": provider_config,
                    "dataset_config": dataset_config
                })
                st.success("Paramètres appliqués avec succès!")
                st.rerun()
            else:
                st.error("Veuillez remplir tous les champs obligatoires")
    
    # --- Expander pour ajouter un nouveau dataset ---
    with st.expander("Ajouter un nouveau dataset"):
        new_dataset_name = st.text_input("Nom du dataset", key="new_dataset_name")
        new_dataset_url = st.text_input("URL du dataset", key="new_dataset_url")
        new_dataset_help = st.text_input("Description du dataset", key="new_dataset_help")
        
        if st.button("Ajouter dataset"):
            if new_dataset_name and new_dataset_url:
                datasets[new_dataset_name] = {
                    "base_url": new_dataset_url,
                    "help_text": new_dataset_help
                }
                save_datasets(datasets)
                st.success("Dataset ajouté et enregistré!")
                st.rerun()
            else:
                st.error("Veuillez renseigner le nom et l'URL du dataset")
    
    # --- Expander pour gérer (modifier/supprimer) les datasets existants ---
    with st.expander("Gérer les datasets"):
        for ds_name in list(datasets.keys()):
            st.markdown(f"#### {ds_name}")
            new_url = st.text_input("URL du dataset", value=datasets[ds_name]["base_url"], key=f"url_{ds_name}")
            new_help = st.text_input("Description du dataset", value=datasets[ds_name]["help_text"], key=f"help_{ds_name}")
            col_mod, col_del = st.columns(2)
            with col_mod:
                if st.button("Modifier dataset", key=f"mod_{ds_name}"):
                    datasets[ds_name]["base_url"] = new_url
                    datasets[ds_name]["help_text"] = new_help
                    save_datasets(datasets)
                    st.success(f"Dataset {ds_name} modifié!")
                    st.rerun()
            with col_del:
                if st.button("Supprimer dataset", key=f"del_{ds_name}"):
                    del datasets[ds_name]
                    save_datasets(datasets)
                    st.success(f"Dataset {ds_name} supprimé!")
                    st.rerun()
    
    # --- Expander pour ajouter un nouveau provider ---
    with st.expander("Ajouter un nouveau provider"):
        new_provider_name = st.text_input("Nom du provider", key="new_provider_name")
        new_key_label = st.text_input("Label pour la clé API", key="new_key_label")
        new_help_text = st.text_input("Description du provider", key="new_help_text")
        new_model_name = st.text_input("Nom du modèle", key="new_model_name")
        if st.button("Ajouter provider"):
            if new_provider_name and new_key_label and new_model_name:
                providers[new_provider_name] = {
                    "key_label": new_key_label,
                    "help_text": new_help_text,
                    "model_name": new_model_name
                }
                save_providers(providers)
                st.success("Provider ajouté et enregistré!")
                st.rerun()
            else:
                st.error("Veuillez renseigner le nom du provider, le label et le nom du modèle")
    
    # --- Expander pour gérer (modifier/supprimer) les providers existants ---
    with st.expander("Gérer les providers"):
        for prov_name in list(providers.keys()):
            st.markdown(f"#### {prov_name}")
            new_label = st.text_input("Label pour la clé API", value=providers[prov_name]["key_label"], key=f"label_{prov_name}")
            new_help = st.text_input("Description du provider", value=providers[prov_name]["help_text"], key=f"prov_help_{prov_name}")
            new_model = st.text_input("Nom du modèle", value=providers[prov_name]["model_name"], key=f"model_{prov_name}")
            col_mod, col_del = st.columns(2)
            with col_mod:
                if st.button("Modifier provider", key=f"mod_prov_{prov_name}"):
                    providers[prov_name]["key_label"] = new_label
                    providers[prov_name]["help_text"] = new_help
                    providers[prov_name]["model_name"] = new_model
                    save_providers(providers)
                    st.success(f"Provider {prov_name} modifié!")
                    st.rerun()
            with col_del:
                if st.button("Supprimer provider", key=f"del_prov_{prov_name}"):
                    del providers[prov_name]
                    save_providers(providers)
                    st.success(f"Provider {prov_name} supprimé!")
                    st.rerun()
    
    # --- Chargement de l'historique des conversations ---
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = load_conversations()
    
    # --- Sidebar : Historique des conversations et paramètres supplémentaires ---
    with st.sidebar:
        btn_face = (
            ms.themes["light"]["button_face"] 
            if ms.themes["current_theme"] == "light" 
            else ms.themes["dark"]["button_face"]
        )
        st.button(f"{btn_face} Changer de thème", on_click=change_theme)
        st.selectbox("Police du site", ["police de base", "Comic Sans MS"], key="selected_font")
        font_size_draft = st.slider(
            "Ajustez la taille de police", 
            min_value=12, 
            max_value=24, 
            value=st.session_state["font_size_draft"], 
            key="font_size_draft_slider"
        )
        st.session_state["font_size_draft"] = font_size_draft
        st.button("Appliquer la taille de police", on_click=apply_font_size, key="apply_font_size_btn")
        st.divider()
        st.header("📜 Historique des conversations")
        if not st.session_state.conversation_history:
            st.info("Aucune conversation enregistrée.")
        else:
            for i, conversation in enumerate(st.session_state.conversation_history):
                with st.expander(f"{conversation.get('title', f'Conversation {i+1}')} - {conversation['date']}"):
                    for message in conversation["messages"]:
                        st.markdown(f"**{message['role'].capitalize()}**: {message['content'][:50]}...")
                    if st.button(f"Charger cette conversation", key=f"load_{i}"):
                        st.session_state.messages = conversation["messages"].copy()
                        st.rerun()
                    if st.button(f"Supprimer", key=f"delete_{i}"):
                        st.session_state.conversation_history.pop(i)
                        save_conversations(st.session_state.conversation_history)
                        st.rerun()
        if st.button("➕ Nouvelle conversation"):
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
    
    # --- Initialisation du chatbot et gestion de la conversation ---
    if st.session_state.get("api_key") and st.session_state.get("api_base_url"):
        try:
            chatbot = initialize_chatbot(st.session_state["api_base_url"], st.session_state["api_key"])
            st.success("✅ Chatbot initialisé avec succès!")
            if "messages" not in st.session_state:
                st.session_state.messages = [
                    {"role": "assistant", "content": "Bonjour ! Je suis votre assistant virtuel pour l'accessibilité. Comment puis-je vous aider aujourd'hui ?"}
                ]
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
                st.markdown("**Recherche d'établissements:**")
                if st.button("Où puis-je trouver un restaurant accessible en fauteuil roulant à Paris ?"):
                    st.session_state.selected_question = "Où puis-je trouver un restaurant accessible en fauteuil roulant à Paris ?"
                    st.rerun()
                if st.button("Y a-t-il un musée avec audiodescription à Lyon ?"):
                    st.session_state.selected_question = "Y a-t-il un musée avec audiodescription à Lyon ?"
                    st.rerun()
                if st.button("Je cherche une piscine avec stationnement PMR à Bordeaux"):
                    st.session_state.selected_question = "Je cherche une piscine avec stationnement PMR à Bordeaux"
                    st.rerun()
                st.markdown("**Questions générales:**")
                if st.button("Quelles sont les aides financières pour les personnes handicapées ?"):
                    st.session_state.selected_question = "Quelles sont les aides financières pour les personnes handicapées ?"
                    st.rerun()
                if st.button("Comment fonctionne la MDPH ?"):
                    st.session_state.selected_question = "Comment fonctionne la MDPH ?"
                    st.rerun()
                if st.button("Quels sont mes droits en tant que personne malvoyante ?"):
                    st.session_state.selected_question = "Quels sont mes droits en tant que personne malvoyante ?"
                    st.rerun()
                st.markdown("**Outils disponibles:**")
                st.markdown("Recherche google maps:")
                if st.button("Je cherche un restaurant accessible PMR à Lille, donne moi le liens google maps pour savoir ou il se situe."):
                    st.session_state.selected_question = "Je cherche un restaurant accessible PMR à Lille, génére moi une carte maps pour savoir ou il se situe."
                    st.rerun()
                    
        except Exception as e:
            st.error(f"Erreur lors de l'initialisation du chatbot: {str(e)}")
            st.info("Vérifiez que l'URL de l'API est correct et que votre clé API est valide.")
    else:
        st.warning("⚠️ Veuillez entrer votre clé API Gemini et l'URL de l'API pour initialiser le chatbot.")
        st.image("https://via.placeholder.com/800x400?text=Chatbot+Inclusif+Demo", caption="Aperçu du chatbot")
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

if __name__ == "__main__":
    main()
