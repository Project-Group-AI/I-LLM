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
    st.title("I-LLM")
    st.markdown("""
    ### Trouvez des établissements publics accessibles et obtenez des informations sur le handicap
    Ce chatbot vous aide à trouver des établissements adaptés à vos besoins d'accessibilité et répond à vos questions sur le handicap.
    """)

# Fonction principale
def main():
    display_header()
    
    # Paramètres en haut dans un header
    with st.expander("⚙️ Paramètres de configuration", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            # Champ pour la clé API (masqué)
             gemini_api_key = st.text_input("Clé API Gemini", type="password", 
                                  help="Entrez votre clé API Gemini pour activer le chatbot")
        
        with col2:
            # Sélection du fichier de données
            api_base_url = st.text_input("Lien de l'API pour le dataset", 
                                       value="https://tabular-api.data.gouv.fr/api/resources/93ae96a7-1db7-4cb4-a9f1-6d778370b640/data/",
                                       help="Lien de l'API pour le dataset des établissements accessibles")
        
        # Bouton pour appliquer les paramètres
        if st.button("Appliquer les paramètres"):
            st.success("Paramètres appliqués avec succès!")
            # On met à jour une variable d'état plutôt que de redémarrer l'app
            st.session_state. gemini_api_key =  gemini_api_key
            st.session_state.api_base_url = api_base_url
            # Redirection conditionnelle ou gestion de l'état
            st.experimental_rerun()  # Cela peut être potentiellement supprimé si cela devient redondant.

    
    # Sidebar pour l'historique des conversations
    with st.sidebar:
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
                        st.experimental_rerun()
                    
                    # Option pour supprimer cette conversation
                    if st.button(f"Supprimer", key=f"delete_{i}"):
                        st.session_state.conversation_history.pop(i)
                        st.experimental_rerun()
        
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
            st.experimental_rerun()
        
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
                    st.experimental_rerun()
                
                if col2.button("Aides financières"):
                    st.session_state.messages.append({"role": "user", "content": "Quelles sont les aides financières pour les personnes handicapées ?"})
                    st.experimental_rerun()
                    
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