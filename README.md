# Example-HyDE

## Lien
[https://example-hyde.streamlit.app/](https://example-hyde.streamlit.app/?embed_options=dark_theme)

## Fonctionnement
- Chargement du dataset
- Fonction pour détecter la pertinance de la question
  - General question (question basé sur le sujet global du handicap)
  - Recherche question (question sur la recherche d'un lieu)
  - Hors context (question qui n'a rien a voir avec le contexte)

- General Question : Je l'envoie directement au LLM avec un prompt détaillé pour avoir une reponse optimal
- Recherche Question :
  - Création d'un document hypothétique via le LLM en sa basant sur les colonnes du dataset
  - Extraction des informations du document (ville, activité, handicap)
  - Recherche dans le dataset ces critères et ressort dans un premier temps dans uniquement les informations qui correspondent à tt les critères sinon refait une recherche dans le dataset avec +-1 critère
  - Envoie les données qui sont extraites du dataset au LLM qui lui reformule proprement et naturellement
  - Et affiche le résultat
- Hors context : Renvoie une phrase prédéfini qui demande de poser une question en rapport avec notre context

### Result
**Recherche question**
![CleanShot 2025-03-18 at 08 53 52@2x](https://github.com/user-attachments/assets/fa6b6a2b-a201-460e-a102-9dec9c05b3b6)

**Global question**
![CleanShot 2025-03-18 at 08 54 28@2x](https://github.com/user-attachments/assets/f4969e15-344c-464d-a7a0-ff76f0b7bdbc)

**Hors context**
![CleanShot 2025-03-18 at 08 53 01@2x](https://github.com/user-attachments/assets/cf67b6a9-5c74-4942-a017-76874b874060)


### Explanation
