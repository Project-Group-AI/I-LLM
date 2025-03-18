# Example-HyDE

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
  - Recherche dans le dataset ces critères et ressort dans un premier dans uniquement les informations qui correspondent à tt les critères sinon refait une recherche dans le dataset avec +-1 critère
  - Envoie les données qui sont extraites du dataset au LLM qui lui reformule proprement et naturellement
  - Et affiche le résultat
- Hors context : Renvoie une phrase prédéfini qui demande de poser une question en rapport avec notre context




### Result

### Explanation
