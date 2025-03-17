# Example-HyDE

## Fonctionnement
- Extraction automatique des critères à partir de la requete
![CleanShot 2025-03-17 at 18 30 25@2x](https://github.com/user-attachments/assets/feb80c92-77ed-4021-ac5d-87319fc3fd21)

- j'élargie le champs des possibilitées pour la détection dans la requete 
![CleanShot 2025-03-17 at 18 34 05@2x](https://github.com/user-attachments/assets/bcdfa206-0901-498e-90dc-793ec3c643b4)

- Je crée un document hypothétique en me basant sur les colonnes du dataset, donc par exemple je sais que une personne en mobilité réduite aura besoin d'une entrée pmr + un cheminement extérieur plein pied + un stationnement pmr. ainsi je crée des critères pour chaque cas.
![CleanShot 2025-03-17 at 18 35 52@2x](https://github.com/user-attachments/assets/5f36f511-8fb4-4f58-8d9b-e1943af2822c)

- Ensuite je viens donc basé sur mes critères regarder ceux qui correspond dans mon dataset, le fait d'établir avant ces critères de filtrer la recherche et de ne pas chercher à chaque fois dans tout le dataset.
![CleanShot 2025-03-17 at 18 41 29@2x](https://github.com/user-attachments/assets/a21c0c25-f3e1-4ffb-8eb0-cc3a918c8f6a)



## Difference between the RAG and HyDE method


### Result

### Explanation
the hypothetical response generated to be compared with the documents had to output the name of "Garry Kasparov", so if I put his name in a document without citing the chess the HyDE method will understand that this document is relevant but if I replace "Garry Kasparov" by "Romain Dujardin" so this time no link is made between the hypothetical response and this document and therefore it is not used
