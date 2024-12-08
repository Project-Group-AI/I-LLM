# Example-HyDE

## Functioning
- Generate a hypothetical answer:
Use an LLM (like Mistral 7B) to produce a hypothetical answer based on the user's question.
- Embedding and research:
Encode this hypothetical answer (instead of the original question).
Search the knowledge base for relevant documents by comparing embeddings.
- Generate final answer:
Pass the initial question, along with relevant documents, to an LLM to generate a contextual answer.

## Difference between the RAG and HyDE method
The main difference is that RAG directly compares the question with the documents while HyDE generates a hypothetical answer and compares it to these documents. the hypothetical answer allows us to obtain search more efficiently and with greater context.
for example with RAG, when in the document there was "One of the famous grand masters is Garry Kasparov.", the RAG method does not understand that this document is related to chess, while with HyDE this time we understand that this document is related to chess with to the hypothetical response generated before.

### Result
![CleanShot 2024-12-08 at 20 06 05@2x](https://github.com/user-attachments/assets/bb12fa11-0dd2-4f97-96ba-6d7d741b4d50)
![CleanShot 2024-12-08 at 20 06 17@2x](https://github.com/user-attachments/assets/9656aacd-02ed-4b43-b287-11e65ecb4889)

But if i provided false information inside doc, the answer will not use these documents because the hypothetical answer does not confirm them. like for example here by putting a fake chess Grandmaster.
![CleanShot 2024-12-08 at 20 07 52@2x](https://github.com/user-attachments/assets/d8f12410-4ab0-4897-b8a5-097c77f1157b)
![CleanShot 2024-12-08 at 20 08 02@2x](https://github.com/user-attachments/assets/80521a86-157c-4767-8007-c6ba72688d85)

### Explanation
the hypothetical response generated to be compared with the documents had to output the name of "Garry Kasparov", so if I put his name in a document without citing the chess the HyDE method will understand that this document is relevant but if I replace "Garry Kasparov" by "Romain Dujardin" so this time no link is made between the hypothetical response and this document and therefore it is not used
