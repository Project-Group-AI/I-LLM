import requests
from huggingface_hub import HfFolder
from sentence_transformers import SentenceTransformer
import faiss

# Initialization of the embedding model and the FAISS index
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Embedding template
documents = [
    "One of the famous grand masters is Garry Kasparov.",
    "Machine learning is a subfield of AI that focuses on learning algorithms.",
    "Deep learning uses deep neural networks to solve complex problems.",
    "Chess is a two-player strategy game."
]

# Create embeddings for each document
doc_embeddings = embedding_model.encode(documents)

# Building a FAISS index
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)

# Step 1: Use Mistral to generate a hypothetical response
def generate_hypothetical_answer(prompt):
    """
    Generates a hypothetical response with Mistral via the Hugging Face API.
    """
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
    HF_TOKEN = HfFolder.get_token()
    if HF_TOKEN is None:
        return "Error: No tokens found. Log in with `huggingface-cli login`."

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 500,
            "temperature": 0.7,
            "top_k": 50,
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    else:
        return f"Error : {response.status_code} - {response.json()}"

# Step 2: Search for relevant documents
def find_relevant_docs(hypothetical_answer, k=2):
    """
    Searches for relevant documents based on the hypothetical answer.
    """
    hypo_embedding = embedding_model.encode([hypothetical_answer])
    distances, indices = index.search(hypo_embedding, k)
    return [documents[idx] for idx in indices[0]]

# Step 3: Generating the final response with Mistral
def generate_final_answer(prompt, relevant_docs):
    """
    Generates a final answer based on the relevant documents and the original question.
    """
    context = "\n".join(relevant_docs)
    final_prompt = f"Context : {context}\n\nQuestion : {prompt}\n\nRéponse :"
    return generate_hypothetical_answer(final_prompt)

# Complete HyDE pipeline
def hyde_pipeline(query, k=2):
    """
    Implements the HyDE method.
    """
    # Generate a hypothetical answer
    hypothetical_answer = generate_hypothetical_answer(query)
    if hypothetical_answer.startswith("Error"):
        return hypothetical_answer

    # Search for relevant documents
    relevant_docs = find_relevant_docs(hypothetical_answer, k)

    # Generate a final answer based on the original question and documents
    return generate_final_answer(query, relevant_docs)

# Example of use
if __name__ == "__main__":
    query = "What is chess?"
    answer = hyde_pipeline(query)
    print("Response generated:")
    print(answer)
