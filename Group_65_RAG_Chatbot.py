import streamlit as st
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import numpy as np
import re
from transformers import pipeline

# Load financial data
with open("financial_data.txt", "r") as f:
    financial_text = f.read()

# Preprocess and chunk data
sentences = [s.strip() for s in financial_text.split("\n") if s.strip()]
bm25 = BM25Okapi([s.split() for s in sentences])
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(sentences, convert_to_tensor=True)

# Load a small open-source LLM for response generation
llm = pipeline("text-generation", model="databricks/dolly-v2-3b", trust_remote_code=True)

def hybrid_search(query, top_k=3):
    # BM25 Sparse Retrieval
    bm25_scores = bm25.get_scores(query.split())
    bm25_top_indices = np.argsort(bm25_scores)[::-1][:top_k]
    
    # Dense Vector Retrieval
    query_embedding = embedding_model.encode([query], convert_to_tensor=True)
    dense_scores = np.inner(query_embedding, embeddings)[0]
    dense_top_indices = np.argsort(dense_scores)[::-1][:top_k]
    
    # Combine results
    final_indices = list(set(bm25_top_indices) | set(dense_top_indices))
    retrieved_sentences = [sentences[i] for i in final_indices]
    return "\n".join(retrieved_sentences)

def validate_input(user_input):
    if len(user_input) < 5:
        return False, "Query is too short. Please provide more details."
    if re.search(r'[^a-zA-Z0-9\s$%.,?-]', user_input):
        return False, "Invalid characters detected. Please rephrase your question."
    return True, ""

def generate_response(context, query):
    prompt = f"Based on the following financial data:\n{context}\n\nAnswer the question: {query}"
    response = llm(prompt, max_length=100, num_return_sequences=1)
    return response[0]['generated_text']

# Streamlit UI
st.title("Financial RAG Chatbot")
query = st.text_input("Enter your financial question:")

if query:
    valid, message = validate_input(query)
    if not valid:
        st.warning(message)
    else:
        context = hybrid_search(query)
        response = generate_response(context, query)
        st.write("### Answer:")
        st.write(response)
