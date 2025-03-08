import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
import streamlit as st

# Load small open-source models
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # ~80MB
GENERATION_MODEL = "google/flan-t5-small"  # ~300MB

# Load models
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL)
generation_model = AutoModelForSeq2SeqLM.from_pretrained(GENERATION_MODEL)

# Load financial data
with open("financial_data.txt", "r") as file:
    financial_data = file.read()

# Preprocess data into chunks
chunks = [chunk.strip() for chunk in financial_data.split("\n\n") if chunk.strip()]

# Embed chunks
chunk_embeddings = embedding_model.encode(chunks)
dimension = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(chunk_embeddings)

# BM25 for hybrid search
tokenized_corpus = [chunk.split() for chunk in chunks]
bm25 = BM25Okapi(tokenized_corpus)

def hybrid_search(query, top_k=3):
    # Dense retrieval
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    dense_results = [chunks[i] for i in indices[0]]

    # Sparse retrieval (BM25)
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_indices = np.argsort(bm25_scores)[-top_k:][::-1]
    sparse_results = [chunks[i] for i in bm25_indices]

    # Combine and deduplicate results
    combined_results = list(set(dense_results + sparse_results))
    return combined_results[:top_k]

def generate_response(query, context):
    input_text = f"Question: {query}\nContext: {context}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = generation_model.generate(**inputs, max_length=150)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Input-side guardrail
def validate_query(query):
    financial_keywords = ["revenue", "income", "eps", "dividend", "profit", "financial"]
    return any(keyword in query.lower() for keyword in financial_keywords)

# Streamlit UI
st.title("Financial RAG Chatbot")
st.write("Ask financial questions about the company's earnings.")

user_query = st.text_input("Enter your question:")

if user_query:
    if not validate_query(user_query):
        st.error("Please ask a financial-related question.")
    else:
        retrieved_chunks = hybrid_search(user_query)
        context = "\n".join(retrieved_chunks)
        response = generate_response(user_query, context)
        st.write("**Answer:**", response)
        st.write("**Confidence Score:**", "High" if len(retrieved_chunks) > 0 else "Low")