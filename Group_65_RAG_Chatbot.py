import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Load embedding model (small open-source model)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Always use default financial data from financial_data.txt

FINANCIAL_DATA_FILE = 'financial_data.txt'

# Function to read financial data from file
def get_financial_data():
    with open(FINANCIAL_DATA_FILE, 'r') as file:
        return file.read().splitlines()

# Preprocess data into optimized chunks
financial_data = get_financial_data()
chunk_size = 10
chunks = []
for i in range(0, len(financial_data), chunk_size):
    chunk = ' '.join(financial_data[i:i+chunk_size])
    chunks.append(chunk)

# Embed chunks for dense retrieval
embeddings = embedding_model.encode(chunks)

# Ensure embeddings have correct dimensions (2D)
if len(embeddings.shape) == 3:
    embeddings = embeddings.squeeze(axis=1)

# Sparse retrieval using TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(chunks)

# Input-side guardrail to filter irrelevant questions
def is_financial_question(query):
    pattern = r'\b(revenue|income|profit|loss|assets|liabilities|balance sheet|cash flow|earnings|equity|debt)\b'
    return bool(re.search(pattern, query, re.IGNORECASE))

# Extract the most relevant line ONLY (precise answer)
def extract_relevant_line(chunk, query):
    pattern = re.compile(r'(Revenue|Net Income|Total Assets|Total Liabilities):\s\$([\d,]+)\s(million|billion)', re.IGNORECASE)
    matches = pattern.findall(chunk)

    extracted_info = []
    for match in matches:
        extracted_info.append(f"{match[0]}: ${match[1]} {match[2]}")

    if len(extracted_info) > 0:
        return extracted_info[0]
    return "No relevant data found."

# Handle annual questions like "What was revenue for 2023?"
def handle_annual_query(query):
    year_match = re.search(r'\b(\d{4})\b', query)
    if not year_match:
        return None

    year = year_match.group(1)
    year_data = []
    capture = False

    for line in financial_data:
        if f"Fiscal Year: {year}" in line:
            capture = True
        elif capture and line.strip() == '':
            break
        if capture:
            year_data.append(line)

    if not year_data:
        return None

    revenue = 0
    for line in year_data:
        match = re.search(r'Revenue:\s\$([\d,]+)\s(million|billion)', line)
        if match:
            value = int(match.group(1).replace(',', ''))
            revenue += value

    if revenue == 0:
        return None

    return f"Revenue for {year}: ${revenue:,} million"

# Output-side guardrail to remove hallucinations
def filter_hallucinations(answer, confidence):
    threshold = 0.3
    if confidence < threshold:
        return "💬 I'm not confident in the answer. Please verify the financial data or rephrase your question."
    return answer

# Optimized hybrid search function
def hybrid_search(query):
    # Handle annual questions first
    annual_answer = handle_annual_query(query)
    if annual_answer:
        return annual_answer, 1.0

    # Sparse retrieval
    query_vec = vectorizer.transform([query])
    sparse_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Dense retrieval
    query_embed = embedding_model.encode([query])
    dense_scores = cosine_similarity(query_embed.reshape(1, -1), embeddings).flatten()

    # Weighted combination of scores (60% Dense + 40% Sparse)
    combined_scores = (0.6 * dense_scores + 0.4 * sparse_scores)
    top_index = combined_scores.argmax()
    relevant_line = extract_relevant_line(chunks[top_index], query)
    return relevant_line, combined_scores[top_index]

# Streamlit UI
st.title("💸 RAG Chatbot for Financial Data")
st.markdown("This chatbot can answer financial questions based on company financial statements.")

st.markdown("### 📊 Ask a Financial Question")
user_query = st.text_input("💬 Enter your financial question below:")

if user_query:
    if not is_financial_question(user_query):
        st.error("❌ This question does not appear to be related to financial data. Please ask a financial question.")
    else:
        answer, confidence = hybrid_search(user_query)
        filtered_answer = filter_hallucinations(answer, confidence)
        st.success("✅ Answer Retrieved")
        confidence_label = "High Confidence" if confidence >= 0.5 else "Low Confidence"
        st.write("### 📜 Answer:")
        st.markdown(
            f"""
            <div style='background-color:#212529;color:#f8f9fa;padding:15px;border-radius:5px;font-family:monospace;'>
                {filtered_answer}
            </div>
            """,
            unsafe_allow_html=True
        )
        st.write(f"**🔢 Confidence Score:** `{confidence:.2f}` ({confidence_label})")
