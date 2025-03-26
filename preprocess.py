import os
import sys
import json
import numpy as np
import faiss
from flask import Flask, request, render_template
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Configure the Gemini API with your API key
genai.configure(api_key="AIzaSyBDQfBKivNhofiw4_rqgQ46wMaf99XB6fM")
model = genai.GenerativeModel('gemini-1.5-pro')

# --------- CONFIGURATION ---------
DATA_DIR = "data"
INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.index")
CHUNKS_FILE = os.path.join(DATA_DIR, "chunks.json")
TOP_K = 5  # Retrieve top 5 chunks initially

# --------- UTILITY FUNCTIONS ---------
def load_index():
    """Loads the FAISS index and chunk metadata from disk."""
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunk_texts = json.load(f)
    return index, chunk_texts

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text(text):
    """Computes an embedding for the given text."""
    return embedding_model.encode(text)

def search_chunks(query, index, chunk_texts, top_k=TOP_K):
    """Uses FAISS to retrieve the top_k chunks relevant to the query."""
    q_embedding = embed_text(query).astype('float32')
    q_embedding = np.expand_dims(q_embedding, axis=0)
    distances, indices = index.search(q_embedding, top_k)
    retrieved = [chunk_texts[i] for i in indices[0] if i < len(chunk_texts)]
    return retrieved

def extract_keywords(query):
    """
    A simple keyword extractor that lowercases the query, splits on whitespace,
    and filters out common stop words. Returns a comma-separated string.
    """
    words = query.lower().split()
    stop_words = {"the", "is", "in", "at", "which", "on", "and", "a", "an", "of", "to", "it", "for", "with", "that"}
    keywords = [word for word in words if word not in stop_words]
    return ", ".join(keywords)

def call_gemini(prompt):
    """
    Calls the Gemini API using the google-generativeai library.
    Uses the model's generate_content method and returns the generated text.
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error calling Gemini API: {e}"

def extra_search(keyword, chunk_texts):
    """
    Scans all chunk texts for the presence of the keyword (case-insensitive)
    and returns a list of those chunks.
    """
    keyword = keyword.lower()
    extra = [chunk for chunk in chunk_texts if keyword in chunk.lower()]
    return extra

# --------- FLASK CHATBOT INTERFACE ---------
app = Flask(__name__)

if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_FILE):
    faiss_index, chunks_metadata = load_index()
else:
    print("Index or chunk file not found. Please run the preprocess script first.")
    sys.exit(1)

@app.route("/", methods=["GET", "POST"])
def chatbot():
    answer = ""
    if request.method == "POST":
        query = request.form["query"]
        # Retrieve initial relevant chunks using FAISS search
        retrieved_chunks = search_chunks(query, faiss_index, chunks_metadata)
        
        # If the query mentions a specific section (e.g., section 198), perform an extra search
        lower_query = query.lower()
        if "section 198" in lower_query:
            extra_chunks = extra_search("section 198", chunks_metadata)
            # Merge extra chunks into the retrieved list, avoiding duplicates
            for chunk in extra_chunks:
                if chunk not in retrieved_chunks:
                    retrieved_chunks.append(chunk)
        
        # Combine all retrieved chunks as context
        context = "\n\n".join(retrieved_chunks)
        # Extract keywords to help guide the model
        keywords = extract_keywords(query)
        # Build the prompt including both retrieved context and keywords
        prompt = (
            "Using the following textbook excerpts and search keywords, "
            "answer the question as accurately as possible.\n\n"
            "Relevant Search Keywords: " + keywords + "\n\n"
            "Textbook Excerpts:\n" + context + "\n\n"
            "Question: " + query + "\nAnswer:"
        )
        answer = call_gemini(prompt)
    return render_template("index.html", answer=answer)

if __name__ == "__main__":
    app.run()
