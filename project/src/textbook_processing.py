import os
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import time

def extract_text_from_pdf(pdf_path):
    print(f"Extracting text from {pdf_path}...")
    reader = PdfReader(pdf_path)
    text = ""
    for i, page in enumerate(reader.pages):
        if i % 10 == 0:
            print(f"  Processing page {i+1}/{len(reader.pages)}")
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    print(f"Extracted {len(text)} characters from {len(reader.pages)} pages")
    return text

def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    print(f"Chunking text with size={chunk_size}, overlap={chunk_overlap}...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    print(f"Created {len(chunks)} chunks")
    return chunks

def create_embeddings(chunks):
    print("Creating embeddings...")
    start_time = time.time()
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = []
    total_chunks = len(chunks)
    
    for i, chunk in enumerate(chunks):
        if i % 100 == 0:
            print(f"  Processing chunk {i+1}/{total_chunks}")
        embedding = model.encode([chunk])[0]
        embeddings.append(embedding)
    
    embeddings = np.array(embeddings)
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    print(f"Created embeddings in {time.time() - start_time:.2f} seconds")
    return embeddings, model

def main():
    os.makedirs("textbook_data", exist_ok=True)
    
    pdf_path = os.path.join("..", "textbook_data", "Companies Act 2013 (2).pdf")
    raw_text = extract_text_from_pdf(pdf_path)
    
    with open("textbook_data/extracted_text.txt", "w", encoding="utf-8") as f:
        f.write(raw_text)
    
    chunks = chunk_text(raw_text)
    with open("textbook_data/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    
    embeddings, model = create_embeddings(chunks)
    
    print("Creating FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    print("Saving data...")
    with open("textbook_data/embedding_model.pkl", "wb") as f:
        pickle.dump(model, f)
    faiss.write_index(index, "textbook_data/embeddings.faiss")
    
    print("Processing complete! You can now run the chatbot.py script.")

if __name__ == "__main__":
    main()
