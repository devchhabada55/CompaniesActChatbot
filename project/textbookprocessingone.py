# import os
# from pypdf import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np
# import pickle
# import time

# def extract_text_from_pdf(pdf_path):
#     """Extract text from a PDF file."""
#     print(f"Extracting text from {pdf_path}...")
#     reader = PdfReader(pdf_path)
#     text = ""
#     for i, page in enumerate(reader.pages):
#         if i % 10 == 0:
#             print(f"  Processing page {i+1}/{len(reader.pages)}")
#         text += page.extract_text() + "\n"
#     print(f"Extracted {len(text)} characters from {len(reader.pages)} pages")
#     return text

# def chunk_text(text, chunk_size=1000, chunk_overlap=200):
#     """Split text into overlapping chunks."""
#     print(f"Chunking text with size={chunk_size}, overlap={chunk_overlap}...")
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#         length_function=len,
#         separators=["\n\n", "\n", ". ", " ", ""]
#     )
#     chunks = text_splitter.split_text(text)
#     print(f"Created {len(chunks)} chunks")
#     return chunks

# def create_embeddings(chunks):
#     """Create vector embeddings for text chunks."""
#     print("Creating embeddings...")
#     start_time = time.time()
    
#     # Load the embedding model
#     model = SentenceTransformer('all-MiniLM-L6-v2')
    
#     # Create embeddings with a progress indicator
#     total_chunks = len(chunks)
#     embeddings = []
    
#     for i, chunk in enumerate(chunks):
#         if i % 100 == 0:
#             print(f"  Processing chunk {i+1}/{total_chunks}")
#         embedding = model.encode([chunk])[0]
#         embeddings.append(embedding)
    
#     embeddings = np.array(embeddings)
    
#     # Normalize embeddings for cosine similarity
#     faiss.normalize_L2(embeddings)
    
#     print(f"Created embeddings in {time.time() - start_time:.2f} seconds")
#     return embeddings, model

# def main():
#     # Create output directory
#     os.makedirs("textbook_data", exist_ok=True)
    
#     # Step 1: Extract text from PDF
#     pdf_path = r'C:\Users\HP\OneDrive\Desktop\Parag Chatbot\project\Companies Act 2013 (2).pdf'  # Make sure this file exists in your directory
#     raw_text = extract_text_from_pdf(pdf_path)
    
#     # Save extracted text (optional but helpful for debugging)
#     with open("textbook_data/extracted_text.txt", "w", encoding="utf-8") as f:
#         f.write(raw_text)
    
#     # Step 2: Create chunks from text
#     chunks = chunk_text(raw_text)
    
#     # Save chunks
#     with open("textbook_data/chunks.pkl", "wb") as f:
#         pickle.dump(chunks, f)
    
#     # Step 3: Create embeddings
#     embeddings, model = create_embeddings(chunks)
    
#     # Step 4: Create FAISS index
#     print("Creating FAISS index...")
#     dimension = embeddings.shape[1]
#     index = faiss.IndexFlatIP(dimension)
#     index.add(embeddings)
    
#     # Save the model, index, and chunks
#     print("Saving data...")
#     with open("textbook_data/embedding_model.pkl", "wb") as f:
#         pickle.dump(model, f)
    
#     faiss.write_index(index, "textbook_data/embeddings.faiss")
    
#     print("Processing complete! You can now run the chatbot.py script.")

# if __name__ == "__main__":
#     main()