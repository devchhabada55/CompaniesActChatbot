import os
import pickle
import streamlit as st
from google import genai  # Import the Google GenAI client
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

def load_processed_data():
    st.write("Loading data...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    chunks_path = os.path.join(parent_dir, "textbook_data", "chunks.pkl")
    st.write(f"Looking for chunks file at: {chunks_path}")
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    st.write(f"Loaded {len(chunks)} chunks")
    return chunks

def setup_google_genai():
    # Set your Google API key (replace with your actual key)
    os.environ["GOOGLE_API_KEY"] = "AIzaSyC750FpGoI-HlMYdQcMPl7qzNT7nDX_7rY"
    
    # Initialize the Google GenAI client
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    
    # Retrieve the desired model (e.g., gemini-1.5-pro)
    model = genai.Client("gemini-1.5-pro")
    return model

def create_chat_chain(chunks):
    st.write("Setting up the chatbot...")
    # Create Document objects from the chunks
    documents = [Document(page_content=chunk, metadata={"source": "textbook"}) for chunk in chunks]
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(documents, embeddings)
    
    # Increase the number of retrieved chunks if needed for better recall
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    
    # Use the direct Google GenAI model instead of langchain_google_genai wrapper
    llm = setup_google_genai()
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=True  # Enables verbose logging for debugging purposes
    )
    return qa_chain

def respond(message, chat_history, qa_chain):
    # Prefix the query to ensure the answer is based solely on textbook content
    prompt_prefix = "Answer the following question strictly based on the textbook content provided: "
    full_question = prompt_prefix + message
    formatted_history = [(h, a) for h, a in chat_history]
    
    result = qa_chain({"question": full_question, "chat_history": formatted_history})
    
    # Optionally display the retrieved source documents for debugging
    if "source_documents" in result:
        st.write("**Retrieved Chunks:**")
        for doc in result["source_documents"]:
            st.write(doc.metadata.get("source", "unknown"), "->", doc.page_content[:200], "...")
    
    return result["answer"]

def main():
    st.title("Textbook Chatbot with Google GenAI")
    st.write("Ask questions about your textbook and get answers based on its content.")
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'qa_chain' not in st.session_state:
        try:
            chunks = load_processed_data()
            qa_chain = create_chat_chain(chunks)
            st.session_state.qa_chain = qa_chain
        except Exception as e:
            st.error(f"Error during setup: {e}")
            return
    
    user_input = st.text_input("Your message:", key="input")
    
    if st.button("Send") and user_input:
        try:
            answer = respond(user_input, st.session_state.chat_history, st.session_state.qa_chain)
            st.session_state.chat_history.append((user_input, answer))
        except Exception as e:
            st.error(f"Error during conversation: {e}")
    
    st.markdown("### Conversation History")
    for human, bot in st.session_state.chat_history:
        st.markdown(f"**You:** {human}")
        st.markdown(f"**Bot:** {bot}")

if __name__ == "__main__":
    main()
