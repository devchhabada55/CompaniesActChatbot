# # import os
# # import pickle
# # import gradio as gr
# # from langchain.chains import ConversationalRetrievalChain
# # from langchain.memory import ConversationBufferMemory
# # from langchain_google_genai import ChatGoogleGenerativeAI
# # from langchain_community.vectorstores import FAISS
# # from langchain_community.embeddings import HuggingFaceEmbeddings
# # from langchain.docstore.document import Document

# # # Set API key globally
# # os.environ["GOOGLE_API_KEY"] = "AIzaSyC750FpGoI-HlMYdQcMPl7qzNT7nDX_7rY"

# # def load_processed_data():
# #     """Load the processed textbook data."""
# #     print("Loading data...")
    
# #     # Fix the path to reflect the actual file structure
# #     current_dir = os.path.dirname(os.path.abspath(__file__))
# #     parent_dir = os.path.dirname(current_dir)
# #     chunks_path = os.path.join(parent_dir, "textbook_data", "chunks.pkl")
    
# #     print(f"Looking for chunks file at: {chunks_path}")
    
# #     # Load chunks
# #     with open(chunks_path, "rb") as f:
# #         chunks = pickle.load(f)
    
# #     print(f"Loaded {len(chunks)} chunks")
# #     return chunks

# # # Global variables to store the QA chain
# # qa_chain = None

# # def setup_qa_chain():
# #     """Set up the QA chain with the Gemini model."""
# #     global qa_chain
    
# #     # Only set up once
# #     if qa_chain is not None:
# #         return qa_chain
    
# #     # Load data
# #     chunks = load_processed_data()
    
# #     # Create Documents from chunks
# #     documents = [Document(page_content=chunk, metadata={"source": "textbook"}) for chunk in chunks]
    
# #     # Initialize the embedding function
# #     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
# #     # Create a FAISS vector store
# #     vector_store = FAISS.from_documents(documents, embeddings)
    
# #     # Initialize the Gemini model
# #     llm = ChatGoogleGenerativeAI(
# #         model="gemini-1.5-pro",
# #         temperature=0.5,
# #         max_output_tokens=1024,
# #         top_p=0.95
# #     )
    
# #     # Create memory for conversation history
# #     memory = ConversationBufferMemory(
# #         memory_key="chat_history",
# #         return_messages=True
# #     )
    
# #     # Create the conversational retrieval chain
# #     qa_chain = ConversationalRetrievalChain.from_llm(
# #         llm=llm,
# #         retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
# #         memory=memory
# #     )
    
# #     return qa_chain

# # def chat_response(message, history):
# #     """Handle chat interactions with the QA chain."""
# #     # Setup the QA chain if not already done
# #     chain = setup_qa_chain()
    
# #     # Format history for the chain
# #     formatted_history = []
# #     for human_msg, ai_msg in history:
# #         formatted_history.append((human_msg, ai_msg))
    
# #     # Get response from chain
# #     response = chain({"question": message, "chat_history": formatted_history})
    
# #     return response["answer"]

# # # Create a simple Gradio interface
# # def create_interface():
# #     """Create and launch the chat interface."""
# #     with gr.Blocks(title="Textbook Chatbot") as demo:
# #         gr.Markdown("# Textbook Chatbot with Gemini 1.5 Pro")
# #         gr.Markdown("Ask questions about your textbook and get answers based on its content.")
        
# #         chatbot = gr.Chatbot()
# #         msg = gr.Textbox(label="Your question")
# #         clear = gr.Button("Clear")
        
# #         # Example questions
# #         examples = [
# #             "What is the main topic of this textbook?",
# #             "Can you summarize a key concept from the textbook?",
# #             "Explain the relationship between the topics in chapter 1 and chapter 2."
# #         ]
        
# #         # Set up the chat functionality
# #         msg.submit(chat_response, [msg, chatbot], [chatbot])
# #         clear.click(lambda: None, None, chatbot, queue=False)
        
# #         # Add example questions
# #         gr.Examples(examples=examples, inputs=msg)
        
# #     return demo

# # if __name__ == "__main__":
# #     try:
# #         print("Setting up the chatbot interface...")
# #         demo = create_interface()
# #         demo.launch(share=True)
# #     except Exception as e:
# #         print(f"Error: {str(e)}")
# #         import traceback
# #         traceback.print_exc()
# import os
# import pickle
# import streamlit as st
# from langchain.chains import ConversationalRetrievalChain
# from langchain.memory import ConversationBufferMemory
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.docstore.document import Document

# def load_processed_data():
#     st.write("Loading data...")
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     parent_dir = os.path.dirname(current_dir)
#     chunks_path = os.path.join(parent_dir, "textbook_data", "chunks.pkl")
#     st.write(f"Looking for chunks file at: {chunks_path}")
#     with open(chunks_path, "rb") as f:
#         chunks = pickle.load(f)
#     st.write(f"Loaded {len(chunks)} chunks")
#     return chunks

# def setup_gemini_api():
#     os.environ["GOOGLE_API_KEY"] = "AIzaSyC750FpGoI-HlMYdQcMPl7qzNT7nDX_7rY"
#     llm = ChatGoogleGenerativeAI(
#         model="gemini-1.5-pro",
#         temperature=0.5,
#         max_output_tokens=1024,
#         top_p=0.95
#     )
#     return llm

# def create_chat_chain(chunks):
#     st.write("Setting up the chatbot...")
#     # Create Documents from chunks
#     documents = [Document(page_content=chunk, metadata={"source": "textbook"}) for chunk in chunks]
#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#     vector_store = FAISS.from_documents(documents, embeddings)
    
#     # Increase the number of retrieved chunks if needed (e.g., k=10)
#     retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    
#     llm = setup_gemini_api()
    
#     memory = ConversationBufferMemory(
#         memory_key="chat_history",
#         return_messages=True
#     )
    
#     qa_chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=retriever,
#         memory=memory,
#         # Optionally, you can inject a system prompt to force textbook reference:
#         verbose=True  # This will print out additional debugging info to your console
#     )
    
#     return qa_chain

# def respond(message, chat_history, qa_chain):
#     # Optionally: prepend a system instruction to ensure textbook referencing
#     prompt_prefix = "Answer the following question strictly based on the textbook content provided: "
#     full_question = prompt_prefix + message
    
#     # Format the chat history
#     formatted_history = [(h, a) for h, a in chat_history]
    
#     result = qa_chain({"question": full_question, "chat_history": formatted_history})
    
#     # For debugging: log retrieved sources if available
#     if "source_documents" in result:
#         st.write("**Retrieved Chunks:**")
#         for doc in result["source_documents"]:
#             st.write(doc.metadata.get("source", "unknown"), "->", doc.page_content[:200], "...")
    
#     return result["answer"]

# def main():
#     st.title("Textbook Chatbot with Gemini 1.5 Pro")
#     st.write("Ask questions about your textbook and get answers based on its content.")

#     if 'chat_history' not in st.session_state:
#         st.session_state.chat_history = []
    
#     if 'qa_chain' not in st.session_state:
#         try:
#             chunks = load_processed_data()
#             # If you find that key definitions are missing, consider revisiting your chunking parameters in your textbookprocessing.py script.
#             qa_chain = create_chat_chain(chunks)
#             st.session_state.qa_chain = qa_chain
#         except Exception as e:
#             st.error(f"Error during setup: {e}")
#             return

#     user_input = st.text_input("Your message:", key="input")
    
#     if st.button("Send") and user_input:
#         try:
#             answer = respond(user_input, st.session_state.chat_history, st.session_state.qa_chain)
#             st.session_state.chat_history.append((user_input, answer))
#         except Exception as e:
#             st.error(f"Error during conversation: {e}")
    
#     st.markdown("### Conversation History")
#     for human, bot in st.session_state.chat_history:
#         st.markdown(f"**You:** {human}")
#         st.markdown(f"**Bot:** {bot}")

# if __name__ == "__main__":
#     main()
