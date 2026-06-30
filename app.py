import streamlit as st
import time

# TODO: Import your actual backend function here
from backend import process_query, process_new_pdf

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="PDF RAG Assistant", page_icon="📄", layout="centered")

st.title("📄 PDF RAG Assistant")
st.markdown("Powered by **Gemini 1.5 Flash**, **Pinecone**, and **LangChain**.")

# --- SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- MOCK BACKEND FUNCTION (Replace with your actual import) ---
def mock_get_rag_response(user_input):
    """Replace this with your actual LangChain/Gemini backend call."""
    time.sleep(1.5) # Simulating retrieval time
    return f"This is a simulated RAG response to: '{user_input}'. The backend is ready to be plugged in!"

# --- CHAT INTERFACE ---
# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about your documents..."):
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat widget
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat widget
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating response..."):
            # Call your backend RAG pipeline here
          # 4. Call your backend RAG pipeline here
            response = process_query(prompt)
    
    # Force clean the response
    if isinstance(response, dict):
        clean_text = response.get('text', str(response))
    else:
        clean_text = str(response)
    
    # Display ONLY the clean text
    st.markdown(clean_text)
    if isinstance(response, dict):
        st.markdown(response.get('text', response))
    else:
        st.markdown(response)
            
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# --- SIDEBAR ---
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Drop a new PDF here", type="pdf")
    
    if uploaded_file is not None:
        with st.spinner("Processing your PDF..."):
            try:
                # Save the uploaded file temporarily
                temp_file_path = "temp_uploaded.pdf"
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Process the new file through your backend
                process_new_pdf(temp_file_path)
                st.success("PDF processed and ready for chat!")
            
            except Exception as e:
                st.error(f"Could not process PDF: {e}")

    st.divider() # Adds a nice line to separate sections
    
    st.header("About")
    st.write("This application uses Retrieval-Augmented Generation (RAG) to answer questions based on uploaded documents.")
    st.write("**Tech Stack:**")
    st.write("- Frontend: Streamlit")
    st.write("- LLM: Gemini 1.5 Flash")
    st.write("- Vector DB: Pinecone")
    st.write("- Embeddings: HuggingFace")