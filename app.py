import streamlit as st
import time


from backend import process_query, process_new_pdf


st.set_page_config(page_title="PDF RAG Assistant", page_icon="📄", layout="centered")

st.title("📄 PDF RAG Assistant")
st.markdown("Powered by **Gemini 1.5 Flash**, **Pinecone**, and **LangChain**.")


if "messages" not in st.session_state:
    st.session_state.messages = []


def mock_get_rag_response(user_input):
    """Replace this with your actual LangChain/Gemini backend call."""
    time.sleep(1.5) 
    return f"This is a simulated RAG response to: '{user_input}'. The backend is ready to be plugged in!"



for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Ask a question about your documents..."):
    
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    
    with st.chat_message("user"):
        st.markdown(prompt)

    
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating response..."):
            
          
            response = process_query(prompt)
    
   
    if isinstance(response, list) and len(response) > 0:
        clean_text = response[0].get('text', str(response))
    elif isinstance(response, dict):
        clean_text = response.get('text', str(response))
    elif hasattr(response, 'content'):  
        clean_text = response.content
    else:
        clean_text = str(response)
    
    
    st.markdown(clean_text)
            
    
    st.session_state.messages.append({"role": "assistant", "content": clean_text})


with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Drop a new PDF here", type="pdf")
    
    if uploaded_file is not None:
        with st.spinner("Processing your PDF..."):
            try:
                
                temp_file_path = "temp_uploaded.pdf"
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                
                process_new_pdf(temp_file_path)
                st.success("PDF processed and ready for chat!")
            
            except Exception as e:
                st.error(f"Could not process PDF: {e}")

    st.divider() 
    
    st.header("About")
    st.write("This application uses Retrieval-Augmented Generation (RAG) to answer questions based on uploaded documents.")
    st.write("**Tech Stack:**")
    st.write("- Frontend: Streamlit")
    st.write("- LLM: Gemini 1.5 Flash")
    st.write("- Vector DB: Pinecone")
    st.write("- Embeddings: HuggingFace")