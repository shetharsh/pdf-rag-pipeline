import os
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# ... keep your existing imports below ...

# Load environment variables from your .env file
load_dotenv()

# --- 1. INITIALIZE EMBEDDINGS ---
# This needs to match the model you used to upload the data
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# --- 2. CONNECT TO PINECONE ---
# We use os.getenv to pull from the .env file instead of hardcoding keys
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "langchainvector"
index = pc.Index(index_name)

# Connect LangChain to your existing Pinecone index
vectorstore = PineconeVectorStore(
    index=index, 
    embedding=embeddings, 
    text_key="text" # Note: 'text' is the default key LangChain uses for page_content
)

# --- 3. INITIALIZE GEMINI LLM ---
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    temperature=0.3,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# --- 4. THE MAIN FUNCTION ---
def process_query(user_query):
    """
    Takes the user's question, finds relevant docs in Pinecone, 
    and sends them to Gemini to get an answer.
    """
    # 1. Retrieve matching documents from Pinecone
    matching_docs = vectorstore.similarity_search(user_query, k=3)
    
    # 2. Combine the document text into one context string
    context = "\n\n".join([doc.page_content for doc in matching_docs])
    
    # 3. Create the prompt for Gemini
    prompt = f"""
    You are a helpful assistant. Use the following context from the document to answer the question.
    If you don't know the answer based on the context, say you don't know.
    
    Context:
    {context}
    
    Question:
    {user_query}
    """
    
    
    # 4. Get the answer from Gemini
    response = llm.invoke(prompt)
    
    # Check if the response object has a 'content' attribute
    if hasattr(response, 'content'):
        return response.content
    else:
        return str(response)


def process_new_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    # This uses the 'vectorstore' variable already defined at the top of this file
    vectorstore.add_documents(chunks)
    return True