import os
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter



load_dotenv()



embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)



pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "langchainvector"
index = pc.Index(index_name)


vectorstore = PineconeVectorStore(
    index=index, 
    embedding=embeddings, 
    text_key="text" 
)


llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    temperature=0.3,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)


def process_query(user_query):
    """
    Takes the user's question, finds relevant docs in Pinecone, 
    and sends them to Gemini to get an answer.
    """
    
    matching_docs = vectorstore.similarity_search(user_query, k=3)
    
    
    context = "\n\n".join([doc.page_content for doc in matching_docs])
    
    
    prompt = f"""
    You are a helpful assistant. Use the following context from the document to answer the question.
    If you don't know the answer based on the context, say you don't know.
    
    Context:
    {context}
    
    Question:
    {user_query}
    """
    
    
    
    response = llm.invoke(prompt)
    
    
    if hasattr(response, 'content'):
        return response.content
    else:
        return str(response)


def process_new_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    
    vectorstore.add_documents(chunks)
    return True