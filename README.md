# 📄 End-to-End PDF RAG Pipeline

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B)
![LangChain](https://img.shields.io/badge/LangChain-Framework-1C3C3C)
![Pinecone](https://img.shields.io/badge/Pinecone-VectorDB-000000)
![Gemini](https://img.shields.io/badge/Gemini_1.5_Flash-LLM-8A2BE2)

A full-stack Retrieval-Augmented Generation (RAG) application that allows users to chat with PDF documents. The system extracts text, generates dense vector embeddings, stores them in a cloud vector database, and uses a state-of-the-art LLM to generate highly accurate, context-aware responses.

## 🚀 Features
* **Document Processing:** Ingests and chunks PDF documents effectively using LangChain.
* **Semantic Search:** Utilizes HuggingFace embeddings and Pinecone for fast, relevant context retrieval.
* **Generative AI:** Powered by Google's Gemini 1.5 Flash for rapid, intelligent answer generation.
* **Modern UI:** Clean, conversational interface built with Streamlit.

## 🛠️ Tech Stack
* **Frontend:** Streamlit
* **Orchestration:** LangChain
* **Embeddings:** HuggingFace
* **Vector Database:** Pinecone
* **Language Model:** Google Gemini 1.5 Flash

## ⚙️ Local Setup & Installation

**1. Clone the repository**
```bash
git clone [https://github.com/yourusername/pdf-rag-pipeline.git](https://github.com/yourusername/pdf-rag-pipeline.git)
cd pdf-rag-pipeline