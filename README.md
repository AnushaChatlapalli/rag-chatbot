# 🤖 RAG Chatbot

An AI-powered chatbot that answers questions based on your own documents using Retrieval-Augmented Generation (RAG).

## 🚀 Features
- Load documents from PDFs, URLs, and text files
- Smart document chunking and embedding
- Fast similarity search using FAISS
- Powered by Google Gemini AI
- Clean ChatGPT-style web interface built with Streamlit

## 🛠️ Tech Stack
- Python
- LangChain
- Google Gemini API
- FAISS (Vector Store)
- Streamlit

## ⚙️ How to Run

1. Clone the repo
2. Install dependencies:
   pip install -r requirements.txt
3. Add your API keys to a `.env` file:
   GOOGLE_API_KEY=your-key-here
4. Ingest documents:
   python ingest.py
5. Run the chatbot:
   streamlit run app.py

## 📸 How it Works
1. Documents are loaded and split into chunks
2. Chunks are converted to embeddings using Gemini
3. Embeddings are stored in a FAISS vector store
4. User questions are matched to relevant chunks
5. Gemini generates answers based on the context
