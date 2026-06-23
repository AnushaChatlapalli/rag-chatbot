import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

def load_documents(pdf_paths=[], urls=[], txt_paths=[]):
    docs = []
    for path in pdf_paths:
        print(f"Loading PDF: {path}")
        docs.extend(PyPDFLoader(path).load())
    for url in urls:
        print(f"Loading URL: {url}")
        docs.extend(WebBaseLoader(url).load())
    for path in txt_paths:
        print(f"Loading text file: {path}")
        docs.extend(TextLoader(path).load())
    return docs

def build_vectorstore(docs):
    print("Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    print(f"Total chunks created: {len(chunks)}")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    print("Creating embeddings in batches...")
    batch_size = 50
    vectorstore = None

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1} of {len(chunks)//batch_size + 1}...")
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)
        if i + batch_size < len(chunks):
            print("Waiting 60 seconds for rate limit...")
            time.sleep(60)

    vectorstore.save_local("vectorstore")
    print("Vectorstore saved!")

if __name__ == "__main__":
    docs = load_documents(
        pdf_paths=[],
        urls=["https://en.wikipedia.org/wiki/Python_(programming_language)"],
        txt_paths=[]
    )
    build_vectorstore(docs)