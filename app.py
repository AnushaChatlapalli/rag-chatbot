import os
import time
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage

load_dotenv()

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 RAG Chatbot")
st.caption("Upload multiple PDFs or enter URLs and ask me anything!")

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

def process_pdfs(uploaded_files):
    all_chunks = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        all_chunks.extend(chunks)
        os.unlink(tmp_path)
        st.info(f"✅ {uploaded_file.name} — {len(chunks)} chunks")
    return all_chunks

def process_url(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(docs)

def build_vectorstore(chunks):
    batch_size = 50
    vectorstore = None
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        current_batch = i//batch_size + 1
        st.info(f"Embedding batch {current_batch} of {total_batches}...")
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)
        if i + batch_size < len(chunks):
            time.sleep(60)
    return vectorstore

def ask_question(question, vectorstore):
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    prompt = f"""You are a helpful assistant. First try to answer using the context below.
If the answer is not in the context, answer using your own knowledge but mention
it's not from the uploaded documents.

Context:
{context}

Question: {question}

Answer:"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.header("📂 Add your documents")
    tab1, tab2 = st.tabs(["Upload PDFs", "Enter URL"])

    with tab1:
        uploaded_files = st.file_uploader(
            "Choose PDFs (select multiple!)",
            type="pdf",
            accept_multiple_files=True
        )
        if uploaded_files and st.button("Process PDFs"):
            with st.spinner("Processing PDFs..."):
                chunks = process_pdfs(uploaded_files)
                st.session_state.vectorstore = build_vectorstore(chunks)
                st.session_state.messages = []
                st.success(f"Done! {len(chunks)} total chunks from {len(uploaded_files)} files.")

    with tab2:
        url = st.text_input("Enter a webpage URL")
        if url and st.button("Process URL"):
            with st.spinner("Processing URL..."):
                chunks = process_url(url)
                st.session_state.vectorstore = build_vectorstore(chunks)
                st.session_state.messages = []
                st.success(f"Done! {len(chunks)} chunks processed.")

    if st.session_state.vectorstore and st.button("🗑️ Clear all documents"):
        st.session_state.vectorstore = None
        st.session_state.messages = []
        st.rerun()

    if st.session_state.messages:
        st.divider()
        st.subheader("💬 Chat History")
        for msg in st.session_state.messages:
            role = "🧑 You" if msg["role"] == "user" else "🤖 Bot"
            st.markdown(f"**{role}:** {msg['content'][:80]}...")

if st.session_state.vectorstore is None:
    st.info("👈 Upload PDFs or enter a URL in the sidebar to get started!")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = ask_question(prompt, st.session_state.vectorstore)
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
