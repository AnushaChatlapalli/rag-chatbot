import os
import time
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage

load_dotenv()

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 RAG Chatbot")
st.caption("Upload a PDF or ask questions about Python!")

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

@st.cache_resource
def load_default_vectorstore():
    return FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)

def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    batch_size = 50
    vectorstore = None
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)
        if i + batch_size < len(chunks):
            time.sleep(60)

    os.unlink(tmp_path)
    return vectorstore

def ask_question(question, vectorstore):
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    prompt = f"""Use the following context to answer the question.
If the answer is not in the context, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}

Answer:"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = load_default_vectorstore()

with st.sidebar:
    st.header("📄 Upload your PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file:
        if st.button("Process PDF"):
            with st.spinner("Processing PDF... this may take a few minutes due to rate limits"):
                st.session_state.vectorstore = process_pdf(uploaded_file)
                st.session_state.messages = []
                st.success("PDF processed! Ask me anything about it.")

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
