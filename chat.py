import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage

load_dotenv()

def load_vectorstore():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
    return vectorstore

def ask_question(question, vectorstore):
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)

    prompt = f"""Use the following context to answer the question.
If the answer is not in the context, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}

Answer:"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

if __name__ == "__main__":
    print("Loading knowledge base...")
    vectorstore = load_vectorstore()
    print("Ready! Type your question (or 'quit' to exit)\n")

    while True:
        question = input("You: ")
        if question.lower() == "quit":
            break
        answer = ask_question(question, vectorstore)
        print(f"\nBot: {answer}\n")