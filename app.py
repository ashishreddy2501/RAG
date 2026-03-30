import streamlit as st
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# ---------- PAGE ----------
st.set_page_config(page_title="RAG Chatbot", layout="wide")
#st.title("📄 AI Document Chatbot (Latest LangChain)")
st.title("Your Reading Assistant")
st.text("Created by S Ashish Reddy")

# ---------- API KEY ----------
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = st.text_input("Enter OpenAI API Key", type="password")

# ---------- CACHE ----------
@st.cache_resource
def load_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)

@st.cache_resource
def load_embeddings():
    return OpenAIEmbeddings()

# ---------- FILE UPLOAD ----------
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    # Load PDF
    docs = PyPDFLoader(pdf_path).load()

    # Split
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    ).split_documents(docs)

    # Vector DB
    db = FAISS.from_documents(chunks, load_embeddings())

    retriever = db.as_retriever(search_kwargs={"k": 4})

    # ---------- PROMPT ----------
    prompt = ChatPromptTemplate.from_template("""
    You are a helpful AI assistant.

    Answer ONLY from the context below.
    If the answer is not present, say:
    "I don't know based on the document."

    Context:
    {context}

    Question:
    {input}

    Provide a clear and detailed answer.
    """)

    # ---------- CHAINS ----------
    llm = load_llm()

    document_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)

    st.success("✅ Document ready!")

    # ---------- CHAT ----------
    if "messages" not in st.session_state:
        st.session_state.messages = []

    query = st.chat_input("Ask a question...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})

        with st.spinner("Thinking..."):
            response = rag_chain.invoke({"input": query})
            answer = response["answer"]

        st.session_state.messages.append({"role": "assistant", "content": answer})

    # Display messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
