import streamlit as st
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📄 AI Document Chatbot (RAG)")

# ---------- API KEY ----------
if "OPENAI_API_KEY" not in os.environ:
    st.warning("Enter your OpenAI API Key")
    os.environ["OPENAI_API_KEY"] = st.text_input("API Key", type="password")

# ---------- CACHE ----------
@st.cache_resource
def load_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

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

    # Load + split
    docs = PyPDFLoader(pdf_path).load()

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    ).split_documents(docs)

    # Vector DB
    db = FAISS.from_documents(chunks, load_embeddings())

    # Prompt (important for accuracy)
    prompt_template = """
    You are an AI assistant.

    Answer ONLY from the context below.
    If the answer is not in the context, say:
    "I don't know based on the document."

    Context:
    {context}

    Question:
    {question}

    Give a clear and detailed answer.
    """

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # RAG chain
    qa = RetrievalQA.from_chain_type(
        llm=load_llm(),
        retriever=db.as_retriever(search_kwargs={"k": 4}),
        chain_type_kwargs={"prompt": PROMPT}
    )

    st.success("✅ Document ready!")

    # Chat UI
    if "messages" not in st.session_state:
        st.session_state.messages = []

    query = st.chat_input("Ask a question...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})

        with st.spinner("Thinking..."):
            response = qa.invoke({"query": query})
            answer = response["result"]

        st.session_state.messages.append({"role": "assistant", "content": answer})

    # Display chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
