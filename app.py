import os
import shutil
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

CHROMA_DIR = "chroma_db"

st.set_page_config(page_title="Chat with your PDF", page_icon="📚", layout="wide")

# ---------------------------------------------------------------------------
# Cached resources (loaded once per session, not on every rerun)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatMistralAI(model="mistral-small-2506")


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
""",
        ),
        (
            "human",
            """
Context:
{context}

Question:
{question}
""",
        ),
    ]
)

# ---------------------------------------------------------------------------
# Ingestion: build the Chroma DB from an uploaded PDF (create_database.py logic)
# ---------------------------------------------------------------------------


def build_vectorstore_from_pdf(uploaded_file, chunk_size=1000, chunk_overlap=200):
    # Wipe any previous DB so each new upload starts fresh
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    # PyPDFLoader needs a real file path, so write the upload to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = splitter.split_documents(docs)

        embedding_model = get_embedding_model()

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=CHROMA_DIR,
        )
        return vectorstore, len(docs), len(chunks)
    finally:
        os.remove(tmp_path)


def load_existing_vectorstore():
    embedding_model = get_embedding_model()
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding_model)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = os.path.exists(CHROMA_DIR)
if "processed_filename" not in st.session_state:
    st.session_state.processed_filename = None

# ---------------------------------------------------------------------------
# Sidebar: upload + process
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("📄 Upload your book")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    with st.expander("Chunking settings"):
        chunk_size = st.number_input("Chunk size", value=1000, min_value=100, step=100)
        chunk_overlap = st.number_input("Chunk overlap", value=200, min_value=0, step=50)

    process_clicked = st.button(
        "Process document", type="primary", disabled=uploaded_file is None
    )

    if process_clicked and uploaded_file is not None:
        with st.spinner("Reading PDF, splitting into chunks, and building the vector store..."):
            _, n_docs, n_chunks = build_vectorstore_from_pdf(
                uploaded_file, chunk_size, chunk_overlap
            )
        st.session_state.vectorstore_ready = True
        st.session_state.processed_filename = uploaded_file.name
        st.session_state.messages = []  # reset chat for the new document
        st.success(f"Indexed {n_docs} page(s) into {n_chunks} chunks.")

    st.divider()
    if st.session_state.vectorstore_ready:
        label = st.session_state.processed_filename or "an existing document"
        st.info(f"✅ Ready to chat about: **{label}**")
    else:
        st.warning("Upload and process a PDF to get started.")

    if st.session_state.vectorstore_ready and st.button("Clear document & chat"):
        if os.path.exists(CHROMA_DIR):
            shutil.rmtree(CHROMA_DIR)
        st.session_state.vectorstore_ready = False
        st.session_state.processed_filename = None
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# Main: chat interface
# ---------------------------------------------------------------------------

st.title("📚 Chat with your PDF")
st.caption("Upload a book/PDF in the sidebar, process it, then ask questions about its content.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if not st.session_state.vectorstore_ready:
    st.info("👈 Upload a PDF and click **Process document** to start chatting.")
else:
    query = st.chat_input("Ask a question about your document...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                vectorstore = load_existing_vectorstore()
                retriever = vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
                )

                docs = retriever.invoke(query)
                context = "\n\n".join(doc.page_content for doc in docs)

                final_prompt = PROMPT.invoke({"context": context, "question": query})
                llm = get_llm()
                response = llm.invoke(final_prompt)

                st.markdown(response.content)

                with st.expander("Sources used"):
                    for i, doc in enumerate(docs, start=1):
                        page = doc.metadata.get("page", "unknown")
                        st.markdown(f"**Chunk {i} (page {page})**")
                        st.text(doc.page_content[:400] + "...")

        st.session_state.messages.append(
            {"role": "assistant", "content": response.content}
        )