"""
Chatbot with RAG — Unified deployment app
Supports: PDF document QA (RAG) + Tavily web search QA
"""

import os
import warnings
import logging

import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Phase 3 — PDF RAG imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.indexes import VectorstoreIndexCreator
from langchain_classic.chains import RetrievalQA

# Tavily web search imports
from tavily import TavilyClient

# ──────────────────────────────────────────────
# Environment & Logging
# ──────────────────────────────────────────────
load_dotenv()
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "deepseek/deepseek-v4-flash")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def get_llm():
    """Return a ChatOpenAI LLM instance pointed at OpenRouter."""
    return ChatOpenAI(
        model=OPENROUTER_MODEL,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_BASE_URL,
        temperature=0.1,
    )


def process_pdf(pdf_bytes, _embedding):
    """Process PDF, extract metrics, and build Chroma vectorstore."""
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        page_count = len(docs)
        char_count = sum(len(d.page_content) for d in docs)

        if char_count == 0:
            return None, page_count, char_count, 0

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100
        )
        chunks = text_splitter.split_documents(docs)
        chunk_count = len(chunks)

        # Build Chroma vectorstore
        from langchain_community.vectorstores import Chroma
        vectorstore = Chroma.from_documents(chunks, _embedding)

        return vectorstore, page_count, char_count, chunk_count
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


def search_tavily(query: str, max_results: int = 5):
    """Run a Tavily web search and return a list of result dicts."""
    client = TavilyClient(api_key=TAVILY_API_KEY)
    return client.search(query, max_results=max_results).get("results", [])


def format_web_context(results: list) -> str:
    """Turn Tavily search results into a plain-text context block."""
    if not results:
        return "No web search results found."
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(
            f"[Source {i}] {r.get('title', 'Untitled')}\n"
            f"URL: {r.get('url', 'N/A')}\n"
            f"{r.get('content', '')}"
        )
    return "\n\n".join(parts)


def status_badge(ok: bool, label: str) -> str:
    """Return a status badge markdown string."""
    color = "#10b981" if ok else "#ef4444"
    icon = "✅" if ok else "❌"
    return f"<span style='color:{color}; font-weight:600'>{icon} {label}</span>"


# ──────────────────────────────────────────────
# Streamlit Page
# ──────────────────────────────────────────────
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

# ── Custom CSS ───────────────────────────────
st.markdown(
    """
    <style>
        .main-header {
            background: linear-gradient(90deg, #4f46e5 0%, #06b6d4 100%);
            padding: 1.5rem;
            border-radius: 1rem;
            color: white;
            margin-bottom: 1rem;
        }
        .main-header h1 {
            margin: 0;
            font-size: 2.2rem;
        }
        .main-header p {
            margin: 0.5rem 0 0 0;
            opacity: 0.9;
        }
        .welcome-card {
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 1rem;
            padding: 2rem;
            text-align: center;
            color: #475569;
        }
        .stButton button {
            border-radius: 0.6rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ───────────────────────────────────
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown(
        """
        <div class="main-header">
            <h1>🤖 RAG Chatbot</h1>
            <p>PDF document QA &amp; live web search powered by OpenRouter + LangChain</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f"""
        <div style="background:#f8fafc; border-radius:1rem; padding:1rem; border:1px solid #e2e8f0;">
            <div style="font-size:0.85rem; color:#64748b; margin-bottom:0.4rem">Model</div>
            <div style="font-weight:600; color:#1e293b;">{OPENROUTER_MODEL}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Sidebar ──────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    mode = st.radio(
        "Answer source",
        ["PDF Document", "Web Search (Tavily)"],
        help="Choose whether answers come from a PDF or from live web search.",
    )

    uploaded_file = None
    if mode == "PDF Document":
        st.markdown("---")
        st.subheader("📄 PDF Upload")
        uploaded_file = st.file_uploader(
            "Upload a PDF",
            type=["pdf"],
            help="Your file is processed locally and never stored.",
        )
        if uploaded_file:
            import hashlib
            file_bytes = uploaded_file.getvalue()
            pdf_hash = hashlib.md5(file_bytes).hexdigest()

            # Check if this is a new/different PDF
            if st.session_state.get("pdf_hash") != pdf_hash:
                with st.spinner("Processing document…"):
                    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2")
                    vectorstore, page_count, char_count, chunk_count = process_pdf(file_bytes, embeddings)

                    st.session_state.vectorstore = vectorstore
                    st.session_state.pdf_hash = pdf_hash
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.page_count = page_count
                    st.session_state.chunk_count = chunk_count
                    st.session_state.char_count = char_count

                    # Clear chat history for new PDF
                    st.session_state.messages = []
                    st.rerun()

            # Display Diagnostics
            if st.session_state.get("char_count", 0) == 0:
                st.error("⚠️ This PDF contains no extractable text. It is likely scanned. OCR is required.")
            else:
                st.markdown(
                    f"""
                    <div style="background:#f8fafc; border-radius:0.6rem; padding:0.8rem; border:1px solid #e2e8f0; margin-top:0.5rem; font-size:0.9rem;">
                        <div style="font-weight:600; color:#1e293b; margin-bottom:0.4rem;">📄 Document Info</div>
                        <div style="color:#475569; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;"><b>Filename:</b> {st.session_state.pdf_name}</div>
                        <div style="color:#475569;"><b>Pages:</b> {st.session_state.page_count}</div>
                        <div style="color:#475569;"><b>Characters:</b> {st.session_state.char_count:,}</div>
                        <div style="color:#475569;"><b>Chunks:</b> {st.session_state.chunk_count}</div>
                        <div style="color:#475569;"><b>Embedding model:</b> all-MiniLM-L12-v2</div>
                        <div style="margin-top:0.4rem;">{status_badge(True, "Status: Ready")}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            # Clear stored PDF states if file is cleared
            st.session_state.pop("vectorstore", None)
            st.session_state.pop("pdf_hash", None)
            st.session_state.pop("pdf_name", None)
            st.session_state.pop("page_count", None)
            st.session_state.pop("chunk_count", None)
            st.session_state.pop("char_count", None)
            st.info("Upload a PDF file to start asking questions.")

    st.markdown("---")
    st.subheader("🔑 API Status")
    st.markdown(status_badge(bool(OPENROUTER_API_KEY), "OpenRouter"), unsafe_allow_html=True)
    st.markdown(status_badge(bool(TAVILY_API_KEY), "Tavily"), unsafe_allow_html=True)

    if not OPENROUTER_API_KEY or not TAVILY_API_KEY:
        st.warning("Add missing keys to your `.env` file and refresh.")

    st.markdown("---")
    if st.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption("Built with Streamlit · LangChain · OpenRouter · Tavily")

# ── Chat state ───────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Welcome or Render history ────────────────
if not st.session_state.messages:
    st.markdown(
        """
        <div class="welcome-card">
            <h3>👋 Welcome!</h3>
            <p>
                Ask questions about an uploaded PDF or use live web search.<br>
                Select your answer source in the sidebar to begin.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

# ── Prompt input ─────────────────────────────
prompt = st.chat_input("Ask something…")

if prompt:
    # Show & store user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # ── PDF RAG mode ─────────────────────────────────────────────────
    if mode == "PDF Document":
        if not uploaded_file:
            st.error("Please upload a PDF in the sidebar first.")
        elif not OPENROUTER_API_KEY:
            st.error("Set `OPENROUTER_API_KEY` in your environment or .env file.")
        elif st.session_state.get("char_count", 0) == 0:
            st.error("Cannot query a scanned/empty PDF without extractable text.")
        else:
            with st.spinner("Thinking…"):
                try:
                    vectorstore = st.session_state.vectorstore
                    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

                    # Check retrieval results before calling the chain
                    docs = retriever.invoke(prompt)
                    if not docs:
                        response = "No relevant information found in the uploaded PDF."
                        st.chat_message("assistant").markdown(response)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": response}
                        )
                    else:
                        chain = RetrievalQA.from_chain_type(
                            llm=get_llm(),
                            chain_type="stuff",
                            retriever=retriever,
                            return_source_documents=True,
                        )
                        result = chain.invoke({"query": prompt})
                        response = result.get("result", "No answer generated.")
                        source_docs = result.get("source_documents", [])

                        st.chat_message("assistant").markdown(response)
                        if source_docs:
                            with st.expander("📄 View source passages"):
                                for i, doc in enumerate(source_docs, 1):
                                    src = doc.metadata.get("source", "Unknown")
                                    page = doc.metadata.get("page")
                                    page_label = f" · page {page + 1}" if page is not None else ""
                                    st.markdown(
                                        f"<div style='background:#f8fafc; padding:0.8rem; "
                                        f"border-radius:0.6rem; border-left:4px solid #4f46e5; margin-bottom:0.6rem;'>"
                                        f"<div style='font-size:0.8rem; color:#64748b; margin-bottom:0.3rem;'>"
                                        f"Source {i}{page_label}</div>"
                                        f"<div style='color:#334155;'>{doc.page_content[:600]}…</div>"
                                        f"</div>",
                                        unsafe_allow_html=True,
                                    )

                        st.session_state.messages.append(
                            {"role": "assistant", "content": response}
                        )
                except Exception as e:
                    st.error(f"Error: {e}")

    # ── Tavily Web Search mode ───────────────────────────────────────
    else:
        if not TAVILY_API_KEY:
            st.error("Set `TAVILY_API_KEY` in your environment or .env file.")
        elif not OPENROUTER_API_KEY:
            st.error("Set `OPENROUTER_API_KEY` in your environment or .env file.")
        else:
            with st.spinner("Searching the web…"):
                try:
                    results = search_tavily(prompt, max_results=5)
                    context = format_web_context(results)

                    web_prompt = ChatPromptTemplate.from_template(
                        "You are a helpful assistant that answers questions using "
                        "information found on the web. Always cite sources by number.\n\n"
                        "Web search results:\n{context}\n\n"
                        "Question: {question}\n\n"
                        "Provide a comprehensive, well-structured answer. "
                        "Start directly — no filler phrases."
                    )
                    chain = web_prompt | get_llm() | StrOutputParser()
                    response = chain.invoke(
                        {"context": context, "question": prompt}
                    )

                    st.chat_message("assistant").markdown(response)
                    if results:
                        with st.expander("🌐 View web sources"):
                            for i, r in enumerate(results, 1):
                                st.markdown(
                                    f"<div style='background:#f0f9ff; padding:0.8rem; "
                                    f"border-radius:0.6rem; border-left:4px solid #06b6d4; margin-bottom:0.6rem;'>"
                                    f"<div style='font-weight:600; margin-bottom:0.3rem;'>"
                                    f"[{i}] [{r.get('title', 'Source')}]({r.get('url', '#')})</div>"
                                    f"<div style='font-size:0.85rem; color:#475569;'>{r.get('content', '')[:300]}…</div>"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )

                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )
                except Exception as e:
                    st.error(f"Error: {e}")
