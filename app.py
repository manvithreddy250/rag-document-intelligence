import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from rag_engine import retrieve, answer_question

load_dotenv()
st.set_page_config(page_title="Document RAG Assistant", page_icon="📚", layout="wide")
st.title("📚 AI Document Intelligence & RAG Assistant")
st.caption("Upload documents, retrieve relevant evidence, and optionally generate grounded answers.")
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Retrieved passages", 1, 5, 3)
    api_key = os.getenv("OPENAI_API_KEY", "")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    st.info("Without an API key, the app still demonstrates document retrieval and source passages.")
documents = {}
sample_dir = Path(__file__).parent / "sample_docs"
for path in sample_dir.glob("*.txt"):
    documents[path.name] = path.read_text(encoding="utf-8")
uploads = st.file_uploader("Upload TXT or Markdown files", type=["txt", "md"], accept_multiple_files=True)
for upload in uploads or []:
    documents[upload.name] = upload.getvalue().decode("utf-8", errors="replace")
st.write(f"**Documents loaded:** {len(documents)}")
question = st.text_input("Ask a question", placeholder="What is the refund policy?")
if st.button("Search documents", type="primary") and question:
    passages = retrieve(question, documents, top_k=top_k)
    st.subheader("Answer")
    st.write(answer_question(question, passages, api_key, model))
    st.subheader("Retrieved evidence")
    if not passages:
        st.warning("No relevant passages found.")
    for p in passages:
        with st.expander(f"{p['source']} · chunk {p['chunk']} · relevance {p['score']:.3f}"):
            st.write(p["text"])
st.divider()
st.caption("Demo note: retrieval uses TF-IDF. LLM generation is optional and requires your own API key.")
