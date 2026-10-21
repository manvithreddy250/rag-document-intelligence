# AI Document Intelligence & RAG Assistant

A small Streamlit app that ingests text/Markdown documents, splits them into chunks, retrieves relevant passages using TF-IDF similarity, and optionally uses an OpenAI chat model to write a context-grounded answer with source filenames.

## Run
```bash
pip install -r requirements.txt
cp .env.example .env
# Optional: set OPENAI_API_KEY in .env; never commit it.
streamlit run app.py
```
On Windows, copy `.env.example` to `.env` manually. Without an API key, retrieval still works and displays the evidence passages.

## Features
- Upload multiple TXT/Markdown documents
- Chunking with overlap
- Top-k evidence retrieval
- Optional LLM answer grounded in retrieved text
- Source filename display

## Limitations
This is a portfolio demo. TF-IDF is a lexical retrieval baseline, not a production vector database. Do not upload confidential documents. Validate LLM answers before use.

## Smoke test

Run `python -m pytest test_smoke.py` after installing the requirements.
