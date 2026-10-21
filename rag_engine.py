from pathlib import Path
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

def split_text(text: str, chunk_size: int = 900, overlap: int = 120) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks

def retrieve(question: str, documents: Dict[str, str], top_k: int = 3) -> List[dict]:
    chunks = []
    for name, text in documents.items():
        for i, chunk in enumerate(split_text(text)):
            chunks.append({"source": name, "chunk": i + 1, "text": chunk})
    if not chunks or not question.strip():
        return []
    corpus = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(corpus + [question])
    scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    ranked = scores.argsort()[::-1][:top_k]
    return [{**chunks[i], "score": float(scores[i])} for i in ranked if scores[i] > 0]

def answer_question(question: str, passages: List[dict], api_key: str = "", model: str = "gpt-4o-mini") -> str:
    if not passages:
        return "I couldn't find relevant information in the uploaded documents. Try rephrasing the question or adding a relevant document."
    context = "\n\n".join(f"[Source: {p['source']}] {p['text']}" for p in passages)
    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": "Answer only from the supplied context. If the answer is not in the context, say you cannot find it. Cite source filenames in your answer."},
                    {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"}
                ],
            )
            return response.choices[0].message.content or "No answer was returned."
        except Exception as exc:
            return f"LLM call failed ({type(exc).__name__}). Retrieved evidence is shown below; check your API key/model configuration."
    return "I found these relevant passages. Add an OPENAI_API_KEY in a local .env file to generate a grounded answer.\n\n" + "\n\n".join(
        f"**{p['source']}** — {p['text']}" for p in passages
    )
