from rag_engine import retrieve

def test_retrieval_returns_relevant_passage():
    docs = {"shipping.txt": "Standard shipping takes three to five business days.",
            "security.txt": "Never share your password or one-time passcode."}
    hits = retrieve("How long does shipping take?", docs)
    assert hits
    assert hits[0]["source"] == "shipping.txt"
