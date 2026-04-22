"""
Build a FAISS index from raw CSV data and save to index/ directory.
Usage:  python scripts/build_index.py
"""

import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from models.embedding_model import EmbeddingModel
from rag.vector_store import VectorStore

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
INDEX_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "index")


def load_documents():
    """Load text documents from all raw CSVs."""
    docs = []

    # Banking77 — customer support queries
    for fname in ["banking77_train.csv", "banking77_test.csv"]:
        path = os.path.join(RAW_DIR, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                docs.append({
                    "text": str(row["text"]),
                    "source": "banking77",
                    "category": str(row.get("category", "")),
                })

    # Complaints
    path = os.path.join(RAW_DIR, "complaints.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, header=None)
        df.columns = ["tweet_id", "text", "label", "category"]
        for _, row in df.iterrows():
            docs.append({
                "text": str(row["text"]),
                "source": "complaints",
                "category": str(row.get("category", "")),
            })

    return docs


def main():
    print("Loading documents...")
    docs = load_documents()
    print(f"Loaded {len(docs)} documents.")

    if not docs:
        print("No documents found. Place CSVs in data/raw/ first.")
        return

    texts = [d["text"] for d in docs]

    print("Initializing embedding model...")
    model = EmbeddingModel()

    print("Encoding documents (this may take a minute)...")
    embeddings = model.encode_batch(texts, batch_size=128)
    print(f"Embeddings shape: {embeddings.shape}")

    print("Building FAISS index...")
    store = VectorStore(dim=embeddings.shape[1])
    store.add(embeddings, docs)
    print(f"Index size: {store.size()}")

    store.save(INDEX_DIR)
    print(f"Index saved to {INDEX_DIR}/")


if __name__ == "__main__":
    main()
