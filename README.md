# 🔍 End-to-End NLP-Based Retrieval-Augmented Generation (RAG) System

A production-ready RAG pipeline built in Python that combines dense vector embeddings with FAISS-powered similarity search to retrieve semantically relevant documents for any query. Designed for scalability, modularity, and seamless LLM integration.

---

## ✨ Features
- Dense semantic embeddings using `all-MiniLM-L6-v2` (Sentence Transformers)
- High-performance vector indexing and similarity search via FAISS
- Modular architecture — embedder, vector store, and retriever are fully decoupled
- Top-k document retrieval with formatted context output
- Persistent index — save and load FAISS indexes from disk
- Ready for LLM integration (OpenAI, HuggingFace, LangChain, LangGraph etc.)

---

## 🛠️ Tech Stack
| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Vector Store | FAISS (Facebook AI Similarity Search) |
| NLP Utilities | HuggingFace Transformers |
| Data Handling | NumPy, JSON |

---

## 📁 Project Structure
```
├── configs/                  # Configuration files
├── data/
│   ├── processed/            # Preprocessed datasets
│   └── raw/                  # Raw input data
├── database/                 # Database connectors / storage
├── evaluation/               # Evaluation scripts and metrics
├── llm_pipeline/             # LLM integration layer
├── models/
│   ├── embedding_model/      # Embedding model weights/config
│   ├── intent_model/         # Intent classification model
│   ├── llm/                  # LLM model artifacts
│   └── sentiment_model/      # Sentiment analysis model
├── nlp/
│   └── intent_training.py    # Intent model training script
├── notebooks/                # Jupyter notebooks for EDA & testing
├── rag/
│   ├── embedder.py           # Generates dense vector embeddings
│   ├── retriever.py          # Retrieves top-k docs & formats context
│   └── vector_store.py       # Manages FAISS index (add/search/save/load)
├── scripts/
│   ├── download_data.py      # Data download automation
│   └── process_data.py       # Data preprocessing pipeline
├── venv/                     # Virtual environment
├── .gitignore
└── main.py                   # Entry point
```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/your-username/rag-nlp-system.git
cd rag-nlp-system

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Core dependencies:**
```bash
pip install sentence-transformers faiss-cpu numpy
```

---

## 🚀 Usage

### 1. Initialize the Embedder
```python
from rag.embedder import Embedder

embedder = Embedder(model_name="all-MiniLM-L6-v2")
```

### 2. Add Documents to the Vector Store
```python
from rag.vector_store import VectorStore

documents = [
    "Python is a high-level, general-purpose programming language.",
    "FAISS enables efficient similarity search over dense vectors.",
    "Sentence Transformers produce semantically rich embeddings.",
    "RAG combines retrieval systems with generative language models.",
]

vector_store = VectorStore(dimension=384)
embeddings = embedder.encode(documents)
vector_store.add(embeddings, documents)

# Persist the index
vector_store.save("models/embedding_model/faiss_index")
```

### 3. Retrieve Relevant Documents
```python
from rag.retriever import Retriever

retriever = Retriever(embedder=embedder, vector_store=vector_store)

query = "How does FAISS perform similarity search?"
context = retriever.retrieve(query, top_k=3)

print(context)
# Returns formatted top-k documents most relevant to the query
```

---

## 🔄 How It Works

```
Query ──► Embedder ──► Query Vector ──► FAISS Index
                                            │
                                       Top-K IDs
                                            │
                                       Metadata Lookup
                                            │
                                        Retriever
                                            │
                                    Formatted Context
```

| Step | Module | Description |
|---|---|---|
| **1. Embed Query** | `embedder.py` | Converts raw query text into a 384-dim dense vector using `all-MiniLM-L6-v2` |
| **2. Vector Search** | `vector_store.py` | Runs L2/cosine similarity search over the FAISS index to fetch top-k IDs |
| **3. Metadata Lookup** | `vector_store.py` | Maps retrieved IDs back to original document text |
| **4. Context Assembly** | `retriever.py` | Formats and returns ranked documents as a structured context string |

---

## 🔮 Future Improvements
- **Reranking** — Integrate cross-encoder rerankers (e.g., `ms-marco-MiniLM`) for precision boost on top-k results
- **Hybrid Search** — Combine dense retrieval (FAISS) with sparse BM25 using Reciprocal Rank Fusion (RRF)
- **LLM Integration** — Pipe retrieved context directly into GPT-4 / Mistral / LLaMA for full RAG responses
- **Metadata Filtering** — Support filtered search by document type, date, or category
- **Streaming API** — Expose retrieval pipeline via a FastAPI endpoint with async support
- **Evaluation Suite** — Add RAGAS metrics (faithfulness, answer relevancy, context recall)

---

## 📄 License
This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.

---

