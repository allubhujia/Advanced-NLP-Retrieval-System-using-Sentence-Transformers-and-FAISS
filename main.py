import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.embedding_model import EmbeddingModel
from rag.vector_store import VectorStore
from models.llm import PromptBuilder, LLMClient, LLMProviders


def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline")
    parser.add_argument("--query", type=str, default=None, help="Query string")
    args = parser.parse_args()

    query = args.query or input("Enter your query: ")

    # --- Initialize modules ---
    embedding_model = EmbeddingModel()
    vector_store = VectorStore.load("index/")
    prompt_builder = PromptBuilder()
    llm = LLMClient(config=LLMProviders().ollama(model="qwen2.5-coder"))

    # --- Retrieve ---
    query_vec = embedding_model.encode(query)
    results = vector_store.search(query_vec, top_k=5)
    context = "\n\n---\n\n".join(meta.get("text", "") for meta, _ in results)

    # --- Generate ---
    prompt = prompt_builder.build_prompt(query, context)
    answer = llm.generate(prompt)

    print(answer)


if __name__ == "__main__":
    main()
