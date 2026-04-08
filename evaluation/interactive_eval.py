"""
Interactive RAG Evaluation Script
==================================
Run from project root:
    python -m evaluation.interactive_eval
"""

import sys
import time

from models.embedding_model import EmbeddingModel
from models.sentiment_model import SentimentModel
from models.llm import LLMClient, LLMProviders, PromptBuilder
from nlp.intent_training import IntentClassifier
from llm_pipeline.llm_pipeline import RAGPipeline
from rag import Retriever


# ── Formatting helpers ──────────────────────────────────────────────

DIVIDER = "─" * 60

def print_header():
    print()
    print(DIVIDER)
    print("  RAG Interactive Evaluation")
    print("  Type 'quit' or 'exit' to stop. 'clear' to reset screen.")
    print(DIVIDER)
    print()


def print_result(result: dict, intent: str, retrieval_time: float):
    print()
    print(DIVIDER)
    print(f"  Intent        : {intent}")
    print(f"  Retrieval     : {retrieval_time:.3f}s")

    if "answer" in result:
        print(f"  Answer        : {result['answer']}")

    sentiment = result.get("sentiment")
    if sentiment:
        label = sentiment.get("label", "N/A")
        confidence = sentiment.get("confidence", 0.0)
        print(f"  Sentiment     : {label} ({confidence:.2%})")

    sources = result.get("sources")
    if sources:
        print(f"  Sources used  : {len(sources)}")
        for i, src in enumerate(sources, 1):
            preview = (src[:80] + "…") if len(src) > 80 else src
            print(f"    [{i}] {preview}")

    print(DIVIDER)
    print()


# ── Initialisation ──────────────────────────────────────────────────

def init_components():
    """Load all models and return the ready-to-use pipeline + helpers."""

    print("Loading embedding model …")
    embedding_model = EmbeddingModel()

    print("Loading sentiment model …")
    sentiment_model = SentimentModel()

    print("Loading LLM client …")
    llm_client = LLMClient(provider=LLMProviders.DEFAULT)
    prompt_builder = PromptBuilder()

    print("Loading intent classifier …")
    intent_classifier = IntentClassifier()

    print("Loading retriever …")
    retriever = Retriever()

    print("Building RAG pipeline …")
    pipeline = RAGPipeline(
        embedding_model=embedding_model,
        sentiment_model=sentiment_model,
        llm_client=llm_client,
        prompt_builder=prompt_builder,
    )

    print("All components ready.\n")
    return pipeline, intent_classifier, retriever, embedding_model


# ── Main loop ───────────────────────────────────────────────────────

TOP_K = 5

def main():
    print_header()

    try:
        pipeline, intent_classifier, retriever, embedding_model = init_components()
    except Exception as exc:
        print(f"[ERROR] Failed to initialise components: {exc}", file=sys.stderr)
        sys.exit(1)

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit"):
            print("Goodbye.")
            break
        if query.lower() == "clear":
            print("\033[2J\033[H", end="")
            print_header()
            continue

        # 1. Classify intent
        intent = intent_classifier.predict(query)

        # 2. Retrieve context (skip retrieval for pure chat intent)
        retrieved_context = []
        t0 = time.perf_counter()

        if intent != "chat":
            retrieved_context = retriever.search(query, top_k=TOP_K)

        retrieval_time = time.perf_counter() - t0

        # 3. Run the pipeline
        include_sentiment = intent != "chat"
        try:
            result = pipeline.run(
                query=query,
                retrieved_context=retrieved_context,
                include_sentiment=include_sentiment,
            )
        except Exception as exc:
            print(f"\n[ERROR] Pipeline failed: {exc}\n")
            continue

        # 4. Display results
        print_result(result, intent, retrieval_time)


if __name__ == "__main__":
    main()