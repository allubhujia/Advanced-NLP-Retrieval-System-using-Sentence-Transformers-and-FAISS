import argparse
import sys
import os

# Add root directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.embedding_model import EmbeddingModel
from rag.vector_store import VectorStore
from rag.retriever import Retriever
from models.llm import PromptBuilder, LLMClient, LLMProviders
from nlp.intent_training import IntentClassifier

from models.sentiment_model.sentiment_model import SentimentModel


def main():
    parser = argparse.ArgumentParser(description="Advanced RAG Pipeline")
    parser.add_argument("--query", type=str, default=None, help="Query string")
    args = parser.parse_args()

    query = args.query or input("Enter your query: ")

    # --- 1. Initialize Modules ---
    try:
        # NLP Intent Classification & Sentiment
        intent_classifier = IntentClassifier()
        sentiment_model = SentimentModel()
        
        # RAG Components
        embedding_model = EmbeddingModel()
        vector_store = VectorStore.load("index/")
        retriever = Retriever(embedder=embedding_model, vector_store=vector_store)
        
        # LLM Generation
        prompt_builder = PromptBuilder()
        # Using qwen2.5-coder from Ollama as requested
        llm = LLMClient(config=LLMProviders().ollama(model="qwen2.5-coder"))
    
    except FileNotFoundError as e:
        print(f"\n[Error] {e}")
        return
    except Exception as e:
        print(f"\n[Error] Failed to initialize components: {e}")
        return

    # --- 2. Predict Intent (NLP Part) ---
    intent = intent_classifier.predict(query)
    print(f"\nDetected Intent: {intent}")

    # --- 3. Retrieve Context ---
    # We use the official Retriever class now
    context = retriever.get_context_for_llm(query, top_k=5)

    # --- 4. Generate Answer ---
    prompt = prompt_builder.build_prompt(query, context)
    answer = llm.generate(prompt)

    # --- 5. Analyze Sentiment ---
    sentiment = sentiment_model.predict(answer)

    print("\n" + "="*50)
    print(f"ANSWER:\n{answer}")
    print(f"\nSENTIMENT: {sentiment['label']} ({sentiment['confidence']:.2%})")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
