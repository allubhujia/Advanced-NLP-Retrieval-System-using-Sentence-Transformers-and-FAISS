# rag_pipeline.py

from sentence_transformers import SentenceTransformer
import numpy as np

from models.embedding_model import EmbeddingModel
from models.sentiment_model import SentimentModel
from models.llm import LLMClient, PromptBuilder, LLMProviders


class RAGPipeline:
    """
    Orchestrates the full RAG generation stage:
      1. Embeds the query (for retriever handoff or re-ranking).
      2. Builds a grounded prompt from query + retrieved context.
      3. Generates an answer via the LLM.
      4. Optionally annotates the answer with sentiment metadata.
    """
    def __init__(
        self,
        embedding_model: EmbeddingModel | None = None,
        llm: LLMClient | None = None,
        sentiment_model: SentimentModel | None = None,
    ):
        self.embedding_model = embedding_model or EmbeddingModel()
        self.llm = llm or LLMClient()
        self.sentiment_model = sentiment_model or SentimentModel()
        self.prompt_builder = PromptBuilder()

    def run(
        self,
        query: str,
        retrieved_context: list[str],
        include_sentiment: bool = False,
    ) -> dict:
        """
        Args:
            query: The raw user question.
            retrieved_context: Ordered list of passage strings from the retriever.
            include_sentiment: If True, runs sentiment analysis on the answer.

        Returns:
            dict with keys: query, context, prompt, answer, [sentiment]
        """
        # Embed query (available for re-ranking or logging)
        query_embedding = self.embedding_model.encode(query)

        # Flatten context passages into a single block
        context_block = "\n\n".join(retrieved_context)

        # Build grounded prompt
        prompt = self.prompt_builder.build_prompt(query, context_block)

        # Generate answer
        answer = self.llm.generate(prompt)

        result = {
            "query": query,
            "context": context_block,
            "prompt": prompt,
            "answer": answer,
            "query_embedding_shape": query_embedding.shape,
        }

        if include_sentiment:
            result["sentiment"] = self.sentiment_model.predict(answer)

        return result


# --- Usage example ---

if __name__ == "__main__":
    pipeline = RAGPipeline(
        embedding_model=EmbeddingModel("all-MiniLM-L6-v2", device="cpu"),
        llm=LLMClient(config=LLMProviders().openai(model="gpt-4o")),
        sentiment_model=SentimentModel(device="cpu"),
    )

    query = "What is the capital of France?"
    context_passages = [
        "France is a country in Western Europe.",
        "Paris is the capital and largest city of France, with a population of over 2 million.",
    ]

    output = pipeline.run(query, context_passages, include_sentiment=True)

    print("Answer:", output["answer"])
    print("Sentiment:", output["sentiment"])