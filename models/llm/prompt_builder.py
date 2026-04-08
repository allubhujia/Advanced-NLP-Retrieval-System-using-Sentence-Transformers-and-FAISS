class PromptBuilder:
    """
    Assembles structured prompts from a query and retrieved context.
    Template enforces context-grounding and explicit instruction.
    """
    TEMPLATE = (
        "Context:\n{context}\n\n"
        "Question:\n{query}\n\n"
        "Instruction:\n"
        "Answer the question using only the provided context. "
        "If the context does not contain enough information, say so explicitly."
    )

    def build_prompt(self, query: str, context: str) -> str:
        return self.TEMPLATE.format(context=context.strip(), query=query.strip())
