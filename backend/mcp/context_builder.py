# mcp/context_builder.py

class MCPContextBuilder:
    def __init__(self, providers):
        self.providers = providers

    def build_prompt(self, query: str) -> str:
        context_blocks = [provider.get_context(query) for provider in self.providers]

        full_context = "\n\n---\n\n".join(context_blocks)

        prompt = f"""You are a helpful assistant. Use the following context to answer the question.

Context:
{full_context}

Question: {query}
Answer:"""

        return prompt
