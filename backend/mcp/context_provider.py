# mcp/context_provider.py

from abc import ABC, abstractmethod

class ContextProvider(ABC):
    @abstractmethod
    def get_context(self, query: str) -> str:
        """
        Given a user query, return a string containing relevant context.
        """
        pass
