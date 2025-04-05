# mcp/profile_provider.py

from .context_provider import ContextProvider

class ProfileContextProvider(ContextProvider):
    def get_context(self, query: str) -> str:
        # This could later fetch from a DB or API
        return "User profile: Name - Achyuth, Role - Backend Engineer, Interests - AI, Productivity Tools"
