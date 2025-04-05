from sentence_transformers import SentenceTransformer
import chromadb
from .context_provider import ContextProvider

class DocumentContextProvider(ContextProvider):
    def __init__(self, collection, embedding_model, top_k=3):
        self.collection = collection
        self.embedding_model = embedding_model
        self.top_k = top_k

    def get_context(self, query: str) -> str:
        query_embedding = self.embedding_model.encode([query])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.top_k
        )

        documents = results.get("documents", [[]])[0]
        return "\n".join(documents)