from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class DirectEmbeddingService:
    """Direct embedding service for testing - no Flask needed"""

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = None

    def load_model(self):
        """Load the embedding model"""
        if self.model is None:
            print(f"Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model loaded! Dimension: {self.model.get_sentence_embedding_dimension()}")
        return self.model

    def getEmbedding(self, query: str) -> list:
        """
        Generate embedding for a query - matches your Java interface
        """
        model = self.load_model()
        embedding = model.encode(query, normalize_embeddings=True)
        return embedding.tolist()

    def calculate_similarity(self, embedding1: list, embedding2: list) -> float:
        """Calculate cosine similarity between two embeddings"""
        vec1 = np.array(embedding1).reshape(1, -1)
        vec2 = np.array(embedding2).reshape(1, -1)
        similarity = cosine_similarity(vec1, vec2)[0][0]
        return float(similarity)

embedding_service = DirectEmbeddingService()
