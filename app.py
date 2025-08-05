from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import logging
from functools import lru_cache
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class EmbeddingService:
    """Simple embedding service for search queries"""

    def __init__(self):
        # Choose the model that matches your vector dimension in Redis
        # Common dimensions: 384 (MiniLM), 768 (mpnet), 512 (distilbert)
        self.model_name = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')  # 384 dimensions
        self.model = None

    @lru_cache(maxsize=1)
    def load_model(self):
        """Load and cache the embedding model"""
        if self.model is None:
            try:
                logger.info(f"Loading embedding model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
                logger.info(f"Model loaded successfully. Dimension: {self.model.get_sentence_embedding_dimension()}")
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")
                raise
        return self.model

    def get_embedding(self, query: str) -> list:
        """
        Generate embedding for a single query - matches your Java interface
        Returns: List of floats (can be converted to float[] in Java)
        """
        model = self.load_model()

        # Generate normalized embedding (important for cosine similarity in Redis)
        embedding = model.encode(query, normalize_embeddings=True)

        # Convert numpy array to Python list for JSON serialization
        return embedding.tolist()

# Initialize service
embedding_service = EmbeddingService()

@app.route('/embedding-service/api/v1/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Test model loading
        model = embedding_service.load_model()
        dimension = model.get_sentence_embedding_dimension()

        return jsonify({
            'status': 'healthy',
            'model': embedding_service.model_name,
            'dimension': dimension
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/embedding-service/api/v1/embed', methods=['POST'])
def get_embedding():
    """
    Generate embedding for a query - direct replacement for embeddingService.getEmbedding(query)

    Expected JSON payload:
    {
        "query": "your search query here"
    }

    Returns:
    {
        "embedding": [0.1, 0.2, -0.3, ...],  // Array of floats
        "dimension": 384
    }
    """
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400

        data = request.get_json()

        if 'query' not in data:
            return jsonify({'error': 'Missing required field: query'}), 400

        query = data['query']
        if not isinstance(query, str) or not query.strip():
            return jsonify({'error': 'Query must be a non-empty string'}), 400

        # Generate embedding
        embedding = embedding_service.get_embedding(query.strip())

        return jsonify({
            'embedding': embedding,
            'dimension': len(embedding)
        })

    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        return jsonify({'error': 'Failed to generate embedding'}), 500

@app.route('/embedding-service/api/v1/embed/batch', methods=['POST'])
def get_batch_embeddings():
    """
    Generate embeddings for a batch of queries.

    Expected JSON payload:
    {
        "queries": ["query 1", "query 2", ...]
    }

    Returns:
    {
        "embeddings": [[...], [...], ...],  // List of embeddings
        "dimension": 384
    }
    """
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400

        data = request.get_json()

        if 'queries' not in data:
            return jsonify({'error': 'Missing required field: queries'}), 400

        queries = data['queries']
        if not isinstance(queries, list) or not all(isinstance(q, str) and q.strip() for q in queries):
            return jsonify({'error': 'queries must be a non-empty list of strings'}), 400

        model = embedding_service.load_model()
        embeddings = model.encode(queries, normalize_embeddings=True)

        return jsonify({
            'embeddings': [e.tolist() for e in embeddings],
            'dimension': len(embeddings[0]) if embeddings else 0
        })

    except Exception as e:
        logger.error(f"Error generating batch embeddings: {str(e)}")
        return jsonify({'error': 'Failed to generate batch embeddings'}), 500


if __name__ == '__main__':
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'

    logger.info(f"Starting Embedding Service on {host}:{port}")
    logger.info(f"Model: {embedding_service.model_name}")

    app.run(host=host, port=port, debug=debug)