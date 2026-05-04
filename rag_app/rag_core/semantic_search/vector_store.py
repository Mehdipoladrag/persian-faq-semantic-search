import os
import uuid
import logging
from typing import List, Dict, Optional, Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Vector database wrapper for Qdrant.
    Handles collection creation, document insertion, and similarity search.
    """

    def __init__(self, config: Any, embedder: Any) -> None:
        """
        Initialize Qdrant client and prepare collection.

        Args:
            config: Configuration object with vector_db settings
            embedder: Embedder instance for vector conversion
        """
        self.config = config
        self.embedder = embedder
        self.path = config.vector_db["path"]
        self.collection = config.vector_db["collection_name"]

        # Create directory and connect to Qdrant
        os.makedirs(self.path, exist_ok=True)
        self.client = QdrantClient(path=self.path)

        self._init_collection()

    def _init_collection(self) -> None:
        """Check if collection exists and set readiness flag."""
        collections = [c.name for c in self.client.get_collections().collections]
        self.collection_ready = self.collection in collections

    def _ensure_collection(self, vector_size: int) -> None:
        """
        Create collection if it doesn't exist.

        Args:
            vector_size: Dimension of embedding vectors
        """
        if self.collection_ready:
            return

        # Set distance metric
        distance = Distance.COSINE
        if self.config.vector_db.get("distance") == "Euclidean":
            distance = Distance.EUCLID
        elif self.config.vector_db.get("distance") == "Dot":
            distance = Distance.DOT

        # Create collection
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=vector_size, distance=distance),
        )
        self.collection_ready = True
        logger.info(f"Created collection '{self.collection}' with size {vector_size}")

    def add_documents(
        self, texts: List[str], metadatas: Optional[List[Dict]] = None
    ) -> None:
        """
        Add documents to vector database.

        Args:
            texts: List of text chunks to add
            metadatas: Optional list of metadata dicts for each chunk
        """
        if not texts:
            return

        if metadatas is None:
            metadatas = [{}] * len(texts)

        # Generate embeddings
        embeddings = [self.embedder.encode_document(text) for text in texts]
        vector_size = len(embeddings[0])
        self._ensure_collection(vector_size)

        # Create points
        points = []
        for text, metadata, embedding in zip(texts, metadatas, embeddings):
            point_id = str(uuid.uuid4())
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={"text": text, "metadata": metadata},
                )
            )

        # Insert into database
        self.client.upsert(collection_name=self.collection, points=points)
        logger.info(f"Added {len(points)} documents to Qdrant")

    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        Search for similar documents using semantic similarity.

        Args:
            query: User query string
            top_k: Number of results to return (default from config)

        Returns:
            List of dicts with 'text', 'metadata', and 'score' keys
        """
        if not self.collection_ready:
            return []

        if top_k is None:
            top_k = self.config.retrieval["top_k"]

        # Encode query and search
        query_embedding = self.embedder.encode_query(query)
        response = self.client.query_points(
            collection_name=self.collection,
            query=query_embedding,
            limit=top_k,
            with_payload=True,
        )

        # Format results
        results = []
        for hit in response.points:
            results.append(
                {
                    "text": hit.payload["text"],
                    "metadata": hit.payload.get("metadata", {}),
                    "score": hit.score,
                }
            )

        return results
