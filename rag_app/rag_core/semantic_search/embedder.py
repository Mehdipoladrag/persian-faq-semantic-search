import logging
from typing import List

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Embedder:
    """
    Embedder class for converting text to vector embeddings.
    
    Supports multilingual models and automatically adds prefixes
    for E5 family models (query: / passage:).
    """
    
    def __init__(self, model_name: str, device: str = "cpu") -> None:
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name or path of the sentence-transformers model
            device: Device to run the model on ('cpu' or 'cuda')
        """
        logger.info(f"Loading embedding model: {model_name} on {device}")
        self.model = SentenceTransformer(model_name, device=device)
        self.is_e5 = "e5" in model_name.lower()
    
    def encode_query(self, text: str) -> List[float]:
        """
        Encode a user query into vector embedding.
        
        Args:
            text: User query text
            
        Returns:
            List of floats representing the embedding vector
        """
        if self.is_e5:
            text = "query: " + text
        return self.model.encode(text).tolist()
    
    def encode_document(self, text: str) -> List[float]:
        """
        Encode a document text into vector embedding.
        
        Args:
            text: Document text content
            
        Returns:
            List of floats representing the embedding vector
        """
        if self.is_e5:
            text = "passage: " + text
        return self.model.encode(text).tolist()