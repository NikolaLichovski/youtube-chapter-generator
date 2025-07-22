# embedder.py
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class TextEmbedder:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded embedding model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {str(e)}")
            # Fallback
            try:
                fallback_model = 'all-MiniLM-L6-v2'
                self.model = SentenceTransformer(fallback_model)
                logger.info(f"Loaded fallback model: {fallback_model}")
            except Exception as e2:
                logger.error(f"Fallback model loading failed: {str(e2)}")
                raise

    def embed_texts(self, texts: List[str]) -> Optional[np.ndarray]:
        if not self.model or not texts:
            logger.error("Model not loaded or empty texts")
            return None
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=False, show_progress_bar=False)
            return np.array(embeddings)
        except Exception as e:
            logger.error(f"Embedding generation error: {str(e)}")
            return None

    def embed_single_text(self, text: str) -> Optional[np.ndarray]:
        embeddings = self.embed_texts([text])
        return embeddings[0] if embeddings is not None else None
