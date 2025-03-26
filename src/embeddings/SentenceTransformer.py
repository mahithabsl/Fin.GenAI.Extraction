import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config
from sentence_transformers import SentenceTransformer



class SentenceTransformersEmbedder():
    """
    SentenceTransformersEmbedder base class for Verba.
    This class handles the generation of embeddings using SentenceTransformer models.
    """

    def __init__(self):
        """
        Initializes the SentenceTransformersEmbedder with a default model configuration.
        """
        super().__init__()
        self.config = {
            "Model": config.get('embeddings', {}).get('sentence_transformer', {}).get('model')
        }

    def vectorize(self, content: list[str]) -> list[float]:
        """
        Generates vector embeddings for the provided content using the specified SentenceTransformer model.

        Args:
            content (list[str]): A list of text strings to be vectorized.

        Returns:
            list[float]: A list of vector embeddings representing the input content.

        Raises:
            Exception: If vectorization fails due to an error in the embedding process.
        """
        try:
            model_name = self.config.get("Model")
            model = SentenceTransformer(model_name)
            embeddings = model.encode(content).tolist()
            return embeddings
        except Exception as e:
            raise Exception(f"Failed to vectorize chunks: {str(e)}")
