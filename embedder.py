import numpy as np
from typing import List
from globals import OLLAMA_EMBED_MODEL


Embedding = np.ndarray


class Embedder:
    def __init__(self):
        pass

    def __call__(self, texts: List[str]) -> List[Embedding]:
        pass


class OllamaEmbedder(Embedder):
    def __init__(self, model_name: str = OLLAMA_EMBED_MODEL):
        try:
            import ollama
        except ImportError:
            raise ValueError(
                "The `ollama` package is not installed. Install it with `pip install ollama`"
            )

        super().__init__()
        self.model_name = model_name
        self._ollama = ollama

    def __call__(self, texts: List[str]) -> List[Embedding]:
        response = self._ollama.embed(
            model=self.model_name, input=texts, keep_alive=False
        )
        return [
            np.asarray(embedding, dtype=np.float32) for embedding in response.embeddings
        ]


class HuggingFaceEmbedder(Embedder):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ValueError(
                "The `sentence_transformers` package is not installed. Install it with `pip install sentence-transformers`"
            )

        super().__init__()
        self.model = SentenceTransformer(model_name)
        
    def __call__(self, texts: List[str]) -> List[Embedding]:
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return [
            np.asarray(embedding, dtype=np.float32) for embedding in embeddings
        ]
