from typing import List
import re
import numpy as np
from chromadb.api.types import Embedding, EmbeddingFunction


class TextChunker:
    def __init__(self) -> None:
        pass

    def __call__(self, document: str) -> List[str]:
        pass

    def _cosine_similarity(self, embed1: Embedding, embed2: Embedding):
        embed1 = np.asarray(embed1)
        embed2 = np.asarray(embed2)
        return float(
            np.dot(embed1, embed2) / (np.linalg.norm(embed1) * np.linalg.norm(embed2))
        )

    def _split_to_sentences(self, document: str) -> List[str]:
        math_pattern = (
            r"(\$.*?\$|\\\[.*?\\\]|\\begin\{equation\}.*?\\end\{equation\}|\\\(.*?\\\))"
        )
        parts = re.split(math_pattern, document, flags=re.DOTALL)

        sentences = []

        for part in parts:
            if part.strip():
                if re.match(math_pattern, part, flags=re.DOTALL):
                    sentences.append(part)
                else:
                    chunks = re.split(r"(?<=[.?!])\s+", part)
                    sentences.extend([c for c in chunks if c])

        merged_sentences = []
        i = 0
        while i < len(sentences):
            current = sentences[i]

            if (
                i > 0
                and merged_sentences
                and not merged_sentences[-1].rstrip().endswith((".", "?", "!"))
            ):
                merged_sentences[-1] = merged_sentences[-1] + current
            else:
                merged_sentences.append(current)

            i += 1

        return merged_sentences


class SemanticChunker(TextChunker):
    def __init__(
        self,
        embed_func: EmbeddingFunction,
        threshold: float = 0.75,
        max_chunk_length: int = 500,
    ) -> None:
        super().__init__()
        self.embed_func = embed_func
        self.threshold = threshold
        self.max_chunk_length = max_chunk_length

    def __call__(self, document: str) -> List[str]:
        sentences = self._split_to_sentences(document)
        embeddings = self.embed_func(sentences)

        chunks = []
        i = 0
        while i < len(sentences):
            if (
                i > 0
                and chunks
                and len(chunks[-1]) <= self.max_chunk_length
                and self._cosine_similarity(embeddings[i - 1], embeddings[i])
                > self.threshold
            ):
                chunks[-1] = chunks[-1] + sentences[i]
            else:
                chunks.append(sentences[i])

            i += 1

        return chunks
