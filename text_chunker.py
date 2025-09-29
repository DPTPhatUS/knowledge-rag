from typing import List, Callable
import re
import numpy as np
from embedder import Embedding, OllamaEmbedder
from globals import OLLAMA_EMBED_MODEL


class TextChunker:
    def __init__(self) -> None:
        pass

    def __call__(self, document: str) -> List[str]:
        pass


class FixedChunker(TextChunker):
    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.overlap = min(overlap, chunk_size - 1)

    def __call__(self, document: str) -> List[str]:
        chunks = []
        step = self.chunk_size - self.overlap
        for i in range(0, len(document), step):
            chunk = document[i : i + self.chunk_size]
            chunks.append(chunk)
        return chunks


class RecursiveChunker(TextChunker):
    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.overlap = min(overlap, chunk_size - 1)

    def __call__(self, document: str) -> List[str]:
        chunks = self._split_recursive(document.strip())
        return self._apply_overlap(chunks)

    def _split_recursive(self, text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        paragraphs = re.split(r"\n\s*\n", text)
        if len(paragraphs) > 1:
            return self._split_by_units(paragraphs)

        sentences = re.split(r"(?<=[.?!])\s+", text)
        if len(sentences) > 1:
            return self._split_by_units(sentences)

        return [
            text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)
        ]

    def _split_by_units(self, units: List[str]) -> List[str]:
        chunks = []
        current = ""
        for u in units:
            u = u.strip()
            if not u:
                continue

            if len(current) + len(u) + 1 <= self.chunk_size:
                current += (" " if current else "") + u
            else:
                if current:
                    chunks.append(current)
                if len(u) > self.chunk_size:
                    chunks.extend(self._split_recursive(u))
                    current = ""
                else:
                    current = u

        if current:
            chunks.append(current)
        return chunks

    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        if not chunks:
            return []

        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = overlapped[-1]
            overlap_text = prev[-self.overlap :] if len(prev) > self.overlap else prev
            overlapped.append(overlap_text + " " + chunks[i])
        return overlapped


class SemanticChunker(TextChunker):
    def __init__(
        self,
        embed_func: Callable[[List[str]], List[Embedding]],
        threshold: float = 0.5,
        max_chunk_size: int = 500,
    ) -> None:
        super().__init__()
        self.embed_func = embed_func
        self.threshold = threshold
        self.max_chunk_size = max_chunk_size

    def __call__(self, document: str) -> List[str]:
        sentences = self._split_to_sentences(document)
        embeddings = self.embed_func(sentences)

        chunks = []
        i = 0
        while i < len(sentences):
            if (
                i > 0
                and chunks
                and len(chunks[-1]) <= self.max_chunk_size
                and self._cosine_similarity(embeddings[i - 1], embeddings[i])
                > self.threshold
            ):
                chunks[-1] = chunks[-1] + " " + sentences[i]
            else:
                chunks.append(sentences[i])
            i += 1
        return chunks

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
                merged_sentences[-1] = merged_sentences[-1] + " " + current
            else:
                merged_sentences.append(current)

            i += 1

        return merged_sentences


if __name__ == "__main__":
    document = """This is the first paragraph. It has a few sentences. 
    Here is another sentence in the first paragraph.

    This is the second paragraph. It should be split separately if the chunker is smart.
    Sometimes, paragraphs are longer. They can contain multiple sentences!

    Mathematics inline $E=mc^2$ and display:
    \[
    a^2 + b^2 = c^2
    \]
    are also included."""

    embed_func = OllamaEmbedder(model_name=OLLAMA_EMBED_MODEL)

    fixed_chunker = FixedChunker(chunk_size=100, overlap=20)
    recursive_chunker = RecursiveChunker(chunk_size=100, overlap=20)
    semantic_chunker = SemanticChunker(
        embed_func=embed_func, threshold=0.6, max_chunk_size=100
    )

    print(fixed_chunker(document))
    print(recursive_chunker(document))
    print(semantic_chunker(document))
