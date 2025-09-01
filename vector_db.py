from typing import Any, List, Literal, Dict, Tuple
from globals import COLLECTION_NAME
import numpy as np
from dataclasses import dataclass, field


@dataclass
class Record:
    id: str
    vector: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Database:
    embedding_dim: int = 512
    records: List[Record] = field(default_factory=list)


class VectorDB:
    def __init__(
        self,
        embedding_dim: int = 512,
        metric: Literal["cosine", "l2"] = "cosine",
        file_path: str = COLLECTION_NAME + ".json",
    ) -> None:
        self.embedding_dim = embedding_dim
        self.metric = metric
        self.file_path = file_path
        self.database = Database(embedding_dim=embedding_dim)

    def add(
        self, vector: np.ndarray, id: str, metadata: Dict[str, Any] | None = None
    ) -> None:
        assert (
            vector.shape[-1] == self.embedding_dim
        ), f"Embedding dimension mismatch, expected: {self.embedding_dim}, but loaded: {vector.shape[-1]}"

        if metadata is None:
            metadata = {}

        new_record = Record(id=id, vector=vector, metadata=metadata)
        self.database.records.append(new_record)

    def add_batch(
        self,
        vectors: List[np.ndarray],
        ids: List[str],
        metadatas: List[Dict[str, Any] | None] | None = None,
    ) -> None:
        if metadatas is None:
            metadatas = [None] * len(vectors)

        for vec, id, meta in zip(vectors, ids, metadatas):
            self.add(vector=vec, id=id, metadata=meta)

    def search(self, query: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        pass

    def get(self, id: str) -> Record:
        pass

    def delete(self, id: str) -> None:
        pass
