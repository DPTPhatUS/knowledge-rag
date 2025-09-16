import json
from typing import Any, List, Literal, Dict, Optional, Tuple
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
    vectors_matrix: np.ndarray = field(init=False)
    id_index_map: Dict[str, int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if not self.records:
            self.vectors_matrix = np.zeros((0, self.embedding_dim), dtype=np.float32)
            self.id_index_map = {}
        else:
            self.vectors_matrix = np.vstack([record.vector for record in self.records])
            self.id_index_map = {record.id: i for i, record in enumerate(self.records)}


class VectorDB:
    def __init__(
        self,
        embedding_dim: int = 512,
        metric: Literal["cosine", "euclid", "inner"] = "cosine",
        storage_path: str = COLLECTION_NAME,
        auto_save: bool = True
    ) -> None:
        self.embedding_dim = embedding_dim
        self.metric = metric
        self.storage_path = storage_path
        self.database = Database(embedding_dim=embedding_dim)
        self.auto_save = auto_save
        
        if auto_save:
            self.load()

    def add(
        self, vector: np.ndarray, id: str, metadata: Dict[str, Any] | None = None
    ) -> None:
        if id in self.database.id_index_map:
            raise ValueError(f"Duplicate ID found: {id}")
        if vector.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch, expected: {self.embedding_dim}, but got: {vector.shape[-1]}"
            )

        if metadata is None:
            metadata = {}

        new_record = Record(id=id, vector=vector.astype(np.float32), metadata=metadata)
        self.database.records.append(new_record)
        self.database.vectors_matrix = np.vstack(
            [self.database.vectors_matrix, new_record.vector]
        )
        self.database.id_index_map[id] = len(self.database.records) - 1
        
        if self.auto_save:
            self.save()

    def add_batch(
        self,
        vectors: List[np.ndarray],
        ids: List[str],
        metadatas: List[Dict[str, Any] | None] | None = None,
    ) -> None:
        if metadatas is None:
            metadatas = [{}] * len(vectors)

        for vec, id, meta in zip(vectors, ids, metadatas):
            self.add(vector=vec, id=id, metadata=meta)

    def search(
        self,
        query: np.ndarray,
        top_k: int = 5,
        metric: Literal["cosine", "euclid", "inner"] | None = None,
    ) -> List[Tuple[str, float]]:
        if query.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Query dimension mismatch, expected: {self.embedding_dim}, but got: {query.shape[-1]}"
            )

        if len(self) == 0:
            return []

        vectors = self.database.vectors_matrix

        if metric is None:
            metric = self.metric

        if metric == "cosine":
            query = query / np.linalg.norm(query)
            vectors = vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)
            cosines = vectors @ query
            top = np.argsort(-cosines)[:top_k]
            return [(self.database.records[i].id, float(cosines[i])) for i in top]

        if metric == "inner":
            inners = vectors @ query
            top = np.argsort(-inners)[:top_k]
            return [(self.database.records[i].id, float(inners[i])) for i in top]

        dists = np.linalg.norm(vectors - query)
        top = np.argsort(dists)[:top_k]
        return [(self.database.records[i].id, float(dists[i])) for i in top]

    def get(self, id: str) -> Optional[Record]:
        idx = self.database.id_index_map[id]
        if idx is None:
            return None

        return self.database.records[idx]

    def delete(self, id: str) -> None:
        idx = self.database.id_index_map.pop(id)
        if idx is None:
            return

        self.database.records.pop(idx)
        self.database.vectors_matrix = np.delete(
            self.database.vectors_matrix, idx, axis=0
        )
        self.database.id_index_map = {
            rec.id: i for i, rec in enumerate(self.database.records)
        }

    def __len__(self) -> int:
        return len(self.database.records)

    def save(self) -> None:
        save_data = {
            "embedding_dim": self.embedding_dim,
            "metric": self.metric,
            "records": [
                {"id": rec.id, "vector": rec.vector.tolist(), "metadata": rec.metadata}
                for rec in self.database.records
            ],
        }

        file_path = self.storage_path + "/vectors.json"
        with open(file_path, "w") as file:
            json.dump(save_data, file, indent=2)

    def load(self, file_path: str | None = None) -> None:
        if file_path is None:
            file_path = self.storage_path + "/vectors.json"

        try:
            with open(file_path, "r") as file:
                data = json.load(file)

            self.embedding_dim = data["embedding_dim"]
            self.metric = data["metric"]

            records = []
            for rec_data in data["records"]:
                records.append(
                    Record(
                        rec_data["id"],
                        np.array(rec_data["vector"]),
                        rec_data["metadata"],
                    )
                )
            self.database = Database(self.embedding_dim, records)
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            print(f"Could not load database from {file_path}: {e}")
            self.database = Database(embedding_dim=self.embedding_dim)
