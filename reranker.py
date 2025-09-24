import numpy as np
from typing import List
from FlagEmbedding import FlagReranker
import torch
from globals import RERANK_MODEL


class Reranker:
    def __init__(
        self,
        model_name: str,
        device: str = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.mps.is_available() else "cpu"
        ),
        normalize: bool = False,
    ):
        self.model = FlagReranker(model_name, devices=device, normalize=normalize)

    def rerank(
        self, documents: List[str], query: str, top_k: int | None = None
    ) -> List[str]:
        pairs = [[query, doc] for doc in documents]

        scores = self.model.compute_score(pairs)
        top = np.argsort(-np.array(scores))[:top_k]

        return [documents[i] for i in top]


if __name__ == "__main__":
    reranker = Reranker(model_name=RERANK_MODEL)
    results = reranker.rerank(
        documents=[
            "hi",
            "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.",
        ],
        query="What is panda?",
    )
    print(results)
