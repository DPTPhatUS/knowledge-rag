import os
from vector_db import VectorDB
from transformers import VisionEncoderDecoderModel, NougatProcessor
import torch
from pdf2image import convert_from_path
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from globals import *
from text_chunker import SemanticChunker
import uuid
import ollama
from typing import Iterable, Dict, Any
from functools import partial
from reranker import Reranker
from datetime import datetime


class KnowledgeRAG:
    def __init__(
        self,
        storage_path: str | None = "collections",
        device: str | None = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.mps.is_available() else "cpu"
        ),
    ) -> None:
        self.ocr_processor = NougatProcessor.from_pretrained(HUGGINGFACE_OCR_MODEL)
        self.ocr_model = VisionEncoderDecoderModel.from_pretrained(
            HUGGINGFACE_OCR_MODEL
        )
        self.ocr_model.to(device)
        self.device = device

        self.embedding_model = OllamaEmbeddingFunction(model_name=OLLAMA_EMBED_MODEL)
        self.llm_model = partial(
            ollama.generate, model=OLLAMA_LLM_MODEL, keep_alive=False
        )

        self.chunker = SemanticChunker(
            embed_func=self.embedding_model, threshold=0.4, max_chunk_size=500
        )

        self.reranker = Reranker(RERANK_MODEL, device=device)

        self.vector_db = VectorDB(
            embedding_dim=768, storage_path=f"{storage_path}/text", auto_save=True
        )

        self.docs_path = f"{storage_path}/docs"
        os.makedirs(self.docs_path, exist_ok=True)

    def add_document(self, file_path: str) -> None:
        _, ext = os.path.splitext(file_path.lower())
        if ext == ".pdf":
            file_pages_images = convert_from_path(file_path, fmt="jpeg")

            pages_texts = []
            for page_image in file_pages_images:
                rgb_image = page_image.convert("RGB")
                pixel_values = self.ocr_processor(
                    rgb_image, return_tensors="pt"
                ).pixel_values
                output = self.ocr_model.generate(
                    pixel_values.to(self.device),
                    max_new_tokens=2048,
                    bad_words_ids=[[self.ocr_processor.tokenizer.unk_token_id]],
                )
                page_text = self.ocr_processor.batch_decode(
                    output, skip_special_tokens=True
                )[0]
                page_text = self.ocr_processor.post_process_generation(
                    page_text, fix_markdown=False
                )
                pages_texts.append(page_text)

            file_content = "\n\n".join(pages_texts)
        elif ext == ".txt":
            with open(file_path, "r") as file:
                file_content = file.read()
        else:
            raise ValueError("Unsupported file format, expected pdf or txt")

        chunks = self.chunker(file_content)
        chunk_ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
        chunk_metadatas = [
            {
                "file": file_path,
                "date": datetime.now().date().isoformat(),
                "time": datetime.now().time().isoformat(timespec="seconds"),
            }
        ] * len(chunks)
        embeddings = self.embedding_model(chunks)

        self.vector_db.add_batch(embeddings, chunk_ids, chunk_metadatas)

        for id, chunk in zip(chunk_ids, chunks):
            chunk_file_path = f"{self.docs_path}/{id}.txt"
            with open(chunk_file_path, "w") as chunk_file:
                chunk_file.write(chunk)

    def delete_document(self, file_path: str) -> None:
        ids_to_delete = self.vector_db.delete(metadata_filter={"file": file_path})
        for doc_id in ids_to_delete:
            os.remove(f"{self.docs_path}/{doc_id}.txt")

    def answer_question(self, query: str) -> Iterable[ollama.GenerateResponse]:
        query_embedding = self.embedding_model([query])[0]

        search_results = self.vector_db.search(query=query_embedding, top_k=10)
        results_ids = [id for id, _ in search_results]

        def _get_doc_from_id(doc_id: str):
            doc_file_path = f"{self.docs_path}/{doc_id}.txt"
            with open(doc_file_path, "r") as doc_file:
                return doc_file.read()

        results_docs = [_get_doc_from_id(id) for id in results_ids]
        print(results_docs)
        results_docs = self.reranker.rerank(results_docs, query=query, top_k=5)
        print(results_docs)

        context = "\n".join(results_docs)
        prompt = BASE_PROMPT.format(context=context, question=query)

        return self.llm_model(prompt=prompt, think=False, stream=True)

    def summary(self) -> Dict[str, Any]:
        records = self.vector_db.get()
        num_chunks = len(records)

        file_to_ids: dict[str, list[str]] = {}
        unknown_ids: list[str] = []
        for rec in records:
            src = rec.metadata.get("file") if isinstance(rec.metadata, dict) else None
            if src:
                file_to_ids.setdefault(src, []).append(rec.id)
            else:
                unknown_ids.append(rec.id)

        distinct_docs = list(file_to_ids.keys())
        num_documents = len(distinct_docs)

        try:
            doc_files = [f for f in os.listdir(self.docs_path) if f.endswith(".txt")]
        except FileNotFoundError:
            doc_files = []

        disk_chunk_ids = {os.path.splitext(f)[0] for f in doc_files}
        db_chunk_ids = {rec.id for rec in records}
        orphan_on_disk = sorted(list(disk_chunk_ids - db_chunk_ids))
        missing_on_disk = sorted(list(db_chunk_ids - disk_chunk_ids))

        print("\n=== Knowledge RAG Summary ===")
        print(f"Device: {self.device}")
        print("Models:")
        print(f"  - OCR: {HUGGINGFACE_OCR_MODEL}")
        print(f"  - Embeddings: {OLLAMA_EMBED_MODEL}")
        print(f"  - LLM: {OLLAMA_LLM_MODEL}")
        print(f"  - Reranker: {RERANK_MODEL}")

        print("Vector DB:")
        print(f"  - Embedding dim: {self.vector_db.embedding_dim}")
        print(f"  - Metric: {self.vector_db.metric}")
        print(f"  - Storage path: {self.vector_db.storage_path}")
        print(f"  - Collection file: {self.vector_db.storage_path}/vectors.json")
        print(f"  - Total chunks: {num_chunks}")
        print(f"  - Distinct documents: {num_documents}")

        if hasattr(self, "chunker"):
            print("Chunker:")
            print(f"  - Type: {type(self.chunker).__name__}")
            if hasattr(self.chunker, "threshold"):
                print(f"  - Threshold: {self.chunker.threshold}")
            if hasattr(self.chunker, "max_chunk_size"):
                print(f"  - Max chunk size: {self.chunker.max_chunk_size}")

        if num_documents:
            print("Documents (up to 10 shown):")
            for i, (file_path, ids) in enumerate(sorted(file_to_ids.items())[:10], 1):
                print(f"  {i}. {file_path} -> {len(ids)} chunks")
            remaining = num_documents - min(10, num_documents)
            if remaining > 0:
                print(f"  ... and {remaining} more")
        else:
            print("Documents: none imported yet")

        print("Consistency checks:")
        print(f"  - Chunk files directory: {self.docs_path}")
        print(f"  - Chunk files on disk: {len(doc_files)}")
        if orphan_on_disk:
            print(
                f"  - Orphan chunk files on disk (no DB entry): {len(orphan_on_disk)}"
            )
        if missing_on_disk:
            print(
                f"  - Missing chunk files (in DB, not on disk): {len(missing_on_disk)}"
            )

        return {
            "device": self.device,
            "models": {
                "ocr": HUGGINGFACE_OCR_MODEL,
                "embedding": OLLAMA_EMBED_MODEL,
                "llm": OLLAMA_LLM_MODEL,
                "reranker": RERANK_MODEL,
            },
            "vector_db": {
                "embedding_dim": self.vector_db.embedding_dim,
                "metric": self.vector_db.metric,
                "storage_path": self.vector_db.storage_path,
                "collection_file": f"{self.vector_db.storage_path}/vectors.json",
                "num_chunks": num_chunks,
                "num_documents": num_documents,
            },
            "chunker": {
                "type": type(self.chunker).__name__,
                "threshold": getattr(self.chunker, "threshold", None),
                "max_chunk_size": getattr(self.chunker, "max_chunk_size", None),
            },
            "documents": {
                "by_file": {k: len(v) for k, v in file_to_ids.items()},
                "unknown_chunk_ids": unknown_ids,
            },
            "files": {
                "docs_path": self.docs_path,
                "chunks_on_disk": len(doc_files),
                "orphan_chunk_files": orphan_on_disk,
                "missing_chunk_files": missing_on_disk,
            },
        }
