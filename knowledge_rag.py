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
from typing import Iterable
from functools import partial


class KnowledgeRAG:
    def __init__(
        self,
        storage_path: str | None = "collections/text",
        device: str | None = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.mps.is_available() else "cpu"
        ),
    ) -> None:
        self.vector_db = VectorDB(
            embedding_dim=768, storage_path=storage_path, auto_save=True
        )

        self.docs_path = f"{storage_path}/docs"
        os.makedirs(self.docs_path, exist_ok=True)

        self.ocr_processor = NougatProcessor.from_pretrained("facebook/nougat-small")
        self.ocr_model = VisionEncoderDecoderModel.from_pretrained(
            "facebook/nougat-small"
        )
        self.ocr_model.to(device)
        self.device = device

        self.embedding_model = OllamaEmbeddingFunction(model_name=OLLAMA_EMBED_MODEL)
        self.llm_model = partial(
            ollama.generate, model=OLLAMA_LLM_MODEL, keep_alive=False
        )

        self.chunker = SemanticChunker(
            embed_func=self.embedding_model, threshold=0.4, max_chunk_length=500
        )

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
        chunk_metadatas = [{"file": file_path}] * len(chunks)
        embeddings = self.embedding_model(chunks)

        self.vector_db.add_batch(embeddings, chunk_ids, chunk_metadatas)

        for id, chunk in zip(chunk_ids, chunks):
            chunk_file_path = f"{self.docs_path}/{id}.txt"
            with open(chunk_file_path, "w") as chunk_file:
                chunk_file.write(chunk)

    def delete_document(self, document_id: str) -> None:
        self.vector_db.delete(id=document_id)
        os.remove(f"{self.docs_path}/{document_id}.txt")

    def answer_question(self, query: str) -> Iterable[ollama.GenerateResponse]:
        query_embedding = self.embedding_model([query])[0]

        search_results = self.vector_db.search(query=query_embedding, top_k=5)
        results_ids = [id for id, _ in search_results]

        def _get_doc_from_id(doc_id: str):
            doc_file_path = f"{self.docs_path}/{doc_id}.txt"
            with open(doc_file_path, "r") as doc_file:
                return doc_file.read()

        results_docs = [_get_doc_from_id(id) for id in results_ids]

        context = "\n".join(results_docs)
        prompt = BASE_PROMPT.format(context=context, question=query)

        return self.llm_model(prompt=prompt, think=False, stream=True)

    def summary(self):
        pass
