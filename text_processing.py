from text_chunker import SemanticChunker
from typing import List
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
import os
from transformers import NougatProcessor, VisionEncoderDecoderModel
import torch
from pdf2image import convert_from_path
from globals import OLLAMA_EMBED_MODEL


_nougat_processor = None
_nougat_model = None
_nougat_device = None


def _get_nougat_components():
    global _nougat_processor, _nougat_model, _nougat_device
    if _nougat_processor is None or _nougat_model is None or _nougat_device is None:
        _nougat_processor = NougatProcessor.from_pretrained("facebook/nougat-small")
        _nougat_model = VisionEncoderDecoderModel.from_pretrained(
            "facebook/nougat-small"
        )
        _nougat_device = "cuda" if torch.cuda.is_available() else "cpu"
        _nougat_model.to(_nougat_device)
    return _nougat_processor, _nougat_model, _nougat_device


def _extract_text_from_pdf(file_path: str) -> str:
    processor, model, device = _get_nougat_components()

    pages = convert_from_path(file_path, fmt="jpeg")

    results = []
    for page_img in pages:
        rgb_img = page_img.convert("RGB")
        pixel_values = processor(rgb_img, return_tensors="pt").pixel_values
        outputs = model.generate(
            pixel_values.to(device),
            max_new_tokens=2048,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
        )
        seq = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        seq = processor.post_process_generation(seq, fix_markdown=False)
        results.append(seq.strip())

    return "\n\n".join(results)


def extract_text_from_file(file_path: str) -> str:
    _, ext = os.path.splitext(file_path.lower())
    if ext == ".pdf":
        return _extract_text_from_pdf(file_path)
    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        return file.read()


_embed_model: OllamaEmbeddingFunction | None = None


def _get_embed_model() -> OllamaEmbeddingFunction:
    global _embed_model
    if _embed_model is None:
        _embed_model = OllamaEmbeddingFunction(model_name=OLLAMA_EMBED_MODEL)
    return _embed_model


text_chunker = SemanticChunker(embed_func=_get_embed_model())


def chunk_text(text: str) -> List[str]:
    return text_chunker(document=text)


def embed_chunks(chunks: List[str]):
    return _get_embed_model()(chunks)


def embed_query(query: str):
    embeddings = _get_embed_model()([query])
    return embeddings[0]
