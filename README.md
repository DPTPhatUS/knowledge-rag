# Knowledge RAG

Local OCR + Retrieval-Augmented Generation (RAG) pipeline that lets you ingest PDF/TXT documents, chunk them semantically, embed with Ollama, and answer questions using a local LLM — all stored in a lightweight JSON vector store.

---

## Overview

- OCRs PDFs using Meta's Nougat (`facebook/nougat-small`) to extract Markdown-like text.
- Splits text into semantically coherent chunks, preserving inline math and LaTeX blocks.
- Generates embeddings with an Ollama embedding model (default: `nomic-embed-text:latest`).
- Stores vectors in a simple on-disk JSON database with in-memory search.
- Answers questions with an Ollama LLM (default: `gemma3:4b`) using a RAG prompt.

---

## Features

- OCR PDF pages to text via Nougat; TXT files are read directly.
- LaTeX/math-aware sentence splitting and semantic chunking with cosine thresholding.
- Embedding via Ollama (configurable model).
- Simple vector DB with cosine/inner/euclid search and JSON persistence.
- Streaming answers from the LLM via Ollama.

---

## Requirements

- Python 3.10+
- System dependencies:
	- Poppler (for `pdf2image` PDF → image):
		- Ubuntu/Debian: `sudo apt-get update && sudo apt-get install -y poppler-utils`
	- Ollama installed and running locally: https://ollama.com

- Python packages (install suggestions):

```bash
pip install -U pip
pip install numpy torch transformers pdf2image pillow chromadb ollama
```

Notes:
- Torch can leverage CUDA/MPS if available; the code auto-selects `cuda` → `mps` → `cpu`.
- First use of each Ollama model will trigger a pull; you can pre-pull:

```bash
ollama pull nomic-embed-text:latest
ollama pull gemma3:4b
```

---

## Demo (CLI)

```bash
python main.py
```

Menu:
1) Import Document — provide a path to a `.pdf` or `.txt` file
2) Ask a Question — type your question; answers stream to the console
3) List Documents — [planned] not implemented yet
4) Delete Document — delete by chunk/document id (see Data Layout)
5) Exit

Notes:
- PDF ingestion can take time depending on page count and OCR speed.
- Keep the Ollama models pulled beforehand to avoid delays on first run.

---

## Programmatic Usage

```python
from knowledge_rag import KnowledgeRAG

rag = KnowledgeRAG(storage_path="collections/text")

# Add a document (PDF or TXT)
rag.add_document("/path/to/file.pdf")

# Ask a question (streamed response)
stream = rag.answer_question("What are the main theorems?")
print("Answer:", end=" ")
for chunk in stream:
    print(chunk.response, end="", flush=True)
print()
```

---

## Configuration

Edit `globals.py` to customize runtime defaults:

- `COLLECTION_NAME`: default collection name (used as default path in vector DB)
- `OLLAMA_EMBED_MODEL`: embedding model name (Ollama), default `nomic-embed-text:latest`
- `OLLAMA_LLM_MODEL`: LLM name (Ollama), default `gemma3:4b`
- `BASE_PROMPT`: prompt template used for RAG

Constructor options in `KnowledgeRAG` (`knowledge_rag.py`):
- `storage_path` (default: `collections/text`): where vectors/docs are stored
- `device`: auto-selected among `cuda`, `mps`, `cpu`

Semantic chunking (`text_chunker.py`):
- `threshold` (default 0.4): cosine similarity threshold to merge adjacent sentences
- `max_chunk_length` (default 500): approximate max characters per chunk

---

## Data Layout

When using the default `storage_path="collections/text"`:

```
collections/
	text/
		vectors.json        # persisted vector DB (ids, vectors, metadata)
		docs/               # plain-text chunk files
			<chunk-id>.txt
```

Metadata stored per chunk currently includes the source file path under the `file` key.

---

## How It Works

1) Ingest
	 - PDFs → images via `pdf2image` (Poppler), then OCR with Nougat (`facebook/nougat-small`).
	 - TXTs → read directly.
2) Chunk
	 - Sentence-level splitting preserving LaTeX/math blocks.
	 - Merge adjacent sentences if cosine similarity of embeddings exceeds `threshold`.
3) Embed
	 - Use Ollama embedding model on each chunk; store vectors + metadata.
4) Retrieve
	 - Cosine (default), inner product, or euclidean similarity search.
5) Generate
	 - Top-k chunks are joined as context and passed to the LLM with `BASE_PROMPT`.

---

## Troubleshooting

- Poppler missing: `pdf2image` raises errors when converting PDFs.
	- Install with `sudo apt-get install -y poppler-utils` (Linux). For other OSes, see `pdf2image` docs.
- Ollama not running or model missing:
	- Install Ollama, then `ollama pull nomic-embed-text:latest` and `ollama pull gemma3:4b`.

---

## License

See `LICENSE`.
