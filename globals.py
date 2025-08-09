COLLECTION_NAME: str = "knowledge_rag"

OLLAMA_EMBED_MODEL: str = "nomic-embed-text:latest"
OLLAMA_LLM_MODEL: str = "qwen3:4b"

BASE_PROMPT: str = """You are a world knowledge expert. Given the information below, answer the question clearly and naturally. Use the context to inform your response, but do not refer to the context explicitly or repeat it unnecessarily.",

Context:
{context}

Question:
{question}

Answer:"""
