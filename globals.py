COLLECTION_NAME: str = "knowledge_rag"

HUGGINGFACE_OCR_MODEL: str = "facebook/nougat-small"

OLLAMA_EMBED_MODEL: str = "nomic-embed-text:latest"
OLLAMA_LLM_MODEL: str = "gemma3:4b"

RERANK_MODEL: str = "BAAI/bge-reranker-v2-m3"

BASE_PROMPT: str = """You are a world knowledge expert. 
Given the information below, answer the question clearly and naturally. 
Use the context to inform your response, but do not refer to the context explicitly or repeat it unnecessarily.

Context:
{context}

Question:
{question}

Answer:"""

MULTI_QUERY_PROMPT: str = """You are an AI language model assistant. 
Your task is to generate five different versions of the given user question to retrieve relevant documents from a vector database. 
By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search. 
You must only respond by providing these alternative questions separated by newlines. 

Original question: {question}"""
