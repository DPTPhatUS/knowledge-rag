from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

def get_embedding_function():
    return OllamaEmbeddingFunction(model_name='nomic-embed-text:latest')
