import chromadb
from chromadb.config import Settings
from config.settings import LANGCHAIN_TRACING_V2

def get_chroma_client():
    return chromadb.Client(settings=Settings(anonymized_telemetry=False))
