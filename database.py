import uuid
from typing import Iterable, List
import chromadb
from chromadb.api import ClientAPI
from chromadb.config import Settings
from globals import COLLECTION_NAME


def initialize_vector_database() -> ClientAPI:
    chroma_client = chromadb.Client(settings=Settings(anonymized_telemetry=True))
    chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    return chroma_client


def _generate_ids(n: int) -> List[str]:
    return [str(uuid.uuid4()) for _ in range(n)]


def store_embeddings(
    db: ClientAPI,
    embeddings: Iterable[Iterable[float]],
    chunks: List[str],
    file_path: str,
) -> None:
    print("Storing embeddings into vector DB...")

    collection = db.get_or_create_collection(name=COLLECTION_NAME)
    collection.add(
        ids=_generate_ids(len(chunks)),
        embeddings=embeddings,
        metadatas=[{"file": file_path}] * len(chunks),
        documents=chunks,
    )


def list_documents(db: ClientAPI) -> None:
    print("Listing documents in vector DB...")

    collection = db.get_or_create_collection(name=COLLECTION_NAME)
    results = collection.get()
    print(results)


def delete_document(db: ClientAPI, doc_id: str | list[str]) -> None:
    print(f"Deleting document with ID: {doc_id}")

    collection = db.get_or_create_collection(name=COLLECTION_NAME)
    collection.delete(ids=doc_id)
