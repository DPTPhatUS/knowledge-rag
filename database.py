import chromadb
from chromadb.api import ClientAPI
from chromadb.config import Settings
from chromadbx import UUIDGenerator


def initialize_vector_database():
    chroma_client = chromadb.Client(settings=Settings(anonymized_telemetry=True))
    chroma_client.get_or_create_collection(name="knowledge_rag")
    return chroma_client


def store_embeddings(db: ClientAPI, embeddings, chunks, file_path):
    print("Storing embeddings into vector DB...")

    collection = db.get_or_create_collection(name="knowledge_rag")
    collection.add(
        ids=UUIDGenerator(len(chunks)),
        embeddings=embeddings,
        metadatas=[{"file": file_path}] * len(chunks),
        documents=chunks,
    )


def list_documents(db: ClientAPI):
    print("Listing documents in vector DB...")

    collection = db.get_or_create_collection(name="knowledge_rag")
    results = collection.get()
    print(results)


def delete_document(db: ClientAPI, doc_id):
    print(f"Deleting document with ID: {doc_id}")

    collection = db.get_or_create_collection(name="knowledge_rag")
    collection.delete(ids=doc_id)
