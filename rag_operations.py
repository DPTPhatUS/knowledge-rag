from chromadb.api import ClientAPI


def generate_answer_with_rag(query, chunks):
    print("Generating answer using RAG...")
    context = "\n".join(chunks)
    return f"Based on your documents, here's a summary related to: {query}"


def search_vector_db(db: ClientAPI, query_embedding):
    print("Searching vector DB...")

    collection = db.get_or_create_collection(name="knowledge_rag")
    results = collection.query(query_embeddings=query_embedding)
    return results
