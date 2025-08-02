def add_documents(collection):
    collection.add(
        ids=["id1", "id2"],
        documents=[
            "This is a document about pineapple",
            "This is a document about oranges"
        ]
    )

def query_documents(collection, query_texts, n_results=2):
    return collection.query(
        query_texts=query_texts,
        n_results=n_results,
        include=["embeddings", "metadatas", "documents", "distances"]
    )
