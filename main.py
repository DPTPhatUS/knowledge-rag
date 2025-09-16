from database import (
    initialize_vector_database,
    store_embeddings,
    list_documents,
    delete_document,
)
from text_processing import (
    extract_text_from_file,
    chunk_text,
    embed_chunks,
    embed_query,
)
from rag_operations import generate_answer_with_rag, search_vector_db
from vector_db import VectorDB
import uuid
from globals import COLLECTION_NAME
import os


def main():
    # storage_path = f"collections/{COLLECTION_NAME}"
    storage_path = f"collections/text"

    # vector_db = initialize_vector_database()
    vector_db = VectorDB(embedding_dim=768, storage_path=storage_path, auto_save=True)

    docs_path = f"{storage_path}/docs"
    os.makedirs(docs_path, exist_ok=True)

    while True:
        print("\n--- Knowledge RAG ---")
        print("1. Import Document")
        print("2. Ask a Question")
        print("3. List Documents")
        print("4. Delete Document")
        print("5. Exit")

        choice = input("Choose an option: ")

        if choice == "1":
            input_file_path = input("Enter file path: ")
            text = extract_text_from_file(input_file_path)
            chunks = chunk_text(text)
            embeddings = embed_chunks(chunks)
            # store_embeddings(vector_db, embeddings, chunks, file_path)

            chunk_ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
            chunk_metadatas = [{"file": input_file_path}] * len(chunks)

            vector_db.add_batch(embeddings, chunk_ids, chunk_metadatas)

            for id, chunk in zip(chunk_ids, chunks):
                chunk_file_path = f"{docs_path}/{id}.txt"
                with open(chunk_file_path, "w") as chunk_file:
                    chunk_file.write(chunk)

        elif choice == "2":
            query = input("Enter your question: ")
            query_embedding = embed_query(query)
            # results = search_vector_db(vector_db, query_embedding)

            search_results = vector_db.search(query=query_embedding, top_k=5)
            result_ids = [id for id, _ in search_results]
            
            def get_doc_from_id(doc_id: str) -> str:
                doc_file_path = f"{docs_path}/{doc_id}.txt"
                with open(doc_file_path, "r") as doc_file:
                    return doc_file.read()
                    
            results = [get_doc_from_id(doc_id) for doc_id in result_ids]
            print("\nRetrieved documents: ", results)

            answer_stream = generate_answer_with_rag(query, results)
            print("\nAnswer: ", end="")
            for chunk in answer_stream:
                print(chunk.response, end="", flush=True)
            print()

        elif choice == "3":
            # list_documents(vector_db)
            print(len(vector_db))

        elif choice == "4":
            doc_id = input("Enter document ID to delete: ")
            # delete_document(vector_db, doc_id)
            
            vector_db.delete(doc_id)
            os.remove(f"{docs_path}/{doc_id}.txt")

        elif choice == "5":
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
