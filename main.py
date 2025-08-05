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


def main():
    vector_db = initialize_vector_database()

    while True:
        print("\n--- Knowledge RAG ---")
        print("1. Import Document")
        print("2. Ask a Question")
        print("3. List Documents")
        print("4. Delete Document")
        print("5. Exit")

        choice = input("Choose an option: ")

        if choice == "1":
            file_path = input("Enter file path: ")
            text = extract_text_from_file(file_path)
            chunks = chunk_text(text)
            embeddings = embed_chunks(chunks)
            store_embeddings(vector_db, embeddings, chunks, file_path)

        elif choice == "2":
            query = input("Enter your question: ")
            query_embedding = embed_query(query)
            results = search_vector_db(vector_db, query_embedding)
            answer_stream = generate_answer_with_rag(query, results)
            print(f"\nAnswer: ", end="")
            for chunk in answer_stream:
                print(chunk.response, end="", flush=True)
            print()

        elif choice == "3":
            list_documents(vector_db)

        elif choice == "4":
            doc_id = input("Enter document ID to delete: ")
            delete_document(vector_db, doc_id)

        elif choice == "5":
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
