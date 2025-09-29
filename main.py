from knowledge_rag import KnowledgeRAG


def main():
    knowledge_rag = KnowledgeRAG()

    while True:
        print("\n--- Knowledge RAG ---")
        print("1. Import Document")
        print("2. Ask a Question")
        print("3. System Overview")
        print("4. Delete Document")
        print("5. Exit")

        choice = input("Choose an option: ")

        if choice == "1":
            input_file_path = input("Enter file path: ")
            knowledge_rag.add_document(input_file_path)

        elif choice == "2":
            query = input("Enter your question: ")
            answer = knowledge_rag.answer_question(query=query)
            print("\nAnswer: ", answer)

        elif choice == "3":
            knowledge_rag.summary()

        elif choice == "4":
            delete_file_path = input("Enter file path: ")
            knowledge_rag.delete_document(delete_file_path)

        elif choice == "5":
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
