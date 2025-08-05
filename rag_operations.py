from chromadb.api import ClientAPI
import ollama


prompt = """
You are a world knowledge expert. Given the information below, answer the question clearly and naturally. Use the context to inform your response, but do not refer to the context explicitly or repeat it unnecessarily.

Context:
{context}

Question:
{question}

Answer:
"""


def generate_answer_with_rag(query, chunks):
    print("Generating answer using RAG...")

    context = "\n".join(chunks)
    formatted_prompt = prompt.format(context=context, question=query)
    response_stream = ollama.generate(
        model="qwen3:4b", prompt=formatted_prompt, think=False, stream=True
    )
    return response_stream


def search_vector_db(db: ClientAPI, query_embedding):
    print("Searching vector DB...")

    collection = db.get_or_create_collection(name="knowledge_rag")
    results = collection.query(query_embeddings=query_embedding)
    return results["documents"][0]
