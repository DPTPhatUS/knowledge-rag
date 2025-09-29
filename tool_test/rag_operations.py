from typing import Iterable, List
from chromadb.api import ClientAPI
import ollama
from globals import BASE_PROMPT, COLLECTION_NAME, OLLAMA_LLM_MODEL


def generate_answer_with_rag(query: str, chunks: List[str]):
    print("\nGenerating answer using RAG...")

    context = "\n".join(chunks)
    formatted_prompt = BASE_PROMPT.format(context=context, question=query)
    response_stream = ollama.generate(
        model=OLLAMA_LLM_MODEL, prompt=formatted_prompt, think=False, stream=True
    )
    return response_stream


def search_vector_db(
    db: ClientAPI, query_embedding: Iterable[float], n_results: int | None = None
) -> List[str]:
    print("Searching vector DB...")

    collection = db.get_or_create_collection(name=COLLECTION_NAME)
    query_args = {"query_embeddings": query_embedding}
    if n_results is not None:
        query_args["n_results"] = n_results
    results = collection.query(**query_args)
    return results["documents"][0]
