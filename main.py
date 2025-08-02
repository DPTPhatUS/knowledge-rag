import os
from config.settings import get_langchain_api_key, LANGCHAIN_TRACING_V2, LANGCHAIN_ENDPOINT
from services.chroma_client import get_chroma_client
from services.embedding import get_embedding_function
from data.documents import add_documents, query_documents

# Set environment variables
os.environ['LANGCHAIN_TRACING_V2'] = LANGCHAIN_TRACING_V2
os.environ['LANGCHAIN_ENDPOINT'] = LANGCHAIN_ENDPOINT
os.environ['LANGCHAIN_API_KEY'] = get_langchain_api_key()

# Initialize ChromaDB client and collection
chroma_client = get_chroma_client()
collection = chroma_client.create_collection(
    name="my_collection",
    embedding_function=get_embedding_function()
)

# Add documents to the collection
add_documents(collection)

# Query the collection
results = query_documents(collection, query_texts=["This is a document about pineapple"])
print(results)