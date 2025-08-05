from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

def extract_text_from_file(file_path):
    with open(file_path, "r") as file:
        return file.read()


text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300, chunk_overlap=50
)


def chunk_text(text):
    chunks = text_splitter.split_text(text=text)
    return chunks


embed_model = OllamaEmbeddingFunction(model_name="nomic-embed-text:latest")


def embed_chunks(chunks):
    embeddings = embed_model(chunks)
    return embeddings


def embed_query(query):
    embeddings = embed_model([query])
    return embeddings[0]
