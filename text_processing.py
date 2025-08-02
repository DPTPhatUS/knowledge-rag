def extract_text_from_file(file_path):
    print(f"Extracting text from {file_path}...")
    return "Sample text content from file."

def chunk_text(text):
    print("Chunking text...")
    return ["Chunk 1 of text", "Chunk 2 of text"]

def embed_chunks(chunks):
    print("Generating embeddings...")
    return [[0.1, 0.2], [0.3, 0.4]]  # mock embeddings

def embed_query(query):
    print("Embedding user query...")
    return [0.5, 0.6]
