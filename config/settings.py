import os

# Environment variables
LANGCHAIN_TRACING_V2 = os.getenv('LANGCHAIN_TRACING_V2', 'true')
LANGCHAIN_ENDPOINT = os.getenv('LANGCHAIN_ENDPOINT', 'https://api.smith.langchain.com')
LANGCHAIN_API_KEY_PATH = os.path.expanduser('~/langchain-key.txt')

def get_langchain_api_key():
    with open(LANGCHAIN_API_KEY_PATH, 'r') as file:
        return file.readline().strip()
