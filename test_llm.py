from src.llm.ollama_client import get_llm_client

try:
    client = get_llm_client()
    print('LLM client initialized successfully')
    response = client.generate('Hello, test message')
    print('Response:', response)
except Exception as e:
    print('Error:', e)