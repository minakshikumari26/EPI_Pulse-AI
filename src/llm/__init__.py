"""LLM integration module for EpiPulse AI — supports Groq and Ollama."""

from .llm_client import BaseLLMClient, GroqClient, OllamaClient, get_llm_client

# Keep OllamaLLM alias so nothing else breaks
OllamaLLM = OllamaClient

__all__ = [
    "BaseLLMClient",
    "GroqClient",
    "OllamaClient",
    "OllamaLLM",        # backwards-compatible alias
    "get_llm_client",
]
