"""LLM integration module for EpiPulse AI using Ollama."""

from .ollama_client import OllamaLLM, get_llm_client

__all__ = ["OllamaLLM", "get_llm_client"]
