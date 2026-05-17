"""RAG module for EpiPulse AI — semantic retrieval over disease knowledge base."""

from .knowledge_base import KnowledgeBase, get_knowledge_base
from .retriever import RAGRetriever

__all__ = ["KnowledgeBase", "get_knowledge_base", "RAGRetriever"]
