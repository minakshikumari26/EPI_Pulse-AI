"""
EpiPulse AI — RAG Retriever
Combines semantic retrieval with live disease data context.
Builds rich prompts with citations for the LLM.
"""

import json
import re
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


RELEVANCE_THRESHOLD = 0.20   # minimum cosine similarity to include a chunk
MAX_CONTEXT_CHARS   = 3000   # max chars of retrieved text in prompt


class RAGRetriever:
    """
    Retrieves relevant knowledge chunks and builds LLM prompts with citations.
    """

    def __init__(self, knowledge_base=None):
        if knowledge_base is None:
            from src.rag.knowledge_base import get_knowledge_base
            self.kb = get_knowledge_base()
        else:
            self.kb = knowledge_base

    def retrieve(self, query: str, top_k: int = 4) -> list[dict]:
        """Retrieve top-k relevant chunks for a query."""
        return self.kb.retrieve(query, top_k=top_k)

    def build_rag_prompt(
        self,
        question: str,
        live_context: dict,
        top_k: int = 4,
        conversation_history: list = None,
    ) -> tuple[str, list[dict]]:
        """
        Build a RAG-enhanced prompt combining:
        - Retrieved knowledge chunks (WHO guidelines, IDSP reports, etc.)
        - Live disease data context (cases, risk scores, trends)
        - Conversation history (last N turns for memory)

        Returns:
            (prompt_string, retrieved_chunks_list)
        """
        # 1. Retrieve relevant chunks
        chunks = self.kb.retrieve(question, top_k=top_k)

        # 2. Build knowledge section
        knowledge_text = ""
        if chunks:
            knowledge_parts = []
            total_chars     = 0
            for i, chunk in enumerate(chunks):
                if total_chars >= MAX_CONTEXT_CHARS:
                    break
                part = f"[{i+1}] Source: {chunk['source']} (relevance: {chunk['score']:.2f})\n{chunk['text']}"
                knowledge_parts.append(part)
                total_chars += len(part)
            knowledge_text = "\n\n".join(knowledge_parts)

        # 3. Build live data section
        live_text = _format_live_context(live_context)

        # 4. Build conversation history section
        history_text = ""
        if conversation_history:
            recent = conversation_history[-6:]  # last 3 turns
            history_parts = []
            for msg in recent:
                role = "User" if msg["role"] == "user" else "Assistant"
                content = msg["content"][:300] + "..." if len(msg["content"]) > 300 else msg["content"]
                history_parts.append(f"{role}: {content}")
            history_text = "\n".join(history_parts)

        # 5. Assemble the full prompt
        prompt = _build_prompt(
            question=question,
            knowledge_text=knowledge_text,
            live_text=live_text,
            history_text=history_text,
            has_knowledge=bool(chunks),
        )

        return prompt, chunks

    def answer_with_rag(
        self,
        question: str,
        live_context: dict,
        llm_client,
        top_k: int = 4,
        temperature: float = 0.7,
        conversation_history: list = None,
    ) -> tuple[str, list[dict]]:
        """
        Full RAG pipeline: retrieve → build prompt → generate → return answer + citations.

        Returns:
            (answer_text, retrieved_chunks)
        """
        # Handle greetings without RAG
        greetings = {"hi", "hii", "hello", "hey", "helo", "howdy", "sup", "yo", "good morning", "good evening"}
        if question.strip().lower() in greetings:
            return (
                "👋 Hello! I'm your AI epidemiologist, powered by RAG + Groq.\n\n"
                "I can answer questions grounded in **WHO guidelines**, **IDSP surveillance protocols**, "
                "and **live outbreak data**.\n\n"
                f"Currently monitoring: **{live_context.get('region', 'all regions')}** — "
                f"**{live_context.get('disease', 'all diseases')}**\n\n"
                "Ask me anything about cases, risk levels, interventions, or outbreak patterns!",
                [],
            )

        prompt, chunks = self.build_rag_prompt(
            question=question,
            live_context=live_context,
            top_k=top_k,
            conversation_history=conversation_history,
        )

        answer = llm_client.generate(prompt, temperature=temperature)
        return answer, chunks

    def format_citations(self, chunks: list[dict]) -> str:
        """Format retrieved chunks as citation footnotes for display."""
        if not chunks:
            return ""
        lines = ["\n\n---\n📚 **Sources used:**"]
        seen  = set()
        for i, chunk in enumerate(chunks):
            src = chunk["source"]
            if src not in seen:
                seen.add(src)
                lines.append(f"- [{i+1}] {src} *(relevance: {chunk['score']:.0%})*")
        return "\n".join(lines)

    def get_kb_stats(self) -> dict:
        """Return knowledge base statistics for display."""
        return {
            "total_chunks": self.kb.count(),
            "sources":      self.kb.list_sources(),
        }


# ── Prompt builders ───────────────────────────────────────────────────────────

def _format_live_context(ctx: dict) -> str:
    """Convert live context dict into a readable string for the prompt."""
    if not ctx or ctx.get("note") == "No data for this selection":
        return "No live data available for the selected filters."

    parts = []
    if ctx.get("region"):
        parts.append(f"Region: {ctx['region']}")
    if ctx.get("disease"):
        parts.append(f"Disease: {ctx['disease']}")
    if ctx.get("year_filter"):
        parts.append(f"Year filter: {ctx['year_filter']}")
    if ctx.get("latest_date"):
        parts.append(f"Latest data date: {ctx['latest_date']}")
    if ctx.get("latest_cases") is not None:
        parts.append(f"Latest case count: {ctx['latest_cases']:,}")
    if ctx.get("risk_level"):
        parts.append(f"Current risk level: {ctx['risk_level']} (score: {ctx.get('risk_score', 'N/A')})")
    if ctx.get("z_score") is not None:
        parts.append(f"Z-score: {ctx['z_score']:.2f} {'⚠️ SPIKE' if ctx.get('is_spike') else ''}")
    if ctx.get("trend"):
        parts.append(f"Trend: {ctx['trend']} ({ctx.get('growth_rate_pct', 0):+.1f}% vs prev week)")
    if ctx.get("avg_7d"):
        parts.append(f"7-day avg: {ctx['avg_7d']:.1f} cases | Prev 7-day avg: {ctx.get('avg_prev_7d', 'N/A')}")
    if ctx.get("rank_of_total"):
        parts.append(f"Regional rank: {ctx['rank_of_total']} (1 = highest burden)")
    if ctx.get("vs_national_avg_pct") is not None:
        parts.append(f"vs National average: {ctx['vs_national_avg_pct']:+.1f}% (national avg: {ctx.get('national_avg_cases', 'N/A')} cases)")
    if ctx.get("top_3_regions"):
        parts.append(f"Top 3 highest-burden regions: {', '.join(ctx['top_3_regions'])}")
    if ctx.get("spike_days"):
        parts.append(f"Spike days in period: {ctx['spike_days']}")
    if ctx.get("last_7_days_cases"):
        daily = ctx["last_7_days_cases"]
        dates = ctx.get("last_7_days_dates", [])
        if dates:
            trend_str = ", ".join(f"{d}: {c}" for d, c in zip(dates[-4:], daily[-4:]))
            parts.append(f"Recent 4-day trend: {trend_str}")
    if ctx.get("avg_temp"):
        parts.append(f"Avg temperature: {ctx['avg_temp']}°C | Avg humidity: {ctx.get('avg_humidity', 'N/A')}%")
    if ctx.get("available_regions"):
        regions = ctx["available_regions"]
        parts.append(f"All monitored regions ({len(regions)}): {', '.join(regions[:8])}{'...' if len(regions) > 8 else ''}")

    return "\n".join(parts)


def _build_prompt(
    question: str,
    knowledge_text: str,
    live_text: str,
    history_text: str,
    has_knowledge: bool,
) -> str:
    """Assemble the full RAG prompt."""

    knowledge_section = (
        f"""## Knowledge Base (WHO Guidelines / IDSP Protocols)
{knowledge_text}"""
        if has_knowledge else
        "## Knowledge Base\nNo relevant documents found — answering from general epidemiology knowledge."
    )

    history_section = (
        f"""## Conversation History (for context)
{history_text}

"""
        if history_text else ""
    )

    return f"""You are an expert AI epidemiologist assistant for EpiPulse AI, an outbreak intelligence platform monitoring Indian states.

You have access to:
1. A curated knowledge base of WHO guidelines, IDSP surveillance protocols, and epidemiology frameworks
2. Live real-time disease surveillance data for Indian regions
3. Conversation history for context continuity

Your goal: Give specific, data-driven, actionable answers. Always cite which source informed your recommendation when relevant. Use actual numbers from the live data.

{knowledge_section}

## Live Surveillance Data
{live_text}

{history_section}## User Question
{question}

## Instructions
- Use BOTH the knowledge base AND live data in your answer
- Be specific — use actual case numbers, z-scores, risk scores from the live data
- When recommending interventions, cite the WHO/IDSP guideline that supports it
- If the knowledge base has directly relevant information, reference it (e.g., "According to WHO Dengue Guidelines...")
- Keep the answer focused and actionable for a public health professional
- If this is a casual question unrelated to epidemiology, answer briefly and guide back to outbreak analysis

Answer:"""
