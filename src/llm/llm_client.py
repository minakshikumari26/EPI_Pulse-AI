"""Unified LLM client for EpiPulse AI — supports Groq and Ollama."""

import json
import os
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config


# ──────────────────────────────────────────────
# Base class — all providers implement this
# ──────────────────────────────────────────────

class BaseLLMClient:
    """Common interface for all LLM providers."""

    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        raise NotImplementedError

    def answer_question(self, question: str, context: dict = None) -> str:
    # Detect casual/greeting messages and respond naturally
        greetings = ["hi", "hii", "hello", "hey", "helo", "howdy", "sup", "yo"]
        if question.strip().lower() in greetings:
            return (
                "👋 Hello! I'm your AI epidemiologist assistant.\n\n"
                "Ask me anything about the disease outbreak data — "
                "current cases, risk levels, regional trends, or intervention strategies!"
            )

        context_str = (
            f"\nContext (JSON):\n{json.dumps(context, indent=2, default=str)}"
            if context else ""
        )

        # Check if question is off-topic (not about disease/health/data)
        prompt = (
            f"You are a friendly AI epidemiologist assistant for EpiPulse AI.\n"
            f"Answer the user's question naturally. If it's a general question not related "
            f"to disease outbreaks, answer it briefly and politely, then gently guide them "
            f"back to what you can help with (outbreak analysis).\n"
            f"If it IS about disease/health/outbreak data, give a detailed analytical response "
            f"using the context provided.\n\n"
            f"Question: {question}{context_str}\n\n"
            f"Respond naturally and helpfully."
        )
        return self.generate(prompt, temperature=0.7)

    def explain_alert(
        self,
        cases: Optional[float] = None,
        risk_score: Optional[float] = None,
        z_score: Optional[float] = None,
        growth_rate: Optional[float] = None,
        region: str = "Unknown",
    ) -> str:
        metrics = []
        if cases is not None:
            metrics.append(f"cases: {cases:.0f}")
        if risk_score is not None:
            metrics.append(f"risk score: {risk_score:.1f}/100")
        if z_score is not None:
            metrics.append(f"z-score: {z_score:.2f}")
        if growth_rate is not None:
            metrics.append(f"growth rate: {growth_rate:.1%}")
        metrics_str = ", ".join(metrics) or "no data"

        prompt = (
            f"You are a public health epidemiologist. Analyze this disease outbreak "
            f"alert data and provide a concise, professional explanation (1-2 sentences max).\n\n"
            f"Region: {region}\nMetrics: {metrics_str}\n\n"
            f"Provide a brief, actionable explanation of what this data suggests "
            f"about the outbreak situation. Be specific about risk level and any immediate concerns."
        )
        return self.generate(prompt, temperature=0.5)

    def summarize_forecast(self, forecast_data: dict, region: str = "Unknown") -> str:
        prompt = (
            f"You are a public health analyst. Summarize this disease forecast in 2-3 sentences.\n\n"
            f"Region: {region}\nForecast Data: {json.dumps(forecast_data, default=str)}\n\n"
            f"Be concise and focus on trends, turning points, and actionable insights."
        )
        return self.generate(prompt, temperature=0.5)

    def generate_report(self, summary_data: dict) -> str:
        prompt = (
            f"Generate a brief executive summary report for public health officials "
            f"(3-4 short paragraphs).\n\n"
            f"Data Summary:\n{json.dumps(summary_data, indent=2, default=str)}\n\n"
            f"Structure:\n"
            f"1. Current situation (cases, hotspots)\n"
            f"2. Risk assessment\n"
            f"3. Forecast outlook\n"
            f"4. Recommended actions\n\n"
            f"Be professional, data-driven, and actionable."
        )
        return self.generate(prompt, temperature=0.6)


# ──────────────────────────────────────────────
# Groq Client
# ──────────────────────────────────────────────

class GroqClient(BaseLLMClient):
    """LLM client using Groq's free cloud API (fast, no local GPU needed)."""

    SUPPORTED_MODELS = [
        "llama3-8b-8192",
        "llama3-70b-8192",
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ]

    def __init__(self, api_key: str, model: str = "llama3-8b-8192", timeout: int = 60):
        try:
            from groq import Groq
        except ImportError:
            raise ImportError(
                "groq package not installed. Run: pip install groq"
            )

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Get a free key at https://console.groq.com and add it to your .env file."
            )

        self.client = Groq(api_key=api_key)
        self.model = model
        self.timeout = timeout
        self.base_url = "https://api.groq.com"
        self.available_models = self.SUPPORTED_MODELS
        self.fallback_used = False
        self.fallback_model = None

        # Validate model choice
        if model not in self.SUPPORTED_MODELS:
            print(
                f"[GroqClient] Warning: '{model}' not in known model list. "
                f"Proceeding anyway — Groq may still accept it."
            )

    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate text via Groq API."""
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=temperature,
            )
            return chat_completion.choices[0].message.content.strip()

        except Exception as e:
            err = str(e).lower()

            # Model not found — try fallback
            if "model" in err or "not found" in err:
                fallback = "llama3-8b-8192"
                if fallback != self.model:
                    try:
                        chat_completion = self.client.chat.completions.create(
                            messages=[{"role": "user", "content": prompt}],
                            model=fallback,
                            temperature=temperature,
                        )
                        self.fallback_used = True
                        self.fallback_model = fallback
                        self.model = fallback
                        return chat_completion.choices[0].message.content.strip()
                    except Exception:
                        pass

            # Rate limit
            if "rate" in err or "429" in err:
                raise RuntimeError(
                    "Groq rate limit reached. Wait a moment and try again, "
                    "or upgrade your Groq plan at https://console.groq.com"
                )

            # Auth
            if "auth" in err or "401" in err or "api key" in err:
                raise RuntimeError(
                    "Invalid GROQ_API_KEY. "
                    "Check your key at https://console.groq.com/keys"
                )

            raise RuntimeError(f"Groq generation failed: {e}")


# ──────────────────────────────────────────────
# Ollama Client (kept as local fallback)
# ──────────────────────────────────────────────

class OllamaClient(BaseLLMClient):
    """LLM client using local Ollama server (no internet, no API key needed)."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",   # ← fixed port (was 11435)
        model: str = "llama3.2:1b",
        timeout: int = 60,
    ):
        try:
            import requests as _requests
            self._requests = _requests
        except ImportError:
            raise ImportError("requests package not installed. Run: pip install requests")

        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.available_models: list[str] = []
        self.fallback_used = False
        self.fallback_model: str | None = None
        self._verify_connection()

    def _fetch_available_models(self) -> list[str]:
        try:
            response = self._requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            models = response.json().get("models", []) or []
            return [m.get("name") if isinstance(m, dict) else str(m) for m in models]
        except self._requests.exceptions.RequestException as e:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running: `ollama serve`. "
                f"Details: {e}"
            )

    def _verify_connection(self):
        self.available_models = self._fetch_available_models()
        if not self.available_models:
            raise ConnectionError(
                f"Ollama is running but no models are loaded. "
                "Run `ollama pull llama3.2:1b` then restart."
            )
        if self.model not in self.available_models:
            fallback = next(
                (m for m in ["llama3.2:1b", "llama3.2:1b:latest", "mistral", "mistral:latest"]
                 if m in self.available_models),
                self.available_models[0],
            )
            self.model = fallback
            self.fallback_used = True
            self.fallback_model = fallback

    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        try:
            response = self._requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "temperature": temperature,
                    "stream": False,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()["response"].strip()
        except self._requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama generation failed: {e}")


# ──────────────────────────────────────────────
# Factory — reads config and returns right client
# ──────────────────────────────────────────────

def get_llm_client(force_new: bool = False) -> BaseLLMClient:
    """
    Get or create LLM client based on config.yaml provider setting.

    Providers:
        groq   — free cloud API, fast, no GPU needed (recommended)
        ollama — local, free, requires `ollama serve` running

    Returns:
        BaseLLMClient instance (GroqClient or OllamaClient)
    """
    if not hasattr(get_llm_client, "_instance") or force_new:
        config = load_config().get("llm", {})
        provider = os.getenv("LLM_PROVIDER") or config.get("provider", "groq")

        if provider == "groq":
            api_key = os.getenv("GROQ_API_KEY") or config.get("groq_api_key", "")
            model = config.get("model", "llama3-8b-8192")
            timeout = config.get("timeout", 60)
            get_llm_client._instance = GroqClient(
                api_key=api_key, model=model, timeout=timeout
            )

        elif provider == "ollama":
            base_url = config.get("ollama_url", "http://localhost:11434")
            model = config.get("model", "llama3.2:1b")
            timeout = config.get("timeout", 60)
            get_llm_client._instance = OllamaClient(
                base_url=base_url, model=model, timeout=timeout
            )

        else:
            raise ValueError(
                f"Unknown LLM provider: '{provider}'. "
                "Set LLM_PROVIDER=groq or LLM_PROVIDER=ollama in your .env file."
            )

    return get_llm_client._instance
