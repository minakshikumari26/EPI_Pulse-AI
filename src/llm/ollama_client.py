"""Ollama LLM client wrapper for EpiPulse AI."""

import sys
from pathlib import Path
from typing import Optional

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config


class OllamaLLM:
    """Wrapper for Ollama local LLM inference."""

    def __init__(
        self,
        base_url: str = "http://localhost:11435",
        model: str = "mistral",
        timeout: int = 60,
    ):
        """
        Initialize Ollama LLM client.

        Args:
            base_url: Ollama server URL
            model: Model name (e.g., 'mistral', 'llama2', 'neural-chat')
            timeout: Request timeout in seconds
        """
        self.base_url = base_url
        self.model = model
        self.original_model = model
        self.timeout = timeout
        self.available_models: list[str] = []
        self.fallback_used = False
        self.fallback_model: str | None = None
        self._verify_connection()

    def _fetch_available_models(self) -> list[str]:
        """Fetch available models from the Ollama server."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            data = response.json()
            models = data.get("models", []) or []
            if isinstance(models, list):
                return [m.get("name") if isinstance(m, dict) else str(m) for m in models]
            return [str(models)]
        except requests.exceptions.RequestException as e:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running: `ollama serve`. "
                f"Details: {e}"
            )

    def _get_model_candidates(self, fallback_models: list[str] | None = None) -> list[str]:
        candidates = [self.model]
        if self.model and not self.model.endswith(":latest"):
            candidates.append(f"{self.model}:latest")
        if fallback_models:
            for m in fallback_models:
                if m not in candidates:
                    candidates.append(m)
        candidates.append("llama3.2:1b")
        candidates.append("mistral:latest")
        candidates.append("mistral")
        return candidates

    def _get_available_fallback_models(self) -> list[str]:
        config = load_config().get("llm", {})
        fallback_models = config.get("model_fallbacks", [])
        candidates = self._get_model_candidates(fallback_models)
        return [m for m in candidates if m != self.model and m in self.available_models]

    def _try_generate_with_model(self, model: str, prompt: str, temperature: float) -> str:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "temperature": temperature,
                "stream": False,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["response"].strip()

    def _verify_connection(self) -> bool:
        """Verify Ollama server is running and the configured model is available."""
        self.available_models = self._fetch_available_models()
        if not self.available_models:
            raise ConnectionError(
                f"Ollama is running at {self.base_url}, but no models are loaded. "
                "Run `ollama pull llama3.2:1b` or another model and restart the server."
            )

        config = load_config().get("llm", {})
        fallback_models = config.get("model_fallbacks", [])
        candidates = self._get_model_candidates(fallback_models)

        if self.model not in self.available_models:
            matched = next((m for m in candidates if m in self.available_models), None)
            if matched:
                self.model = matched
                self.fallback_used = True
                self.fallback_model = matched
            else:
                raise ConnectionError(
                    f"Configured model '{self.model}' is not available on Ollama. "
                    f"Available models: {self.available_models}. "
                    "Update configs/config.yaml or install a compatible model."
                )

        return True

    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """
        Generate text using Ollama.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0-1)

        Returns:
            Generated text
        """
        try:
            return self._try_generate_with_model(self.model, prompt, temperature)
        except requests.exceptions.HTTPError as e:
            response = e.response
            message = str(e)
            if response is not None and response.status_code == 404:
                message += (
                    " - model not found. "
                    "Verify the Ollama model name and that the model is loaded on the server."
                )
            else:
                if response is not None:
                    try:
                        error_info = response.json().get("error", "")
                    except Exception:
                        error_info = response.text
                    if "memory" in error_info.lower() or "system memory" in error_info.lower():
                        message += (
                            " - model requires more system memory than is available. "
                            "Try a smaller local model or run Ollama with CPU-only mode."
                        )
                    else:
                        message += f" - {error_info}"

                for fallback_model in self._get_available_fallback_models():
                    try:
                        result = self._try_generate_with_model(fallback_model, prompt, temperature)
                        self.model = fallback_model
                        self.fallback_used = True
                        self.fallback_model = fallback_model
                        return result
                    except requests.exceptions.HTTPError:
                        continue

            raise RuntimeError(
                f"Ollama generation failed: {message}. "
                "If you are on a low-memory machine, try `ollama serve --cpu` and pull a smaller model such as `llama3.2:1b` or `llama2-mini`."
            )
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama generation failed: {str(e)}")

    def explain_alert(
        self,
        cases: Optional[float] = None,
        risk_score: Optional[float] = None,
        z_score: Optional[float] = None,
        growth_rate: Optional[float] = None,
        region: str = "Unknown",
    ) -> str:
        """
        Generate explanation for outbreak alert using LLM.

        Args:
            cases: Number of cases
            risk_score: Risk score (0-100)
            z_score: Z-score for anomaly detection
            growth_rate: Growth rate of cases
            region: Region name

        Returns:
            Natural language explanation of the alert
        """
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

        prompt = f"""You are a public health epidemiologist. Analyze this disease outbreak alert data and provide a concise, professional explanation (1-2 sentences max).

Region: {region}
Metrics: {metrics_str}

Provide a brief, actionable explanation of what this data suggests about the outbreak situation. Be specific about risk level and any immediate concerns."""

        return self.generate(prompt, temperature=0.5)

    def summarize_forecast(self, forecast_data: dict, region: str = "Unknown") -> str:
        """
        Generate natural language summary of forecast.

        Args:
            forecast_data: Dictionary with forecast metrics
            region: Region name

        Returns:
            Natural language forecast summary
        """
        prompt = f"""You are a public health analyst. Summarize this disease forecast in 2-3 sentences.

Region: {region}
Forecast Data: {forecast_data}

Be concise and focus on trends, turning points, and actionable insights."""

        return self.generate(prompt, temperature=0.5)

    def generate_report(
        self,
        summary_data: dict,
    ) -> str:
        """
        Generate executive summary report.

        Args:
            summary_data: Dictionary with key metrics, alerts, forecasts

        Returns:
            Formatted report text
        """
        prompt = f"""Generate a brief executive summary report for public health officials (3-4 short paragraphs).

Data Summary:
{summary_data}

Structure: 
1. Current situation (cases, hotspots)
2. Risk assessment
3. Forecast outlook
4. Recommended actions

Be professional, data-driven, and actionable."""

        return self.generate(prompt, temperature=0.6)

    def answer_question(self, question: str, context: dict = None) -> str:
        """
        Answer epidemiological questions about the data.

        Args:
            question: User question
            context: Additional context data

        Returns:
            Answer to the question
        """
        context_str = f"\nContext: {context}" if context else ""
        prompt = f"""You are a disease surveillance expert. Answer this question about an outbreak situation based on epidemiological data.

Question: {question}{context_str}

Provide a clear, concise answer."""

        return self.generate(prompt, temperature=0.7)


def get_llm_client(force_new: bool = False) -> OllamaLLM:
    """
    Get or create Ollama LLM client (singleton pattern).

    Args:
        force_new: Force creation of new client

    Returns:
        OllamaLLM instance
    """
    if not hasattr(get_llm_client, "_instance") or force_new:
        config = load_config().get("llm", {})
        base_url = config.get("ollama_url", "http://localhost:11435")
        model = config.get("model", "llama3.2:1b")
        timeout = config.get("timeout", 60)

        get_llm_client._instance = OllamaLLM(
            base_url=base_url,
            model=model,
            timeout=timeout,
        )

    return get_llm_client._instance
