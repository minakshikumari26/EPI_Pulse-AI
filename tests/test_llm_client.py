import unittest
from unittest.mock import MagicMock, patch

from src.llm.ollama_client import OllamaLLM


class LLMClientTests(unittest.TestCase):
    @patch("src.llm.ollama_client.load_config")
    @patch("src.llm.ollama_client.requests.post")
    @patch("src.llm.ollama_client.requests.get")
    def test_model_fallback_selects_available_model(self, mock_get, mock_post, mock_load_config):
        mock_load_config.return_value = {
            "llm": {
                "model": "mistral",
                "model_fallbacks": ["llama3.2:1b", "mistral:latest"],
            }
        }

        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"models": [{"name": "llama3.2:1b"}]},
            raise_for_status=lambda: None,
        )
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"response": "ok"},
            raise_for_status=lambda: None,
        )

        client = OllamaLLM(base_url="http://localhost:11435", model="mistral")
        self.assertEqual(client.model, "llama3.2:1b")
        self.assertEqual(client.generate("Hello"), "ok")

    @patch("src.llm.ollama_client.load_config")
    @patch("src.llm.ollama_client.requests.post")
    @patch("src.llm.ollama_client.requests.get")
    def test_generate_memory_error_reports_suggestion(self, mock_get, mock_post, mock_load_config):
        mock_load_config.return_value = {
            "llm": {
                "model": "llama3.2:1b",
                "model_fallbacks": ["mistral:latest"],
            }
        }

        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"models": [{"name": "llama3.2:1b"}]},
            raise_for_status=lambda: None,
        )

        import requests

        response_mock = MagicMock(
            status_code=500,
            json=lambda: {"error": "model requires more system memory (3.2 GiB) than is available"},
            text="model requires more system memory",
        )
        response_mock.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Server Error", response=response_mock)
        mock_post.return_value = response_mock

        client = OllamaLLM(base_url="http://localhost:11435", model="llama3.2:1b")
        with self.assertRaises(RuntimeError) as ctx:
            client.generate("Hello")

        self.assertIn("more system memory", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
