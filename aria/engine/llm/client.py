"""Ollama LLM client — low-level chat API and output cleaning."""

import json
import logging
import re
import urllib.request

from aria.engine.config import OllamaConfig

logger = logging.getLogger(__name__)


def strip_think_tags(text):
    """Strip <think>...</think> blocks from deepseek-r1 output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def is_ollama_available(config: OllamaConfig | None = None) -> bool:
    """Check if Ollama service is available before submitting batch jobs."""
    if config is None:
        config = OllamaConfig()

    try:
        # Simple health check: POST empty message to verify service is up
        payload = json.dumps({"model": config.model, "messages": [], "stream": False}).encode()
        req = urllib.request.Request(
            config.url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            # Any 2xx response indicates service is up
            return resp.status < 400
    except Exception as e:
        logger.warning("Ollama health check failed: %s", e)
        return False


def ollama_chat(prompt, config: OllamaConfig | None = None):
    """Send prompt to local Ollama and return response."""
    if config is None:
        config = OllamaConfig()

    # Cap timeout at 30s and default to 30s if None/zero — prevents indefinite blocking
    effective_timeout = config.timeout if config.timeout and config.timeout > 0 else 30

    payload = json.dumps(
        {
            "model": config.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
    ).encode()
    req = urllib.request.Request(
        config.url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=effective_timeout) as resp:
            result = json.loads(resp.read())
        return result.get("message", {}).get("content", "")
    except Exception as e:
        logger.warning("Ollama chat request failed (model=%s): %s", config.model, e)
        return ""
