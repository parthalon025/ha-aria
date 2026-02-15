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


def ollama_chat(prompt, config: OllamaConfig = None):
    """Send prompt to local Ollama and return response."""
    if config is None:
        config = OllamaConfig()

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
        with urllib.request.urlopen(req, timeout=config.timeout) as resp:
            result = json.loads(resp.read())
        return result.get("message", {}).get("content", "")
    except Exception as e:
        logger.warning("Ollama chat request failed (model=%s): %s", config.model, e)
        return ""
