"""Ollama LLM client — low-level chat API and output cleaning."""

import json
import re
import urllib.request

from ha_intelligence.config import OllamaConfig


def strip_think_tags(text):
    """Strip <think>...</think> blocks from deepseek-r1 output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def ollama_chat(prompt, config: OllamaConfig = None):
    """Send prompt to local Ollama and return response."""
    if config is None:
        config = OllamaConfig()

    payload = json.dumps({
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        config.url, data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=config.timeout) as resp:
            result = json.loads(resp.read())
        return result.get("message", {}).get("content", "")
    except Exception:
        return ""
