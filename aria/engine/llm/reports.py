"""LLM-powered insight reports and brief generation."""

import json

from aria.engine.config import OllamaConfig
from aria.engine.llm.client import ollama_chat, strip_think_tags


def generate_insight_report(  # noqa: PLR0913 — report generation requires all context inputs
    snapshot, anomalies, predictions, reliability, correlations, accuracy_history, config: OllamaConfig = None
):
    """Generate natural language insight report via Ollama."""
    if config is None:
        config = OllamaConfig()

    context = {
        "date": snapshot["date"],
        "day": snapshot["day_of_week"],
        "weather": snapshot.get("weather", {}),
        "power_watts": snapshot["power"]["total_watts"],
        "lights_on": snapshot["lights"]["on"],
        "people_home": snapshot["occupancy"]["people_home"],
        "devices_home": snapshot["occupancy"]["device_count_home"],
        "ev": snapshot.get("ev", {}),
        "anomalies": [a["description"] for a in anomalies],
        "predictions_tomorrow": {k: v for k, v in predictions.items() if isinstance(v, dict) and "predicted" in v},
        "degrading_devices": [eid for eid, data in reliability.items() if data.get("trend") == "degrading"],
        "top_correlations": [c["description"] for c in (correlations or [])[:5]],
        "accuracy_trend": accuracy_history.get("trend", "unknown"),
        "recent_accuracy": [s["overall"] for s in accuracy_history.get("scores", [])[-7:]],
    }

    prompt = f"""You are a home intelligence analyst. Analyze this smart home data and provide insights.

DATA:
{json.dumps(context, indent=2)}

Provide a concise report with these sections:
1. TODAY'S SUMMARY (2-3 sentences: what happened, any anomalies)
2. PREDICTIONS (what to expect tomorrow, with confidence)
3. DEVICE HEALTH (any degrading devices, recommended actions)
4. PATTERNS DISCOVERED (interesting correlations)
5. SELF-ASSESSMENT (how accurate have predictions been, what's improving)

Rules:
- Be specific: use actual numbers and device names
- If there are no anomalies, say so briefly
- Predictions should state confidence level
- Device health should suggest specific actions
- Keep total output under 300 words
"""
    report_config = OllamaConfig(url=config.url, model=config.model, timeout=90)
    return strip_think_tags(ollama_chat(prompt, config=report_config))


def generate_brief_line(snapshot, anomalies, predictions, accuracy_history, config: OllamaConfig = None):
    """Generate a single-line intelligence summary for telegram-brief."""
    parts = []
    if anomalies:
        parts.append(f"{len(anomalies)} anomalies")
    else:
        parts.append("normal")
    scores = accuracy_history.get("scores", [])
    if scores:
        parts.append(f"accuracy:{scores[-1]['overall']}%")
    preds = {k: v for k, v in predictions.items() if isinstance(v, dict) and "predicted" in v}
    if preds.get("power_watts"):
        parts.append(f"tmrw power:{preds['power_watts']['predicted']:.0f}W")
    return f"Intelligence: {' | '.join(parts)}"
