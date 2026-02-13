"""LLM-powered HA automation suggestions from learned patterns.

Uses entity correlation data and behavioral patterns to generate
validated Home Assistant automation YAML via Ollama.
"""

import json
import re
from datetime import datetime

from ha_intelligence.config import AppConfig, OllamaConfig
from ha_intelligence.llm.client import ollama_chat, strip_think_tags
from ha_intelligence.storage.data_store import DataStore


AUTOMATION_PROMPT = """You are a Home Assistant automation expert analyzing real behavioral data from a smart home.

## Entity Correlation Patterns (from learned data)
These entity pairs frequently change state within the same time window:
{co_occurrences}

## Hourly Activity Patterns
Peak activity times per entity:
{hourly_patterns}

## Current Correlations (Pearson r between metrics)
{metric_correlations}

## System Baselines
Day-of-week averages for key metrics:
{baselines}

## Task
Generate 1-3 Home Assistant automation suggestions based on these REAL patterns.

Rules:
- ONLY suggest automations that match observed patterns (cite the correlation data)
- Each automation must include valid HA YAML
- Focus on quality of life improvements, not security-critical automations
- Prefer simple trigger → action patterns over complex multi-condition automations
- Include a description explaining WHY this automation makes sense based on the data

Output as a JSON array. Each suggestion must have:
1. "description": Human-readable explanation citing observed patterns
2. "trigger_entity": The entity that triggers the automation
3. "action_entity": The entity that gets controlled
4. "confidence": "high" | "medium" | "low" (based on correlation strength)
5. "yaml": Valid HA automation YAML string

Example:
[{{"description": "Motion in hallway triggers hallway light (observed 20x co-occurrence, P=0.9)", "trigger_entity": "binary_sensor.motion_hallway", "action_entity": "light.hallway", "confidence": "high", "yaml": "alias: Motion-activated hallway light\\ntrigger:\\n  - platform: state\\n    entity_id: binary_sensor.motion_hallway\\n    to: 'on'\\naction:\\n  - service: light.turn_on\\n    target:\\n      entity_id: light.hallway"}}]
"""


def _validate_yaml_structure(yaml_str: str) -> bool:
    """Basic validation that the YAML has required automation keys."""
    required = ["trigger", "action"]
    yaml_lower = yaml_str.lower()
    return all(key in yaml_lower for key in required)


def _format_co_occurrences(entity_corrs: dict) -> str:
    """Format entity correlations for the LLM prompt."""
    pairs = entity_corrs.get("top_co_occurrences", [])
    if not pairs:
        return "No entity correlation data available yet."

    lines = []
    for p in pairs[:10]:
        prob = max(p.get("conditional_prob_a_given_b", 0),
                   p.get("conditional_prob_b_given_a", 0))
        lines.append(
            f"- {p['entity_a']} ↔ {p['entity_b']}: "
            f"{p['count']}x co-occurrence, P={prob:.0%}, "
            f"peak hour={p.get('typical_hour', '?')}, "
            f"strength={p.get('strength', '?')}"
        )
    return "\n".join(lines) or "No patterns found."


def parse_automation_suggestions(llm_response: str) -> list:
    """Parse automation suggestions from LLM response.

    Returns list of validated suggestion dicts, or empty list on failure.
    """
    text = strip_think_tags(llm_response)

    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if not match:
        return []

    try:
        suggestions = json.loads(match.group())
        if not isinstance(suggestions, list):
            return []

        valid = []
        for s in suggestions[:3]:  # Max 3 suggestions
            if not isinstance(s, dict):
                continue
            if not all(k in s for k in ("description", "yaml")):
                continue
            if not _validate_yaml_structure(s["yaml"]):
                continue
            valid.append(s)
        return valid
    except (json.JSONDecodeError, ValueError):
        return []


def generate_automation_suggestions(config: AppConfig = None,
                                     store: DataStore = None) -> dict:
    """Generate HA automation suggestions from learned patterns.

    Returns dict with suggestions list and metadata.
    """
    if config is None:
        config = AppConfig.from_env()
    if store is None:
        store = DataStore(config.paths)

    # Gather context
    entity_corrs = store.load_entity_correlations()
    if not entity_corrs or not entity_corrs.get("top_co_occurrences"):
        return {"error": "no entity correlation data — run --entity-correlations first"}

    metric_corrs = store.load_correlations()
    if isinstance(metric_corrs, dict):
        metric_corrs = metric_corrs.get("correlations", [])

    baselines = store.load_baselines()

    # Format hourly patterns from automation-worthy pairs
    hourly_info = []
    for pair in entity_corrs.get("automation_worthy_pairs", [])[:5]:
        hourly_info.append(
            f"- {pair['entity_a']} & {pair['entity_b']}: "
            f"peak hour {pair.get('typical_hour', '?')}"
        )

    prompt = AUTOMATION_PROMPT.format(
        co_occurrences=_format_co_occurrences(entity_corrs),
        hourly_patterns="\n".join(hourly_info) or "No hourly data yet.",
        metric_correlations=json.dumps(metric_corrs[:5], indent=2) if metric_corrs else "No metric correlations yet.",
        baselines=json.dumps(
            {day: {k: v.get("mean") for k, v in metrics.items() if isinstance(v, dict)}
             for day, metrics in baselines.items()} if baselines else {},
            indent=2,
        ),
    )

    print("Querying LLM for automation suggestions...")
    ollama_config = OllamaConfig(url=config.ollama.url, model=config.ollama.model, timeout=120)
    response = ollama_chat(prompt, config=ollama_config)
    if not response:
        return {"error": "empty LLM response"}

    suggestions = parse_automation_suggestions(response)

    # Save to insights/automation-suggestions/
    output_dir = config.paths.insights_dir / "automation-suggestions"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    result = {
        "generated_at": datetime.now().isoformat(),
        "suggestions": suggestions,
        "entity_patterns_used": len(entity_corrs.get("top_co_occurrences", [])),
        "metric_correlations_used": len(metric_corrs) if metric_corrs else 0,
    }

    output_path = output_dir / f"{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    # Also save individual YAML files for easy import
    for i, s in enumerate(suggestions):
        yaml_path = output_dir / f"{timestamp}_{i+1}.yaml"
        with open(yaml_path, "w") as f:
            f.write(f"# {s.get('description', 'Suggested automation')}\n")
            f.write(f"# Confidence: {s.get('confidence', 'unknown')}\n")
            f.write(f"# Generated: {timestamp}\n\n")
            f.write(s.get("yaml", ""))
            f.write("\n")

    print(f"Saved {len(suggestions)} suggestions to {output_dir}")
    return result
