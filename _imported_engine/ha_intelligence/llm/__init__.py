"""LLM integration — Ollama client, insight reports, meta-learning."""

from ha_intelligence.llm.client import ollama_chat, strip_think_tags
from ha_intelligence.llm.meta_learning import (
    MAX_META_CHANGES_PER_WEEK,
    apply_suggestion_to_config,
    parse_suggestions,
    run_meta_learning,
    validate_suggestion,
)
from ha_intelligence.llm.reports import generate_brief_line, generate_insight_report

__all__ = [
    "ollama_chat",
    "strip_think_tags",
    "generate_insight_report",
    "generate_brief_line",
    "parse_suggestions",
    "apply_suggestion_to_config",
    "validate_suggestion",
    "run_meta_learning",
    "MAX_META_CHANGES_PER_WEEK",
]
