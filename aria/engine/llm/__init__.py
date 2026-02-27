"""LLM integration — Ollama client, insight reports, meta-learning."""

from aria.engine.llm.client import ollama_chat, strip_think_tags
from aria.engine.llm.meta_learning import (
    MAX_META_CHANGES_PER_WEEK,
    apply_suggestion_to_config,
    parse_suggestions,
    run_meta_learning,
    validate_suggestion,
)
from aria.engine.llm.reports import generate_brief_line, generate_insight_report

__all__ = [
    "MAX_META_CHANGES_PER_WEEK",
    "apply_suggestion_to_config",
    "generate_brief_line",
    "generate_insight_report",
    "ollama_chat",
    "parse_suggestions",
    "run_meta_learning",
    "strip_think_tags",
    "validate_suggestion",
]
