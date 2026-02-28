"""Tests that LLM parse failures emit logger.warning before returning [].

Closes #321.
"""

from unittest.mock import patch

from aria.engine.llm.automation_suggestions import parse_automation_suggestions
from aria.engine.llm.meta_learning import parse_suggestions

# A response that matches the JSON-array regex but contains invalid JSON.
# The regex r"\[.*\]" will match "[BAD_JSON_MARKER]" but json.loads will fail
# because it's not valid JSON (bare strings without quotes).
_BAD_META_RESPONSE = "[BAD_JSON_MARKER_12345_action_target]"
_BAD_AUTO_RESPONSE = "[BAD_AUTO_MARKER_67890_description_yaml]"


def test_meta_learning_parse_failure_logs_warning_closes_321():
    """parse_suggestions must call logger.warning when JSON is malformed."""
    with patch("aria.engine.llm.meta_learning.logger") as mock_logger:
        result = parse_suggestions(_BAD_META_RESPONSE)
    assert result == []
    assert mock_logger.warning.called, (
        "parse_suggestions returned [] on bad JSON without logging — silent failure (#321)"
    )


def test_automation_suggestions_parse_failure_logs_warning_closes_321():
    """parse_automation_suggestions must call logger.warning when JSON is malformed."""
    with patch("aria.engine.llm.automation_suggestions.logger") as mock_logger:
        result = parse_automation_suggestions(_BAD_AUTO_RESPONSE)
    assert result == []
    assert mock_logger.warning.called, (
        "parse_automation_suggestions returned [] on bad JSON without logging — silent failure (#321)"
    )


def test_meta_learning_warning_includes_raw_response_closes_321():
    """Warning message must include a preview of the raw LLM response."""
    with patch("aria.engine.llm.meta_learning.logger") as mock_logger:
        parse_suggestions(_BAD_META_RESPONSE)
    assert mock_logger.warning.called
    # The warning args must include the raw response snippet somewhere
    all_args = " ".join(str(a) for a in mock_logger.warning.call_args[0])
    assert "BAD_JSON_MARKER_12345" in all_args, f"Warning args do not include raw response preview. Got: {all_args!r}"
