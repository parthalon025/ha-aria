"""Tests for automation CLI commands — patterns, gaps, suggest, shadow, rollback.

Covers Task 31 (Batch 11) CLI commands.
"""

import json
import sys
from io import StringIO
from unittest.mock import patch

from aria.cli import _build_parser, _dispatch

# ============================================================================
# Helpers
# ============================================================================


def _run_cli(command_args: list[str], hub_response=None, hub_post_response=None):
    """Run a CLI command, mocking the hub API calls.

    Returns (exit_code, stdout_text).
    """
    parser = _build_parser()
    args = parser.parse_args(command_args)

    captured = StringIO()
    old_stdout = sys.stdout

    with (
        patch("aria.cli._hub_api_get", return_value=hub_response),
        patch("aria.cli._hub_api_post", return_value=hub_post_response),
    ):
        try:
            sys.stdout = captured
            _dispatch(args)
            return 0, captured.getvalue()
        except SystemExit as e:
            return e.code, captured.getvalue()
        finally:
            sys.stdout = old_stdout


# ============================================================================
# aria patterns
# ============================================================================


class TestPatternsCommand:
    def test_patterns_no_hub(self):
        """Exits with error when hub is not running."""
        code, output = _run_cli(["patterns"])
        assert code == 1
        assert "hub running" in output.lower()

    def test_patterns_empty(self):
        """Shows message when no patterns detected."""
        code, output = _run_cli(
            ["patterns"],
            hub_response={"data": {"detections": []}},
        )
        assert code == 0
        assert "No patterns" in output

    def test_patterns_with_data(self):
        """Displays pattern detections."""
        detections = [
            {
                "trigger_entity": "binary_sensor.motion_kitchen",
                "confidence": 0.85,
                "source": "pattern",
                "observation_count": 42,
            }
        ]
        code, output = _run_cli(
            ["patterns"],
            hub_response={"data": {"detections": detections}},
        )
        assert code == 0
        assert "motion_kitchen" in output
        assert "0.85" in output

    def test_patterns_json(self):
        """JSON output mode."""
        detections = [{"trigger_entity": "sensor.test", "confidence": 0.9}]
        code, output = _run_cli(
            ["patterns", "--json"],
            hub_response={"data": {"detections": detections}},
        )
        assert code == 0
        parsed = json.loads(output)
        assert len(parsed) == 1


# ============================================================================
# aria gaps
# ============================================================================


class TestGapsCommand:
    def test_gaps_no_hub(self):
        """Exits with error when hub is not running."""
        code, output = _run_cli(["gaps"])
        assert code == 1

    def test_gaps_with_data(self):
        """Displays gap detections."""
        detections = [
            {
                "trigger_entity": "binary_sensor.door_front",
                "confidence": 0.75,
                "area_id": "living_room",
            }
        ]
        code, output = _run_cli(
            ["gaps"],
            hub_response={"data": {"detections": detections}},
        )
        assert code == 0
        assert "door_front" in output
        assert "living_room" in output

    def test_gaps_json(self):
        """JSON output mode."""
        code, output = _run_cli(
            ["gaps", "--json"],
            hub_response={"data": {"detections": [{"id": "g1"}]}},
        )
        assert code == 0
        parsed = json.loads(output)
        assert len(parsed) == 1


# ============================================================================
# aria suggest
# ============================================================================


class TestSuggestCommand:
    def test_suggest_no_hub(self):
        """Exits with error when hub is not running."""
        code, output = _run_cli(["suggest"])
        assert code == 1

    def test_suggest_with_data(self):
        """Displays suggestions."""
        suggestions = [
            {
                "suggestion_id": "abc123def456",
                "metadata": {"trigger_entity": "binary_sensor.motion"},
                "combined_score": 0.82,
                "status": "pending",
                "shadow_status": "new",
            }
        ]
        code, output = _run_cli(
            ["suggest"],
            hub_response={"data": {"suggestions": suggestions}},
        )
        assert code == 0
        assert "motion" in output
        assert "pending" in output

    def test_suggest_filter_status(self):
        """Filters suggestions by status."""
        suggestions = [
            {"suggestion_id": "a", "metadata": {}, "combined_score": 0.5, "status": "pending"},
            {"suggestion_id": "b", "metadata": {}, "combined_score": 0.5, "status": "approved"},
        ]
        code, output = _run_cli(
            ["suggest", "--status", "approved"],
            hub_response={"data": {"suggestions": suggestions}},
        )
        assert code == 0
        # Only 1 suggestion shown
        assert "1)" in output or "Suggestions (1)" in output

    def test_suggest_json(self):
        """JSON output mode."""
        code, output = _run_cli(
            ["suggest", "--json"],
            hub_response={"data": {"suggestions": [{"id": "s1", "status": "pending"}]}},
        )
        assert code == 0
        parsed = json.loads(output)
        assert len(parsed) == 1


# ============================================================================
# aria shadow
# ============================================================================


class TestShadowCommand:
    def test_shadow_no_subcommand(self):
        """Prints usage when no subcommand given."""
        code, output = _run_cli(["shadow"])
        assert code == 1
        assert "Usage" in output

    def test_shadow_sync_success(self):
        """Successful sync shows count."""
        code, output = _run_cli(
            ["shadow", "sync"],
            hub_post_response={"success": True, "count": 10, "changes": 3},
        )
        assert code == 0
        assert "10" in output
        assert "3" in output

    def test_shadow_sync_failure(self):
        """Failed sync shows error."""
        code, output = _run_cli(
            ["shadow", "sync"],
            hub_post_response={"success": False, "error": "connection refused"},
        )
        assert code == 1
        assert "connection refused" in output

    def test_shadow_status(self):
        """Shows shadow status."""
        code, output = _run_cli(
            ["shadow", "status"],
            hub_response={
                "ha_automations_count": 15,
                "ha_automations_last_synced": "2026-02-20T10:00:00",
                "suggestions_count": 5,
                "suggestions_last_generated": "2026-02-20T11:00:00",
                "pipeline_stage": "shadow",
            },
        )
        assert code == 0
        assert "15" in output
        assert "shadow" in output.lower()

    def test_shadow_compare(self):
        """Shows comparison results."""
        code, output = _run_cli(
            ["shadow", "compare"],
            hub_response={
                "comparisons": [
                    {
                        "trigger_entity": "binary_sensor.motion",
                        "shadow_status": "new",
                        "shadow_reason": "No match",
                    }
                ],
                "total_ha_automations": 10,
                "status_counts": {"new": 1},
            },
        )
        assert code == 0
        assert "motion" in output
        assert "new" in output


# ============================================================================
# aria rollback
# ============================================================================


class TestRollbackCommand:
    def test_rollback_no_flag(self):
        """Exits with usage when --last is not provided."""
        code, output = _run_cli(["rollback"])
        assert code == 1
        assert "Usage" in output

    def test_rollback_no_approved(self):
        """Exits when no approved suggestions exist."""
        code, output = _run_cli(
            ["rollback", "--last"],
            hub_response={"data": {"suggestions": [{"status": "pending"}]}},
        )
        assert code == 1
        assert "No approved" in output

    def test_rollback_success(self):
        """Successfully rolls back the last approved suggestion."""
        suggestions = [
            {
                "suggestion_id": "sid-001",
                "status": "approved",
                "created_at": "2026-02-20T10:00:00",
                "metadata": {"trigger_entity": "binary_sensor.motion"},
            },
        ]

        delete_response = {"status": "deleted", "suggestion_id": "sid-001", "remaining": 0}

        with (
            patch("aria.cli._hub_api_get", return_value={"data": {"suggestions": suggestions}}),
            patch("aria.cli._hub_api_delete", return_value=delete_response),
        ):
            captured = StringIO()
            old_stdout = sys.stdout
            parser = _build_parser()
            args = parser.parse_args(["rollback", "--last"])
            try:
                sys.stdout = captured
                _dispatch(args)
                code = 0
            except SystemExit as e:
                code = e.code
            finally:
                sys.stdout = old_stdout

        output = captured.getvalue()
        assert code == 0
        assert "Rolled back" in output
        assert "sid-001" in output
