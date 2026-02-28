"""Regression tests for #241 (heartbeat file) and #320 (returncode check).

#241: run_watchdog must write a heartbeat file so external monitors can detect
      a stalled watchdog via file freshness.
#320: _check_timer_last_run must not silently pass when systemctl returns non-zero;
      it must log a warning and return early.
"""

import logging
from unittest.mock import MagicMock, patch

from aria.watchdog import _check_timer_last_run, run_watchdog


class TestHeartbeatFileCloses241:
    """#241 — watchdog must write a heartbeat file on every run."""

    def test_run_watchdog_writes_heartbeat(self, tmp_path):
        """run_watchdog() must write ~/ha-logs/watchdog/aria-heartbeat on every run."""
        heartbeat = tmp_path / "aria-heartbeat"

        with (
            patch("aria.watchdog.LOG_DIR", tmp_path),
            patch("aria.watchdog.LOG_FILE", tmp_path / "watchdog.log"),
            patch("aria.watchdog._collect_results", return_value=[]),
            patch("aria.watchdog._log_results"),
            patch("aria.watchdog._send_alerts"),
            patch("aria.watchdog.verify_telegram_connectivity", return_value=True),
            patch("aria.watchdog._print_report"),
        ):
            run_watchdog(quiet=True, no_alert=True)

        assert heartbeat.exists(), "Heartbeat file must exist after run_watchdog()"
        content = heartbeat.read_text().strip()
        assert "T" in content, f"Heartbeat must contain ISO timestamp, got: {content!r}"

    def test_heartbeat_timestamp_is_recent(self, tmp_path):
        """Heartbeat file content must be a recent ISO timestamp."""
        from datetime import UTC, datetime

        heartbeat = tmp_path / "aria-heartbeat"

        with (
            patch("aria.watchdog.LOG_DIR", tmp_path),
            patch("aria.watchdog.LOG_FILE", tmp_path / "watchdog.log"),
            patch("aria.watchdog._collect_results", return_value=[]),
            patch("aria.watchdog._log_results"),
            patch("aria.watchdog._send_alerts"),
            patch("aria.watchdog.verify_telegram_connectivity", return_value=True),
            patch("aria.watchdog._print_report"),
        ):
            before = datetime.now(UTC)
            run_watchdog(quiet=True, no_alert=True)
            after = datetime.now(UTC)

        ts = datetime.fromisoformat(heartbeat.read_text().strip())
        assert before <= ts <= after, "Heartbeat timestamp must be within the run window"


class TestReturncodeCheckCloses320:
    """#320 — _check_timer_last_run must log warning on non-zero systemctl returncode."""

    def test_nonzero_returncode_logs_warning(self, caplog):
        """_check_timer_last_run must warn and return early when systemctl fails."""
        failed_proc = MagicMock()
        failed_proc.returncode = 1
        failed_proc.stdout = ""
        failed_proc.stderr = "Unit not found."

        results = []
        with (
            patch("aria.watchdog.subprocess.run", return_value=failed_proc),
            caplog.at_level(logging.WARNING, logger="aria.watchdog"),
        ):
            _check_timer_last_run("aria-nonexistent.timer", results)

        assert not results, "No results should be appended on systemctl failure"
        assert any("returncode" in r.message or "failed" in r.message.lower() for r in caplog.records), (
            "A warning must be logged when systemctl returns non-zero"
        )

    def test_zero_returncode_does_not_warn(self, caplog):
        """_check_timer_last_run must NOT warn when systemctl returns 0 with valid output."""
        ok_proc = MagicMock()
        ok_proc.returncode = 0
        ok_proc.stdout = "LastTriggerUSec=n/a"
        ok_proc.stderr = ""

        results = []
        with (
            patch("aria.watchdog.subprocess.run", return_value=ok_proc),
            caplog.at_level(logging.WARNING, logger="aria.watchdog"),
        ):
            _check_timer_last_run("aria-test.timer", results)

        assert not any("returncode" in r.message for r in caplog.records), "No returncode warning on success"
