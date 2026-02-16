"""Tests for ARIA watchdog module."""

import json
import time
from datetime import UTC
from unittest.mock import MagicMock, patch

from aria.watchdog import (
    SMOKE_ENDPOINTS,
    WatchdogResult,
    _should_alert,
    attempt_restart,
    check_api_endpoints,
    check_cache_freshness,
    check_hub_health,
    check_service_status,
    check_timer_health,
    run_watchdog,
    send_alert,
)

# ---------------------------------------------------------------------------
# WatchdogResult
# ---------------------------------------------------------------------------


class TestWatchdogResult:
    def test_basic_creation(self):
        r = WatchdogResult(check_name="test", level="OK", message="all good")
        assert r.check_name == "test"
        assert r.level == "OK"
        assert r.message == "all good"
        assert r.details == {}

    def test_with_details(self):
        r = WatchdogResult(
            check_name="hub-liveness",
            level="CRITICAL",
            message="Hub down",
            details={"status": 0, "error": "Connection refused"},
        )
        assert r.details["status"] == 0
        assert "Connection refused" in r.details["error"]


# ---------------------------------------------------------------------------
# Hub health checks
# ---------------------------------------------------------------------------


class TestCheckHubHealth:
    @patch("aria.watchdog._http_get")
    def test_hub_healthy(self, mock_get):
        mock_get.return_value = (
            200,
            {
                "status": "ok",
                "uptime_seconds": 3600,
                "modules": {
                    "discovery": "running",
                    "ml_engine": "running",
                    "shadow_engine": "running",
                },
            },
            None,
        )

        results = check_hub_health()
        assert len(results) == 1
        assert results[0].level == "OK"
        assert results[0].check_name == "hub-liveness"

    @patch("aria.watchdog._http_get")
    def test_hub_unreachable(self, mock_get):
        mock_get.return_value = (0, None, "Connection refused")

        results = check_hub_health()
        assert len(results) == 1
        assert results[0].level == "CRITICAL"
        assert results[0].check_name == "hub-liveness"

    @patch("aria.watchdog._http_get")
    def test_module_failure(self, mock_get):
        mock_get.return_value = (
            200,
            {
                "status": "ok",
                "uptime_seconds": 100,
                "modules": {
                    "discovery": "running",
                    "ml_engine": "failed",
                    "shadow_engine": "running",
                },
            },
            None,
        )

        results = check_hub_health()
        # 1 OK for hub-liveness + 1 WARNING for ml_engine
        assert len(results) == 2
        assert results[0].level == "OK"
        assert results[1].level == "WARNING"
        assert results[1].check_name == "module-ml_engine"

    @patch("aria.watchdog._http_get")
    def test_multiple_module_failures(self, mock_get):
        mock_get.return_value = (
            200,
            {
                "status": "ok",
                "uptime_seconds": 100,
                "modules": {
                    "discovery": "failed",
                    "ml_engine": "failed",
                    "shadow_engine": "running",
                },
            },
            None,
        )

        results = check_hub_health()
        warnings = [r for r in results if r.level == "WARNING"]
        assert len(warnings) == 2


# ---------------------------------------------------------------------------
# API endpoint smoke tests
# ---------------------------------------------------------------------------


class TestCheckApiEndpoints:
    @patch("aria.watchdog._http_get")
    def test_all_endpoints_healthy(self, mock_get):
        mock_get.return_value = (200, {}, None)

        results = check_api_endpoints()
        assert len(results) == len(SMOKE_ENDPOINTS)
        assert all(r.level == "OK" for r in results)

    @patch("aria.watchdog._http_get")
    def test_some_endpoints_fail(self, mock_get):
        call_count = 0

        def side_effect(path):
            nonlocal call_count
            call_count += 1
            if call_count == 3:  # Third endpoint fails
                return (500, None, "Internal error")
            return (200, {}, None)

        mock_get.side_effect = side_effect

        results = check_api_endpoints()
        warnings = [r for r in results if r.level == "WARNING"]
        assert len(warnings) == 1

    @patch("aria.watchdog._http_get")
    def test_timeout_endpoint(self, mock_get):
        mock_get.return_value = (0, None, "timeout")

        results = check_api_endpoints()
        assert all(r.level == "WARNING" for r in results)


# ---------------------------------------------------------------------------
# Cache freshness
# ---------------------------------------------------------------------------


class TestCheckCacheFreshness:
    @patch("aria.watchdog._http_get")
    def test_fresh_cache(self, mock_get):
        now = time.time()
        mock_get.return_value = (
            200,
            [
                {"key": "intelligence", "last_updated": now - 3600},  # 1h ago
                {"key": "entities", "last_updated": now - 7200},  # 2h ago
            ],
            None,
        )

        results = check_cache_freshness()
        assert all(r.level == "OK" for r in results)

    @patch("aria.watchdog._http_get")
    def test_stale_daily_cache(self, mock_get):
        now = time.time()
        mock_get.return_value = (
            200,
            [
                {"key": "intelligence", "last_updated": now - 100000},  # ~28h ago
                {"key": "entities", "last_updated": now - 3600},  # 1h ago (fresh)
            ],
            None,
        )

        results = check_cache_freshness()
        stale = [r for r in results if r.level == "WARNING"]
        assert len(stale) == 1
        assert stale[0].check_name == "cache-stale-intelligence"

    @patch("aria.watchdog._http_get")
    def test_stale_activity_cache(self, mock_get):
        now = time.time()
        mock_get.return_value = (
            200,
            [
                {"key": "activity_summary", "last_updated": now - 7200},  # 2h ago
            ],
            None,
        )

        results = check_cache_freshness()
        stale = [r for r in results if r.level == "WARNING"]
        assert len(stale) == 1
        assert "activity_summary" in stale[0].check_name

    @patch("aria.watchdog._http_get")
    def test_cache_endpoint_unreachable(self, mock_get):
        mock_get.return_value = (0, None, "Connection refused")

        results = check_cache_freshness()
        assert len(results) == 1
        assert results[0].level == "WARNING"
        assert results[0].check_name == "cache-freshness"

    @patch("aria.watchdog._http_get")
    def test_iso_timestamp_format(self, mock_get):
        """Cache keys may use ISO 8601 timestamps instead of epoch."""
        from datetime import datetime, timedelta

        recent = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        mock_get.return_value = (
            200,
            [
                {"key": "intelligence", "last_updated": recent},
            ],
            None,
        )

        results = check_cache_freshness()
        assert all(r.level == "OK" for r in results)


# ---------------------------------------------------------------------------
# Timer health
# ---------------------------------------------------------------------------


class TestCheckTimerHealth:
    @patch("subprocess.run")
    def test_systemctl_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="Failed to connect")

        results = check_timer_health()
        assert any(r.level == "WARNING" for r in results)

    @patch("subprocess.run")
    def test_no_timers(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="NEXT LEFT LAST PASSED UNIT ACTIVATES\n\n0 timers listed.",
            stderr="",
        )

        results = check_timer_health()
        assert any(r.level == "OK" for r in results)


# ---------------------------------------------------------------------------
# Service status
# ---------------------------------------------------------------------------


class TestCheckServiceStatus:
    @patch("subprocess.run")
    def test_service_active(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="active\n", stderr="")

        results = check_service_status()
        assert len(results) == 1
        assert results[0].level == "OK"

    @patch("subprocess.run")
    def test_service_inactive(self, mock_run):
        mock_run.return_value = MagicMock(returncode=3, stdout="inactive\n", stderr="")

        results = check_service_status()
        assert len(results) == 1
        assert results[0].level == "CRITICAL"

    @patch("subprocess.run")
    def test_service_failed(self, mock_run):
        mock_run.return_value = MagicMock(returncode=3, stdout="failed\n", stderr="")

        results = check_service_status()
        assert len(results) == 1
        assert results[0].level == "CRITICAL"


# ---------------------------------------------------------------------------
# Auto-restart
# ---------------------------------------------------------------------------


class TestAttemptRestart:
    @patch("subprocess.run")
    def test_restart_success(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        logger = MagicMock()

        with patch("aria.watchdog.COOLDOWN_DIR", tmp_path / "cooldowns"):
            result = attempt_restart(logger)
            assert result is True
            logger.info.assert_called()

    @patch("subprocess.run")
    def test_restart_failure(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="Failed")
        logger = MagicMock()

        with patch("aria.watchdog.COOLDOWN_DIR", tmp_path / "cooldowns"):
            result = attempt_restart(logger)
            assert result is False

    @patch("subprocess.run")
    def test_restart_cooldown(self, mock_run, tmp_path):
        logger = MagicMock()
        cooldown_dir = tmp_path / "cooldowns"
        cooldown_dir.mkdir()
        cooldown_file = cooldown_dir / "restart-hub"
        cooldown_file.touch()

        with patch("aria.watchdog.COOLDOWN_DIR", cooldown_dir):
            result = attempt_restart(logger)
            assert result is False
            mock_run.assert_not_called()


# ---------------------------------------------------------------------------
# Alert cooldowns
# ---------------------------------------------------------------------------


class TestAlertCooldowns:
    def test_first_alert_allowed(self, tmp_path):
        with patch("aria.watchdog.COOLDOWN_DIR", tmp_path / "cooldowns"):
            assert _should_alert("test-alert", "WARNING") is True

    def test_recent_alert_blocked(self, tmp_path):
        cooldown_dir = tmp_path / "cooldowns"
        cooldown_dir.mkdir()
        (cooldown_dir / "test-alert").touch()

        with patch("aria.watchdog.COOLDOWN_DIR", cooldown_dir):
            assert _should_alert("test-alert", "WARNING") is False

    def test_expired_alert_allowed(self, tmp_path):
        cooldown_dir = tmp_path / "cooldowns"
        cooldown_dir.mkdir()
        f = cooldown_dir / "test-alert"
        f.touch()
        # Set mtime to 3 hours ago (past 2h WARNING cooldown)
        old_time = time.time() - 3 * 3600
        import os

        os.utime(str(f), (old_time, old_time))

        with patch("aria.watchdog.COOLDOWN_DIR", cooldown_dir):
            assert _should_alert("test-alert", "WARNING") is True


# ---------------------------------------------------------------------------
# Send alert
# ---------------------------------------------------------------------------


class TestSendAlert:
    @patch("urllib.request.urlopen")
    def test_send_success(self, mock_urlopen, tmp_path):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"ok": True}).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        logger = MagicMock()

        with (
            patch("aria.watchdog.COOLDOWN_DIR", tmp_path / "cooldowns"),
            patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "tok", "TELEGRAM_CHAT_ID": "123"}),
        ):
            result = send_alert("test", "WARNING", "test-key", logger)
            assert result is True

    def test_no_token(self, tmp_path):
        logger = MagicMock()

        with patch("aria.watchdog.COOLDOWN_DIR", tmp_path / "cooldowns"), patch.dict("os.environ", {}, clear=True):
            result = send_alert("test", "WARNING", "test-key", logger)
            assert result is False


# ---------------------------------------------------------------------------
# Full orchestrator
# ---------------------------------------------------------------------------


class TestRunWatchdog:
    @patch("aria.watchdog.check_timer_health")
    @patch("aria.watchdog.check_cache_freshness")
    @patch("aria.watchdog.check_api_endpoints")
    @patch("aria.watchdog.check_hub_health")
    @patch("aria.watchdog.check_service_status")
    @patch("aria.watchdog.setup_logging")
    def test_all_passing(self, mock_log, mock_svc, mock_hub, mock_api, mock_cache, mock_timer):  # noqa: PLR0913
        mock_log.return_value = MagicMock()
        mock_svc.return_value = [WatchdogResult("service-hub", "OK", "active")]
        mock_hub.return_value = [WatchdogResult("hub-liveness", "OK", "healthy")]
        mock_api.return_value = [WatchdogResult("api-core", "OK", "ok")]
        mock_cache.return_value = [WatchdogResult("cache-freshness", "OK", "fresh")]
        mock_timer.return_value = [WatchdogResult("timers", "OK", "ok")]

        ret = run_watchdog(quiet=True, no_alert=True)
        assert ret == 0

    @patch("aria.watchdog.check_timer_health")
    @patch("aria.watchdog.check_cache_freshness")
    @patch("aria.watchdog.check_api_endpoints")
    @patch("aria.watchdog.check_hub_health")
    @patch("aria.watchdog.check_service_status")
    @patch("aria.watchdog.setup_logging")
    def test_with_failures(self, mock_log, mock_svc, mock_hub, mock_api, mock_cache, mock_timer):  # noqa: PLR0913
        mock_log.return_value = MagicMock()
        mock_svc.return_value = [WatchdogResult("service-hub", "OK", "active")]
        mock_hub.return_value = [
            WatchdogResult("hub-liveness", "OK", "healthy"),
            WatchdogResult("module-ml_engine", "WARNING", "module failed"),
        ]
        mock_api.return_value = [WatchdogResult("api-core", "OK", "ok")]
        mock_cache.return_value = [WatchdogResult("cache-freshness", "OK", "fresh")]
        mock_timer.return_value = [WatchdogResult("timers", "OK", "ok")]

        ret = run_watchdog(quiet=True, no_alert=True)
        assert ret == 1

    @patch("aria.watchdog.check_timer_health")
    @patch("aria.watchdog.check_hub_health")
    @patch("aria.watchdog.check_service_status")
    @patch("aria.watchdog.setup_logging")
    def test_hub_down_skips_api_cache(self, mock_log, mock_svc, mock_hub, mock_timer):
        """When hub is down, API and cache checks are skipped."""
        mock_log.return_value = MagicMock()
        mock_svc.return_value = [WatchdogResult("service-hub", "CRITICAL", "down")]
        mock_hub.return_value = [WatchdogResult("hub-liveness", "CRITICAL", "unreachable")]
        mock_timer.return_value = [WatchdogResult("timers", "OK", "ok")]

        with (
            patch("aria.watchdog.attempt_restart", return_value=False),
            patch("aria.watchdog.check_api_endpoints") as mock_api,
            patch("aria.watchdog.check_cache_freshness") as mock_cache,
        ):
            ret = run_watchdog(quiet=True, no_alert=True)
            mock_api.assert_not_called()
            mock_cache.assert_not_called()
            assert ret == 1

    @patch("aria.watchdog.check_timer_health")
    @patch("aria.watchdog.check_cache_freshness")
    @patch("aria.watchdog.check_api_endpoints")
    @patch("aria.watchdog.check_hub_health")
    @patch("aria.watchdog.check_service_status")
    @patch("aria.watchdog.setup_logging")
    def test_json_output(self, mock_log, mock_svc, mock_hub, mock_api, mock_cache, mock_timer, capsys):  # noqa: PLR0913
        mock_log.return_value = MagicMock()
        mock_svc.return_value = [WatchdogResult("service-hub", "OK", "active")]
        mock_hub.return_value = [WatchdogResult("hub-liveness", "OK", "healthy")]
        mock_api.return_value = [WatchdogResult("api-core", "OK", "ok")]
        mock_cache.return_value = [WatchdogResult("cache-freshness", "OK", "fresh")]
        mock_timer.return_value = [WatchdogResult("timers", "OK", "ok")]

        run_watchdog(quiet=False, no_alert=True, json_output=True)
        output = json.loads(capsys.readouterr().out)
        assert output["passed"] == output["total"]
        assert "results" in output
        assert "timestamp" in output
