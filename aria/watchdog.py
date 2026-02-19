"""ARIA Watchdog — monitors hub health, timers, cache freshness, and alerts on failures."""

import json
import logging
import os
import subprocess
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class WatchdogResult:
    """Result of a single watchdog check."""

    check_name: str
    level: str  # "OK", "WARNING", "CRITICAL"
    message: str
    details: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HUB_URL = os.environ.get("ARIA_HUB_URL", "http://127.0.0.1:8001")
HTTP_TIMEOUT = 10
LOG_DIR = Path(os.path.expanduser("~/ha-logs/watchdog"))
LOG_FILE = LOG_DIR / "aria-watchdog.log"
COOLDOWN_DIR = Path("/tmp/aria-watchdog-cooldowns")

COOLDOWN_SECONDS = {
    "CRITICAL": 30 * 60,  # 30 min
    "WARNING": 2 * 60 * 60,  # 2 hours
}

# Cache staleness thresholds (seconds)
DAILY_STALE_THRESHOLD = 26 * 3600  # 26 hours
ACTIVITY_STALE_THRESHOLD = 1 * 3600  # 1 hour

DAILY_CACHE_CATEGORIES = {"intelligence", "entities", "devices", "areas"}
ACTIVITY_CACHE_CATEGORIES = {"activity_summary"}

# API smoke-test endpoints
SMOKE_ENDPOINTS = [
    ("/api/version", "core"),
    ("/api/ml/drift", "ml-engine"),
    ("/api/shadow/accuracy", "shadow-engine"),
    ("/api/activity/current", "activity-monitor"),
    ("/api/capabilities/registry", "discovery"),
    ("/api/pipeline", "pipeline"),
    ("/api/config", "config"),
    ("/api/curation/summary", "data-quality"),
    ("/api/validation/latest", "validation"),
]


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def setup_logging() -> logging.Logger:
    """Configure rotating file logger for watchdog runs."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("aria.watchdog")
    logger.setLevel(logging.DEBUG)

    # Rotating file handler: 5 MB, 3 backups (~20 MB max)
    if not logger.handlers:
        fh = RotatingFileHandler(str(LOG_FILE), maxBytes=5 * 1024 * 1024, backupCount=3)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------


def _http_get(path: str, timeout: int = HTTP_TIMEOUT) -> tuple:
    """GET request to hub. Returns (status_code, parsed_json | None, error | None)."""
    url = f"{HUB_URL}{path}"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read())
            return resp.status, body, None
    except urllib.error.HTTPError as e:
        return e.code, None, str(e)
    except Exception as e:
        return 0, None, str(e)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Telegram startup probe
# ---------------------------------------------------------------------------

# Module-level state for Telegram connectivity (set by verify_telegram_connectivity)
last_telegram_ok: bool = False


def verify_telegram_connectivity() -> bool:
    """Verify Telegram bot token works by calling the getMe API.

    Updates the module-level last_telegram_ok flag. Returns True if the
    bot token is valid and the API responds, False otherwise.
    Does not raise — failures are logged and stored silently.
    """
    global last_telegram_ok

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        logging.getLogger("aria.watchdog").warning("TELEGRAM_BOT_TOKEN not set — Telegram probe skipped")
        last_telegram_ok = False
        return False

    url = f"https://api.telegram.org/bot{token}/getMe"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read())
            if body.get("ok"):
                last_telegram_ok = True
                logging.getLogger("aria.watchdog").info(
                    "Telegram probe OK (bot: %s)", body.get("result", {}).get("username", "?")
                )
                return True
            else:
                last_telegram_ok = False
                logging.getLogger("aria.watchdog").warning("Telegram probe failed: API returned ok=false")
                return False
    except Exception as e:
        last_telegram_ok = False
        logging.getLogger("aria.watchdog").warning("Telegram probe failed: %s", e)
        return False


def check_hub_health() -> list:
    """Check hub /health endpoint — liveness + module health."""
    results = []
    status, data, err = _http_get("/health")

    if status != 200 or data is None:
        results.append(
            WatchdogResult(
                check_name="hub-liveness",
                level="CRITICAL",
                message=f"Hub unreachable at {HUB_URL}/health",
                details={"status": status, "error": err},
            )
        )
        return results

    results.append(
        WatchdogResult(
            check_name="hub-liveness",
            level="OK",
            message=f"Hub healthy (uptime: {data.get('uptime_seconds', '?')}s)",
            details={"uptime_seconds": data.get("uptime_seconds")},
        )
    )

    # Module health from /health response
    modules = data.get("modules", {})
    for mod_id, mod_status in modules.items():
        if mod_status != "running":
            results.append(
                WatchdogResult(
                    check_name=f"module-{mod_id}",
                    level="WARNING",
                    message=f"Module {mod_id} status: {mod_status}",
                    details={"module": mod_id, "status": mod_status},
                )
            )

    return results


def check_api_endpoints() -> list:
    """Smoke-test representative API endpoints."""
    results = []
    for path, label in SMOKE_ENDPOINTS:
        status, _, err = _http_get(path)
        if status == 200:
            results.append(
                WatchdogResult(
                    check_name=f"api-{label}",
                    level="OK",
                    message=f"{path} responded 200",
                )
            )
        else:
            results.append(
                WatchdogResult(
                    check_name=f"api-{label}",
                    level="WARNING",
                    message=f"{path} returned {status}",
                    details={"status": status, "error": err},
                )
            )
    return results


def check_cache_freshness() -> list:
    """Check cache key timestamps for staleness."""
    results = []
    status, data, err = _http_get("/api/cache/keys")

    if status != 200 or data is None:
        results.append(
            WatchdogResult(
                check_name="cache-freshness",
                level="WARNING",
                message="Could not fetch cache keys",
                details={"status": status, "error": err},
            )
        )
        return results

    now = time.time()
    keys = data if isinstance(data, list) else data.get("keys", [])

    for entry in keys:
        # Supports both {"key": "x", "last_updated": "..."} and {"category": "x", ...}
        key = entry.get("key") or entry.get("category", "unknown")
        last_updated = entry.get("last_updated") or entry.get("updated_at")
        if not last_updated:
            continue

        try:
            if isinstance(last_updated, int | float):
                ts = float(last_updated)
            else:
                dt = datetime.fromisoformat(str(last_updated).replace("Z", "+00:00"))
                ts = dt.timestamp()
        except (ValueError, TypeError):
            continue

        age = now - ts
        threshold = None
        if key in DAILY_CACHE_CATEGORIES:
            threshold = DAILY_STALE_THRESHOLD
        elif key in ACTIVITY_CACHE_CATEGORIES:
            threshold = ACTIVITY_STALE_THRESHOLD

        if threshold and age > threshold:
            hours = age / 3600
            results.append(
                WatchdogResult(
                    check_name=f"cache-stale-{key}",
                    level="WARNING",
                    message=f"{key} last updated {hours:.0f}h ago",
                    details={"category": key, "age_hours": round(hours, 1)},
                )
            )

    if not any(r.level != "OK" for r in results):
        results.append(
            WatchdogResult(
                check_name="cache-freshness",
                level="OK",
                message="All cache categories within thresholds",
            )
        )

    return results


def check_timer_health() -> list:
    """Check aria-* systemd timers for missed runs."""
    results = []
    try:
        proc = subprocess.run(
            ["systemctl", "--user", "list-timers", "aria-*", "--no-pager", "--plain"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode != 0:
            results.append(
                WatchdogResult(
                    check_name="timers",
                    level="WARNING",
                    message=f"systemctl list-timers failed: {proc.stderr.strip()}",
                )
            )
            return results

        lines = proc.stdout.strip().split("\n")
        # Skip header line and trailing summary
        for line in lines[1:]:
            if not line.strip() or line.startswith(" ") or "timers listed" in line:
                continue
            parts = line.split()
            # Format: NEXT LEFT LAST PASSED UNIT ACTIVATES
            if len(parts) < 6:
                continue
            unit = parts[-2] if parts[-2].endswith(".timer") else parts[-1]
            # Find LAST timestamp — it's between LEFT and PASSED
            # Parse the PASSED field to determine how long ago
            # systemctl output is tricky, so we'll check via show
            _check_timer_last_run(unit, results)

    except Exception as e:
        results.append(
            WatchdogResult(
                check_name="timers",
                level="WARNING",
                message=f"Timer check error: {e}",
            )
        )

    if not any(r.level != "OK" for r in results):
        results.append(
            WatchdogResult(
                check_name="timers",
                level="OK",
                message="All timers within schedule",
            )
        )

    return results


def _check_timer_last_run(unit: str, results: list):
    """Check a single timer's last trigger time."""
    try:
        proc = subprocess.run(
            ["systemctl", "--user", "show", unit, "--property=LastTriggerUSec"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        line = proc.stdout.strip()
        if "=" not in line:
            return

        value = line.split("=", 1)[1].strip()
        if not value or value == "n/a":
            return

        # Parse systemd timestamp
        try:
            last = datetime.strptime(value, "%a %Y-%m-%d %H:%M:%S %Z")
        except ValueError:
            try:
                last = datetime.strptime(value, "%a %Y-%m-%d %H:%M:%S %z")
            except ValueError:
                return

        age = (datetime.now() - last.replace(tzinfo=None)).total_seconds()
        timer_name = unit.replace(".timer", "")

        # Determine expected interval
        is_weekly = "weekly" in timer_name or timer_name in {
            "aria-retrain",
            "aria-meta-learn",
            "aria-prophet",
        }
        threshold = 8 * 86400 if is_weekly else DAILY_STALE_THRESHOLD

        if age > threshold:
            hours = age / 3600
            results.append(
                WatchdogResult(
                    check_name=f"timer-{timer_name}",
                    level="WARNING",
                    message=f"{timer_name} last ran {hours:.0f}h ago",
                    details={"timer": timer_name, "age_hours": round(hours, 1)},
                )
            )
    except Exception:
        pass  # Non-fatal per-timer check


def check_audit_alerts(  # noqa: PLR0911
    audit_db_path: str,
    threshold: int = 10,
    window_minutes: int = 5,
) -> WatchdogResult:
    """Check audit.db for recent error-severity events."""
    import sqlite3
    from contextlib import closing
    from datetime import timedelta

    if not os.path.exists(audit_db_path):
        return WatchdogResult(
            check_name="audit_alerts",
            level="OK",
            message="Audit DB not found — skipping",
        )

    cutoff = (datetime.now(UTC) - timedelta(minutes=window_minutes)).isoformat()
    try:
        with closing(sqlite3.connect(audit_db_path)) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM audit_events WHERE severity = 'error' AND timestamp >= ?",
                (cutoff,),
            ).fetchone()[0]
    except sqlite3.OperationalError as e:
        if "no such table" in str(e):
            return WatchdogResult(
                check_name="audit_alerts",
                level="OK",
                message="Audit schema not yet initialized — skipping",
            )
        return WatchdogResult(
            check_name="audit_alerts",
            level="WARNING",
            message=f"Failed to read audit.db: {e}",
        )
    except Exception as e:
        return WatchdogResult(
            check_name="audit_alerts",
            level="WARNING",
            message=f"Failed to read audit.db: {e}",
        )

    if count >= threshold:
        return WatchdogResult(
            check_name="audit_alerts",
            level="CRITICAL",
            message=f"{count} audit errors in last {window_minutes}min (threshold: {threshold})",
            details={"error_count": count, "window_minutes": window_minutes},
        )
    elif count > 0:
        return WatchdogResult(
            check_name="audit_alerts",
            level="OK",
            message=f"{count} audit errors in last {window_minutes}min (below threshold {threshold})",
        )
    return WatchdogResult(
        check_name="audit_alerts",
        level="OK",
        message="No recent audit errors",
    )


def check_disk_space(warn_threshold: float = 90.0) -> WatchdogResult:
    """Check disk usage on the root partition.

    Args:
        warn_threshold: Percentage usage that triggers a WARNING.

    Returns:
        WatchdogResult with disk usage details.
    """
    try:
        import shutil

        usage = shutil.disk_usage("/")
        pct_used = (usage.used / usage.total) * 100
        free_gb = usage.free / (1024**3)

        if pct_used >= warn_threshold:
            return WatchdogResult(
                check_name="disk-space",
                level="WARNING",
                message=f"Disk usage at {pct_used:.1f}% ({free_gb:.1f} GB free)",
                details={"percent_used": round(pct_used, 1), "free_gb": round(free_gb, 1)},
            )
        return WatchdogResult(
            check_name="disk-space",
            level="OK",
            message=f"Disk usage at {pct_used:.1f}% ({free_gb:.1f} GB free)",
            details={"percent_used": round(pct_used, 1), "free_gb": round(free_gb, 1)},
        )
    except Exception as e:
        return WatchdogResult(
            check_name="disk-space",
            level="WARNING",
            message=f"Disk check failed: {e}",
        )


def check_ollama_health() -> WatchdogResult:
    """Check if Ollama API is reachable.

    Tries to connect to the Ollama API endpoint and report status.

    Returns:
        WatchdogResult with Ollama health status.
    """
    ollama_url = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    try:
        req = urllib.request.Request(f"{ollama_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            model_count = len(data.get("models", []))
            return WatchdogResult(
                check_name="ollama",
                level="OK",
                message=f"Ollama reachable ({model_count} models loaded)",
                details={"model_count": model_count},
            )
    except Exception as e:
        return WatchdogResult(
            check_name="ollama",
            level="WARNING",
            message=f"Ollama unreachable: {e}",
            details={"url": ollama_url, "error": str(e)},
        )


def check_service_status() -> list:
    """Check if aria-hub.service is active."""
    results = []
    try:
        proc = subprocess.run(
            ["systemctl", "--user", "is-active", "aria-hub.service"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        status = proc.stdout.strip()
        if status == "active":
            results.append(
                WatchdogResult(
                    check_name="service-hub",
                    level="OK",
                    message="aria-hub.service is active",
                )
            )
        else:
            results.append(
                WatchdogResult(
                    check_name="service-hub",
                    level="CRITICAL",
                    message=f"aria-hub.service is {status}",
                    details={"status": status},
                )
            )
    except Exception as e:
        results.append(
            WatchdogResult(
                check_name="service-hub",
                level="CRITICAL",
                message=f"Could not check service status: {e}",
            )
        )
    return results


# ---------------------------------------------------------------------------
# Auto-restart
# ---------------------------------------------------------------------------


def attempt_restart(logger: logging.Logger) -> bool:
    """Attempt to restart aria-hub.service with cooldown guard."""
    cooldown_file = COOLDOWN_DIR / "restart-hub"
    COOLDOWN_DIR.mkdir(parents=True, exist_ok=True)

    if cooldown_file.exists():
        age = time.time() - cooldown_file.stat().st_mtime
        if age < COOLDOWN_SECONDS["CRITICAL"]:
            logger.info(f"Restart cooldown active ({age:.0f}s since last attempt)")
            return False

    logger.warning("Attempting to restart aria-hub.service")
    try:
        proc = subprocess.run(
            ["systemctl", "--user", "restart", "aria-hub.service"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        cooldown_file.touch()
        if proc.returncode == 0:
            logger.info("aria-hub.service restart succeeded")
            return True
        else:
            logger.error(f"Restart failed: {proc.stderr.strip()}")
            return False
    except Exception as e:
        cooldown_file.touch()
        logger.error(f"Restart exception: {e}")
        return False


# ---------------------------------------------------------------------------
# Telegram alerts
# ---------------------------------------------------------------------------


def _should_alert(alert_key: str, level: str) -> bool:
    """Check if enough time has passed since the last alert for this key."""
    COOLDOWN_DIR.mkdir(parents=True, exist_ok=True)
    cooldown_file = COOLDOWN_DIR / alert_key
    cooldown = COOLDOWN_SECONDS.get(level, COOLDOWN_SECONDS["WARNING"])

    if cooldown_file.exists():
        age = time.time() - cooldown_file.stat().st_mtime
        if age < cooldown:
            return False

    return True


def _mark_alerted(alert_key: str):
    """Update the cooldown timestamp for an alert key."""
    COOLDOWN_DIR.mkdir(parents=True, exist_ok=True)
    (COOLDOWN_DIR / alert_key).touch()


def send_alert(message: str, level: str, alert_key: str, logger: logging.Logger) -> bool:
    """Send a Telegram alert if cooldown allows it."""
    if not _should_alert(alert_key, level):
        logger.debug(f"Alert suppressed (cooldown): {alert_key}")
        return False

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        logger.warning("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set — skipping alert")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = json.dumps(
        {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown",
        }
    ).encode()

    try:
        req = urllib.request.Request(
            url,
            data=payload,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            if result.get("ok"):
                _mark_alerted(alert_key)
                logger.info(f"Telegram alert sent: {alert_key}")
                return True
            else:
                logger.error(f"Telegram API error: {result}")
                return False
    except Exception as e:
        logger.error(f"Telegram send failed: {e} — writing to fallback log")
        # Fallback: write to file so alert isn't silently dropped
        try:
            import datetime as dt_module

            fallback_path = Path("/tmp/aria-missed-alerts.jsonl")
            fallback_entry = {
                "timestamp": dt_module.datetime.now(dt_module.UTC).isoformat(),
                "level": level,
                "alert_key": alert_key,
                "message": message,
                "error": str(e),
            }
            with open(fallback_path, "a") as f:
                f.write(json.dumps(fallback_entry) + "\n")
            logger.info(f"Fallback alert logged to {fallback_path}")
        except Exception as fallback_error:
            logger.error(f"Fallback alert log also failed: {fallback_error}")
        return False


def _recovery_needed() -> bool:
    """Check if there was a previous failure (any cooldown file exists with recent alert)."""
    if not COOLDOWN_DIR.exists():
        return False
    return any(f.name.startswith("alert-") for f in COOLDOWN_DIR.iterdir())


def _clear_recovery():
    """Remove alert cooldown files after successful recovery."""
    if not COOLDOWN_DIR.exists():
        return
    for f in COOLDOWN_DIR.iterdir():
        if f.name.startswith("alert-"):
            f.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _collect_results() -> list:
    """Run all watchdog checks and return combined results."""
    all_results = []
    all_results.extend(check_service_status())
    all_results.extend(check_hub_health())

    # Only run API/cache checks if hub is reachable
    hub_up = any(r.check_name == "hub-liveness" and r.level == "OK" for r in all_results)
    if hub_up:
        all_results.extend(check_api_endpoints())
        all_results.extend(check_cache_freshness())

    all_results.extend(check_timer_health())
    all_results.append(
        check_audit_alerts(
            audit_db_path=os.path.expanduser("~/ha-logs/intelligence/cache/audit.db"),
        )
    )
    all_results.append(check_disk_space())
    all_results.append(check_ollama_health())
    return all_results


def _log_results(logger, passed, total, failed):
    """Log watchdog summary and failures."""
    if failed:
        failed_names = ", ".join(r.check_name for r in failed)
        logger.warning(f"Watchdog complete: {passed}/{total} passed, {len(failed)} failed: {failed_names}")
        for r in failed:
            logger.warning(f"{r.check_name}: {r.message}")
            if r.details:
                logger.debug(f"  details: {json.dumps(r.details)}")
    else:
        logger.info(f"Watchdog complete: {passed}/{total} passed")


def _send_alerts(logger, summary, restart_result):
    """Send Telegram alerts for failures or recovery."""
    passed, total, failed, critical = summary["passed"], summary["total"], summary["failed"], summary["critical"]
    if critical:
        lines = ["*ARIA Watchdog* \\[CRITICAL]", ""]
        for r in critical:
            lines.append(f"• {r.message}")
        if restart_result is not None:
            lines.append("")
            lines.append(f"Auto-restart attempted: {'success' if restart_result else 'failed'}")
        lines.append("")
        lines.append(f"Checks: {passed}/{total} passed")
        if failed:
            lines.append(f"Failed: {', '.join(r.check_name for r in failed)}")
        send_alert("\n".join(lines), "CRITICAL", "alert-critical", logger)

    elif failed:
        lines = ["*ARIA Watchdog* \\[WARNING]", ""]
        for r in failed:
            lines.append(f"• {r.message}")
        lines.append("")
        lines.append(f"Checks: {passed}/{total} passed")
        send_alert("\n".join(lines), "WARNING", "alert-warning", logger)

    elif _recovery_needed():
        send_alert(
            f"*ARIA Watchdog* \\[RECOVERY]\n\nAll {total} checks passing.",
            "WARNING",
            "alert-recovery",
            logger,
        )
        _clear_recovery()


def run_watchdog(quiet: bool = False, no_alert: bool = False, json_output: bool = False) -> int:
    """Run all watchdog checks. Returns 0 if all pass, 1 if any fail."""
    logger = setup_logging()

    # Verify Telegram connectivity at startup (non-blocking — warn, don't fail)
    verify_telegram_connectivity()

    all_results = _collect_results()

    # Summarize
    passed = sum(1 for r in all_results if r.level == "OK")
    total = len(all_results)
    failed = [r for r in all_results if r.level != "OK"]
    critical = [r for r in failed if r.level == "CRITICAL"]

    # Auto-restart if hub is down
    restart_result = None
    if any(r.check_name == "service-hub" and r.level == "CRITICAL" for r in all_results):
        restart_result = attempt_restart(logger)

    _log_results(logger, passed, total, failed)

    if not no_alert:
        summary = {"passed": passed, "total": total, "failed": failed, "critical": critical}
        _send_alerts(logger, summary, restart_result)

    # Output
    if json_output:
        output = {
            "timestamp": datetime.now(UTC).isoformat(),
            "passed": passed,
            "total": total,
            "results": [asdict(r) for r in all_results],
        }
        if restart_result is not None:
            output["restart_attempted"] = True
            output["restart_success"] = restart_result
        print(json.dumps(output, indent=2))
    elif not quiet:
        _print_report(all_results, passed, total, restart_result)

    return 0 if not failed else 1


def _print_report(results: list, passed: int, total: int, restart_result):
    """Print a human-readable report to stdout."""
    print("ARIA Watchdog Report")
    print("=" * 50)
    print(f"  Time:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Checks:  {passed}/{total} passed")
    print()

    # Group by level
    for level in ("CRITICAL", "WARNING", "OK"):
        level_results = [r for r in results if r.level == level]
        if not level_results:
            continue

        if level == "OK":
            print(f"  Passing ({len(level_results)}):")
        else:
            print(f"  {level} ({len(level_results)}):")

        for r in level_results:
            marker = "x" if level != "OK" else "+"
            print(f"    [{marker}] {r.check_name}: {r.message}")

        print()

    if restart_result is not None:
        status = "succeeded" if restart_result else "failed"
        print(f"  Auto-restart: {status}")
        print()
