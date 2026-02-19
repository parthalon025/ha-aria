"""Intelligence Module - Reads ARIA engine output files.

Assembles predictions, baselines, trends, insights, run history, and config
from ~/ha-logs/intelligence/ into a single consolidated cache category.
"""

import asyncio
import json
import logging
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import aiohttp

from aria.capabilities import Capability
from aria.engine.schema import validate_snapshot_schema
from aria.hub.constants import CACHE_ACTIVITY_LOG, CACHE_ACTIVITY_SUMMARY, CACHE_INTELLIGENCE
from aria.hub.core import IntelligenceHub, Module
from aria.schemas import validate_intelligence_payload

logger = logging.getLogger(__name__)


def compare_model_accuracy(
    primary_acc: list,
    reference_acc: list,
    threshold_pct: float = 5.0,
) -> dict:
    """Compare accuracy trends between primary and reference models.

    Distinguishes meta-learner errors from genuine behavioral drift by
    comparing how accuracy evolves in the meta-learner-tuned primary model
    vs a clean reference model trained without meta-learner modifications.

    Args:
        primary_acc: List of accuracy values over time for the primary model.
        reference_acc: List of accuracy values over time for the reference model.
        threshold_pct: Minimum delta magnitude to count as degradation.

    Returns:
        Dict with primary_trend, reference_trend, divergence_pct, and
        interpretation (one of: behavioral_drift, meta_learner_error,
        meta_learner_improvement, stable).
    """
    # Use mean-of-halves for robust trend detection (endpoint-only comparison
    # masks mid-series dips like [80, 60, 80] → delta=0)
    if len(primary_acc) >= 2:
        mid = len(primary_acc) // 2
        primary_delta = (sum(primary_acc[mid:]) / len(primary_acc[mid:])) - (sum(primary_acc[:mid]) / mid)
    else:
        primary_delta = 0.0
    if len(reference_acc) >= 2:
        mid = len(reference_acc) // 2
        reference_delta = (sum(reference_acc[mid:]) / len(reference_acc[mid:])) - (sum(reference_acc[:mid]) / mid)
    else:
        reference_delta = 0.0

    primary_degraded = primary_delta < -threshold_pct
    reference_degraded = reference_delta < -threshold_pct

    if primary_degraded and reference_degraded:
        interpretation = "behavioral_drift"
    elif primary_degraded and not reference_degraded:
        interpretation = "meta_learner_error"
    elif not primary_degraded and reference_degraded:
        interpretation = "meta_learner_improvement"
    else:
        interpretation = "stable"

    return {
        "primary_trend": round(primary_delta, 2),
        "reference_trend": round(reference_delta, 2),
        "divergence_pct": round(abs(primary_delta - reference_delta), 2),
        "interpretation": interpretation,
    }


# Metrics to extract from daily/intraday snapshots
METRIC_PATHS = {
    "power_watts": lambda d: d.get("power", {}).get("total_watts"),
    "lights_on": lambda d: d.get("lights", {}).get("on"),
    "devices_home": lambda d: d.get("occupancy", {}).get("device_count_home"),
    "unavailable": lambda d: d.get("entities", {}).get("unavailable"),
    "useful_events": lambda d: d.get("logbook_summary", {}).get("useful_events"),
}


class IntelligenceModule(Module):
    """Reads ARIA engine output files and pushes to hub cache."""

    CAPABILITIES = [
        Capability(
            id="intelligence_assembly",
            name="Intelligence Assembly",
            description="Assembles engine outputs (snapshots, baselines, predictions, ML scores) into unified cache.",
            module="intelligence",
            layer="hub",
            config_keys=[],
            test_paths=["tests/hub/test_intelligence.py"],
            data_paths=["~/ha-logs/intelligence/"],
            systemd_units=["aria-hub.service"],
            status="stable",
            added_version="1.0.0",
            depends_on=[],
        ),
    ]

    def __init__(self, hub: IntelligenceHub, intelligence_dir: str):
        super().__init__("intelligence", hub)
        self.intel_dir = Path(intelligence_dir)
        self.log_path = Path.home() / ".local" / "log" / "aria.log"
        self._last_digest_date: str | None = None
        self._telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN")
        self._telegram_chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    async def initialize(self):
        """Read all intelligence files and push to cache."""
        self.logger.info(f"Intelligence module initializing (dir: {self.intel_dir})")
        try:
            data = await asyncio.to_thread(self._read_intelligence_data)
            data["activity"] = await self._read_activity_data()
            await self.hub.set_cache(
                CACHE_INTELLIGENCE,
                data,
                {
                    "source": "intelligence_module",
                    "file_count": self._count_source_files(),
                },
            )
            self.logger.info("Intelligence data loaded into cache")

            # Track current insight date so we don't re-send on first refresh
            insight = data.get("daily_insight")
            if insight and insight.get("date"):
                self._last_digest_date = insight["date"]
        except Exception as e:
            self.logger.warning(f"Initial intelligence load failed: {e}")

    async def schedule_refresh(self):
        """Schedule periodic refresh every 15 minutes."""

        async def refresh():
            try:
                data = await asyncio.to_thread(self._read_intelligence_data)
                data["activity"] = await self._read_activity_data()
                await self.hub.set_cache(
                    CACHE_INTELLIGENCE,
                    data,
                    {
                        "source": "intelligence_module",
                        "file_count": self._count_source_files(),
                    },
                )
                self.logger.debug("Intelligence cache refreshed")

                # Check for behavioral drift → notify organic discovery
                await self._check_for_drift(data)

                # Check for new daily insight → send digest
                await self._maybe_send_digest(data)
            except Exception as e:
                self.logger.warning(f"Intelligence refresh failed: {e}")

        await self.hub.schedule_task(
            task_id="intelligence_refresh",
            coro=refresh,
            interval=timedelta(minutes=15),
            run_immediately=False,
        )
        self.logger.info("Scheduled intelligence refresh every 15 minutes")

    async def on_event(self, event_type: str, data: dict[str, Any]):
        pass

    async def _check_for_drift(self, data: dict[str, Any]):
        """Check intelligence data for behavioral drift and publish events.

        Reads drift_status and accuracy from the assembled intelligence data.
        If drift_status contains entries with interpretation == 'behavioral_drift',
        publishes drift_detected events so organic discovery can flag capabilities
        for re-evaluation.
        """
        drift_status = data.get("drift_status")
        if not drift_status:
            return

        # drift_status can be a dict keyed by capability name or a list
        entries = []
        if isinstance(drift_status, dict):
            for cap_name, entry in drift_status.items():
                if isinstance(entry, dict):
                    entry["_cap_name"] = cap_name
                    entries.append(entry)
        elif isinstance(drift_status, list):
            entries = drift_status

        for entry in entries:
            interpretation = entry.get("interpretation", "")
            if interpretation == "behavioral_drift":
                cap_name = entry.get("_cap_name") or entry.get("capability", "")
                if not cap_name:
                    continue
                divergence = abs(entry.get("divergence_pct", 0))
                await self.hub.publish(
                    "drift_detected",
                    {
                        "capability": cap_name,
                        "drift_type": "behavioral_drift",
                        "severity": divergence / 100,
                    },
                )

    # ------------------------------------------------------------------
    # Data assembly
    # ------------------------------------------------------------------

    async def _read_activity_data(self) -> dict[str, Any]:
        """Read activity_log and activity_summary from hub cache."""
        activity_log = await self.hub.get_cache(CACHE_ACTIVITY_LOG)
        activity_summary = await self.hub.get_cache(CACHE_ACTIVITY_SUMMARY)
        return {
            "activity_log": activity_log["data"] if activity_log else None,
            "activity_summary": activity_summary["data"] if activity_summary else None,
        }

    def _build_data_maturity(self) -> tuple[dict[str, Any], list]:
        """Build data maturity section and return daily files for trend extraction."""
        daily_dir = self.intel_dir / "daily"
        intraday_dir = self.intel_dir / "intraday"

        daily_files = sorted(daily_dir.glob("*.json")) if daily_dir.exists() else []
        first_date = daily_files[0].stem if daily_files else None
        days_of_data = len(daily_files)

        intraday_count = 0
        if intraday_dir.exists():
            for date_dir in intraday_dir.iterdir():
                if date_dir.is_dir():
                    intraday_count += len(list(date_dir.glob("*.json")))

        ml_active = (self.intel_dir / "models" / "training_log.json").exists()
        meta_active = (self.intel_dir / "meta-learning" / "applied.json").exists()
        phase, next_milestone = self._determine_phase(days_of_data, ml_active, meta_active)

        phase_descriptions = {
            "collecting": (
                "The system is still learning your home's patterns. "
                "It needs at least 7 days of data to build reliable baselines. "
                "Snapshots are collected automatically every 4 hours."
            ),
            "baselines": (
                "Enough data exists to calculate day-of-week baselines. "
                "Statistical predictions are active. ML models will activate "
                "after 14 days of data."
            ),
            "ml-training": (
                "ML models are being trained on your home's data. "
                "Predictions now blend statistical baselines with machine learning. "
                "Meta-learning will activate to auto-tune the system."
            ),
            "ml-active": (
                "Full intelligence is active: statistical baselines, ML predictions, "
                "and meta-learning are all running. The system continuously improves "
                "its understanding of your home."
            ),
        }

        maturity = {
            "first_date": first_date,
            "days_of_data": days_of_data,
            "intraday_count": intraday_count,
            "ml_active": ml_active,
            "meta_learning_active": meta_active,
            "phase": phase,
            "next_milestone": next_milestone,
            "description": phase_descriptions.get(phase, ""),
        }
        return maturity, daily_files

    def _read_intelligence_data(self) -> dict[str, Any]:
        """Assemble the full intelligence payload from disk files."""
        insights_dir = self.intel_dir / "insights"
        maturity, daily_files = self._build_data_maturity()

        data = {
            "data_maturity": maturity,
            "predictions": self._read_json(self.intel_dir / "predictions.json"),
            "baselines": self._read_json(self.intel_dir / "baselines.json"),
            "trend_data": self._extract_trend_data(daily_files),
            "intraday_trend": self._extract_intraday_trend(),
            "daily_insight": self._read_latest_insight(insights_dir),
            "accuracy": self._read_json(self.intel_dir / "accuracy.json"),
            "correlations": self._read_json(self.intel_dir / "correlations.json") or [],
            "ml_models": self._read_ml_models(),
            "meta_learning": self._read_meta_learning(),
            "run_log": self._build_run_log(),
            "config": self._read_config(),
            "entity_correlations": self._read_json(self.intel_dir / "entity_correlations.json"),
            "sequence_anomalies": self._read_json(self.intel_dir / "sequence_anomalies.json"),
            "power_profiles": self._read_json(self.intel_dir / "insights" / "power-profiles.json"),
            "automation_suggestions": self._read_latest_automation_suggestion(),
            "drift_status": self._read_json(self.intel_dir / "drift_status.json"),
            "feature_selection": self._read_json(self.intel_dir / "feature_selection.json"),
            "reference_model": self._read_json(self.intel_dir / "reference_model.json"),
            "shap_attributions": self._read_json(self.intel_dir / "shap_attributions.json"),
            "autoencoder_status": self._read_json(self.intel_dir / "models" / "autoencoder_status.json"),
            "isolation_forest_status": self._read_json(self.intel_dir / "models" / "isolation_forest_status.json"),
        }

        # Validate schema contract — warn on missing keys, don't crash
        missing = validate_intelligence_payload(data)
        if missing:
            self.logger.warning(f"Intelligence payload missing required keys: {missing}")

        return data

    def _determine_phase(self, days: int, ml_active: bool, meta_active: bool) -> tuple:
        """Return (phase_name, next_milestone_text)."""
        if ml_active and meta_active:
            return ("ml-active", "System is fully operational")
        if ml_active:
            return ("ml-training", "Meta-learning activates after weekly review cycle")
        if days >= 7:
            return ("baselines", "ML models activate after 14 days of data")
        return ("collecting", f"{7 - days} more days needed for reliable baselines")

    def _extract_trend_data(self, daily_files: list[Path]) -> list[dict[str, Any]]:
        """Extract key metrics from each daily snapshot (compact)."""
        trends = []
        for f in daily_files[-30:]:  # last 30 days max
            try:
                raw = json.loads(f.read_text())
                # Validate snapshot schema — fail loudly on mismatch
                schema_errors = validate_snapshot_schema(raw)
                if schema_errors:
                    self.logger.warning(f"Snapshot schema mismatch in {f.name}: {schema_errors}")
                    continue
                entry = {"date": f.stem}
                for key, extractor in METRIC_PATHS.items():
                    val = extractor(raw)
                    if val is not None:
                        entry[key] = round(val, 1) if isinstance(val, float) else val
                trends.append(entry)
            except Exception as e:
                self.logger.debug(f"Skipping daily file {f.name}: {e}")
                continue
        return trends

    def _extract_intraday_trend(self) -> list[dict[str, Any]]:
        """Extract today's intraday snapshots as compact trend entries."""
        today = datetime.now().strftime("%Y-%m-%d")
        intraday_dir = self.intel_dir / "intraday" / today
        if not intraday_dir.exists():
            return []

        entries = []
        for f in sorted(intraday_dir.glob("*.json")):
            try:
                raw = json.loads(f.read_text())
                # Validate snapshot schema — fail loudly on mismatch
                schema_errors = validate_snapshot_schema(raw)
                if schema_errors:
                    self.logger.warning(f"Snapshot schema mismatch in {f.name}: {schema_errors}")
                    continue
                hour = raw.get("hour", int(f.stem))
                entry = {"hour": hour}
                for key, extractor in METRIC_PATHS.items():
                    val = extractor(raw)
                    if val is not None:
                        entry[key] = round(val, 1) if isinstance(val, float) else val
                entries.append(entry)
            except Exception as e:
                self.logger.debug(f"Skipping intraday file {f.name}: {e}")
                continue
        return entries

    def _read_latest_insight(self, insights_dir: Path) -> dict[str, Any] | None:
        """Read the most recent insight report."""
        if not insights_dir.exists():
            return None
        import re

        files = sorted(f for f in insights_dir.glob("*.json") if re.match(r"\d{4}-\d{2}-\d{2}\.json", f.name))
        if not files:
            return None
        try:
            data = json.loads(files[-1].read_text())
            return {"date": data.get("date", files[-1].stem), "report": data.get("report", "")}
        except Exception as e:
            self.logger.debug(f"Failed to read insight {files[-1].name}: {e}")
            return None

    def _read_ml_models(self) -> dict[str, Any]:
        """Read ML model training info."""
        log_path = self.intel_dir / "models" / "training_log.json"
        if not log_path.exists():
            return {"count": 0, "last_trained": None, "scores": {}}
        try:
            data = json.loads(log_path.read_text())
            return {
                "count": len(data) if isinstance(data, list) else data.get("count", 0),
                "last_trained": data[-1].get("timestamp")
                if isinstance(data, list) and data
                else data.get("last_trained"),
                "scores": data[-1].get("scores", {}) if isinstance(data, list) and data else data.get("scores", {}),
            }
        except Exception as e:
            self.logger.warning("Failed to read ML model training log: %s", e)
            return {"count": 0, "last_trained": None, "scores": {}}

    def _read_meta_learning(self) -> dict[str, Any]:
        """Read meta-learning applied suggestions."""
        applied_path = self.intel_dir / "meta-learning" / "applied.json"
        if not applied_path.exists():
            return {"applied_count": 0, "last_applied": None, "suggestions": []}
        try:
            data = json.loads(applied_path.read_text())
            if isinstance(data, list):
                return {
                    "applied_count": len(data),
                    "last_applied": data[-1].get("timestamp") if data else None,
                    "suggestions": data[-5:],  # last 5
                }
            return {
                "applied_count": data.get("applied_count", 0),
                "last_applied": data.get("last_applied"),
                "suggestions": data.get("suggestions", [])[-5:],
            }
        except Exception as e:
            self.logger.warning("Failed to read meta-learning applied suggestions: %s", e)
            return {"applied_count": 0, "last_applied": None, "suggestions": []}

    def _scan_dir_for_runs(self, directory: Path, run_type: str, limit: int = 0) -> list[dict[str, Any]]:
        """Scan a directory for JSON files and build run log entries."""
        if not directory.exists():
            return []
        files = sorted(directory.glob("*.json"), reverse=True)
        if limit:
            files = files[:limit]
        return [
            {
                "timestamp": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                "type": run_type,
                "status": "ok",
            }
            for f in files
        ]

    def _parse_error_log(self) -> list[dict[str, Any]]:
        """Parse recent error lines from aria log file."""
        if not self.log_path.exists():
            return []
        errors = []
        try:
            lines = self.log_path.read_text().splitlines()[-50:]
            for line in lines:
                if "ERROR" in line or "FAILED" in line:
                    ts_match = re.match(r"(\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2})", line)
                    errors.append(
                        {
                            "timestamp": ts_match.group(1) if ts_match else None,
                            "type": "error",
                            "status": "error",
                            "message": line.strip()[:200],
                        }
                    )
        except Exception as e:
            self.logger.warning("Failed to parse aria log file %s: %s", self.log_path, e)
        return errors

    def _build_run_log(self) -> list[dict[str, Any]]:
        """Build run history from file mtimes and log file."""
        today = datetime.now().strftime("%Y-%m-%d")
        runs = []
        runs.extend(self._scan_dir_for_runs(self.intel_dir / "daily", "daily", limit=5))
        runs.extend(self._scan_dir_for_runs(self.intel_dir / "intraday" / today, "intraday"))
        runs.extend(self._scan_dir_for_runs(self.intel_dir / "insights", "full_pipeline", limit=3))
        runs.extend(self._parse_error_log())

        runs.sort(key=lambda r: r.get("timestamp") or "", reverse=True)
        return runs[:15]

    def _read_config(self) -> dict[str, Any]:
        """Read feature config or return defaults."""
        config_path = self.intel_dir / "feature_config.json"
        feature_config = self._read_json(config_path)

        return {
            "anomaly_threshold": 2.0,
            "ml_weight_schedule": {
                "<60d": 0.3,
                "<90d": 0.5,
                ">=90d": 0.7,
            },
            "feature_config": feature_config
            or {
                "time_features": True,
                "weather_features": True,
                "home_state_features": True,
                "lag_features": True,
                "interaction_features": False,
            },
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _read_latest_automation_suggestion(self) -> dict[str, Any] | None:
        """Read the most recent automation suggestion file."""
        suggestions_dir = self.intel_dir / "insights" / "automation-suggestions"
        if not suggestions_dir.exists():
            return None
        files = sorted(suggestions_dir.glob("*.json"))
        if not files:
            return None
        try:
            return json.loads(files[-1].read_text())
        except Exception as e:
            self.logger.debug(f"Failed to read automation suggestion: {e}")
            return None

    def _read_json(self, path: Path) -> Any:
        """Read a JSON file, returning None if missing or invalid."""
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except Exception as e:
            self.logger.debug(f"Failed to read {path}: {e}")
            return None

    def _count_source_files(self) -> int:
        """Count total intelligence files on disk."""
        count = 0
        for pattern in [
            "*.json",
            "daily/*.json",
            "intraday/*/*.json",
            "insights/*.json",
            "models/*.json",
            "meta-learning/*.json",
        ]:
            count += len(list(self.intel_dir.glob(pattern)))
        return count

    # ------------------------------------------------------------------
    # Daily digest to Telegram
    # ------------------------------------------------------------------

    async def _maybe_send_digest(self, data: dict[str, Any]):
        """Send a daily digest if there's a new insight we haven't sent yet."""
        if not self._telegram_token or not self._telegram_chat_id:
            return

        insight = data.get("daily_insight")
        if not insight or not insight.get("date"):
            return

        insight_date = insight["date"]
        if insight_date == self._last_digest_date:
            return  # already sent

        # New insight — format and send digest
        try:
            message = self._format_digest(data)
            await self._send_telegram(message)
            self._last_digest_date = insight_date  # mark sent only after successful send
            self.logger.info(f"Daily digest sent for {insight_date}")
        except Exception as e:
            self.logger.warning(f"Failed to send daily digest: {e}")

    @staticmethod
    def _format_intraday_metrics(intraday: list[dict[str, Any]]) -> str | None:
        """Format latest intraday metrics for digest."""
        if not intraday:
            return None
        latest = intraday[-1]
        parts = []
        if "power_watts" in latest:
            parts.append(f"Power: {latest['power_watts']}W")
        if "lights_on" in latest:
            parts.append(f"Lights: {latest['lights_on']}")
        if "devices_home" in latest:
            parts.append(f"Devices: {latest['devices_home']}")
        return f"Now: {' | '.join(parts)}" if parts else None

    @staticmethod
    def _format_trend_deltas(trend: list[dict[str, Any]]) -> str | None:
        """Format today-vs-yesterday trend deltas for digest."""
        if len(trend) < 2:
            return None
        today_t, yesterday_t = trend[-1], trend[-2]
        deltas = []
        for key in ["power_watts", "lights_on", "devices_home", "unavailable"]:
            t_val, y_val = today_t.get(key), yesterday_t.get(key)
            if t_val is not None and y_val is not None:
                diff = t_val - y_val
                sign = "+" if diff > 0 else ""
                deltas.append(f"{key}: {sign}{diff:.0f}")
        return f"vs yesterday: {', '.join(deltas)}" if deltas else None

    def _format_digest(self, data: dict[str, Any]) -> str:
        """Format intelligence data into a Telegram-friendly digest."""
        maturity = data.get("data_maturity", {})
        predictions = data.get("predictions", {})
        insight = data.get("daily_insight", {})

        lines = ["*ARIA Daily Digest*", ""]
        lines.append(f"Phase: *{maturity.get('phase', 'unknown')}* ({maturity.get('days_of_data', 0)} days of data)")

        intraday_line = self._format_intraday_metrics(data.get("intraday_trend", []))
        if intraday_line:
            lines.append(intraday_line)

        trend_line = self._format_trend_deltas(data.get("trend_data", []))
        if trend_line:
            lines.append(trend_line)

        # Predictions summary
        if isinstance(predictions, dict) and predictions.get("power_watts"):
            pred_parts = []
            for key in ["power_watts", "lights_on", "devices_home"]:
                pred = predictions.get(key, {})
                if isinstance(pred, dict) and "predicted" in pred:
                    pred_parts.append(f"{key}: {pred['predicted']} ({pred.get('confidence', '?')})")
            if pred_parts:
                lines.extend(["", "*Predictions:* " + " | ".join(pred_parts)])

        report = (insight.get("report") or "")[:300]
        if report:
            report = re.sub(r"###?\s*", "", report).strip()
            lines.extend(["", f"_Insight:_ {report}"])
            if len(insight.get("report", "")) > 300:
                lines.append("...")

        return "\n".join(lines)

    async def _send_telegram(self, text: str):
        """Send a message via Telegram Bot API (async, non-blocking)."""
        url = f"https://api.telegram.org/bot{self._telegram_token}/sendMessage"
        payload = {
            "chat_id": self._telegram_chat_id,
            "text": text,
            "parse_mode": "Markdown",
        }
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp,
            ):
                result = await resp.json()
                if not result.get("ok"):
                    raise RuntimeError(f"Telegram API error: {result}")
        except aiohttp.ClientError as e:
            raise RuntimeError(f"Telegram request failed: {e}") from e
