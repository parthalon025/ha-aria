"""Intelligence Module - Reads ha-intelligence engine output files.

Assembles predictions, baselines, trends, insights, run history, and config
from ~/ha-logs/intelligence/ into a single consolidated cache category.
"""

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

import aiohttp

from hub.core import Module, IntelligenceHub
from hub.constants import CACHE_ACTIVITY_LOG, CACHE_ACTIVITY_SUMMARY, CACHE_INTELLIGENCE


logger = logging.getLogger(__name__)

# Metrics to extract from daily/intraday snapshots
METRIC_PATHS = {
    "power_watts": lambda d: d.get("power", {}).get("total_watts"),
    "lights_on": lambda d: d.get("lights", {}).get("on"),
    "devices_home": lambda d: d.get("occupancy", {}).get("device_count_home"),
    "unavailable": lambda d: d.get("entities", {}).get("unavailable"),
    "useful_events": lambda d: d.get("logbook_summary", {}).get("useful_events"),
}


class IntelligenceModule(Module):
    """Reads ha-intelligence output files and pushes to hub cache."""

    def __init__(self, hub: IntelligenceHub, intelligence_dir: str):
        super().__init__("intelligence", hub)
        self.intel_dir = Path(intelligence_dir)
        self.log_path = Path.home() / ".local" / "log" / "ha-intelligence.log"
        self._last_digest_date: Optional[str] = None
        self._telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN")
        self._telegram_chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    async def initialize(self):
        """Read all intelligence files and push to cache."""
        self.logger.info(f"Intelligence module initializing (dir: {self.intel_dir})")
        try:
            data = self._read_intelligence_data()
            data["activity"] = await self._read_activity_data()
            await self.hub.set_cache(CACHE_INTELLIGENCE, data, {
                "source": "intelligence_module",
                "file_count": self._count_source_files(),
            })
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
                data = self._read_intelligence_data()
                data["activity"] = await self._read_activity_data()
                await self.hub.set_cache(CACHE_INTELLIGENCE, data, {
                    "source": "intelligence_module",
                    "file_count": self._count_source_files(),
                })
                self.logger.debug("Intelligence cache refreshed")

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

    async def on_event(self, event_type: str, data: Dict[str, Any]):
        pass

    # ------------------------------------------------------------------
    # Data assembly
    # ------------------------------------------------------------------

    async def _read_activity_data(self) -> Dict[str, Any]:
        """Read activity_log and activity_summary from hub cache."""
        activity_log = await self.hub.get_cache(CACHE_ACTIVITY_LOG)
        activity_summary = await self.hub.get_cache(CACHE_ACTIVITY_SUMMARY)
        return {
            "activity_log": activity_log["data"] if activity_log else None,
            "activity_summary": activity_summary["data"] if activity_summary else None,
        }

    def _read_intelligence_data(self) -> Dict[str, Any]:
        """Assemble the full intelligence payload from disk files."""
        daily_dir = self.intel_dir / "daily"
        intraday_dir = self.intel_dir / "intraday"
        insights_dir = self.intel_dir / "insights"

        # Count daily snapshots
        daily_files = sorted(daily_dir.glob("*.json")) if daily_dir.exists() else []
        first_date = daily_files[0].stem if daily_files else None
        days_of_data = len(daily_files)

        # Count intraday snapshots across all date dirs
        intraday_count = 0
        if intraday_dir.exists():
            for date_dir in intraday_dir.iterdir():
                if date_dir.is_dir():
                    intraday_count += len(list(date_dir.glob("*.json")))

        # Determine learning phase
        ml_active = (self.intel_dir / "models" / "training_log.json").exists()
        meta_active = (self.intel_dir / "meta-learning" / "applied.json").exists()
        phase, next_milestone = self._determine_phase(
            days_of_data, ml_active, meta_active
        )

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

        return {
            "data_maturity": {
                "first_date": first_date,
                "days_of_data": days_of_data,
                "intraday_count": intraday_count,
                "ml_active": ml_active,
                "meta_learning_active": meta_active,
                "phase": phase,
                "next_milestone": next_milestone,
                "description": phase_descriptions.get(phase, ""),
            },
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
        }

    def _determine_phase(
        self, days: int, ml_active: bool, meta_active: bool
    ) -> tuple:
        """Return (phase_name, next_milestone_text)."""
        if ml_active and meta_active:
            return ("ml-active", "System is fully operational")
        if ml_active:
            return ("ml-training", "Meta-learning activates after weekly review cycle")
        if days >= 7:
            return ("baselines", "ML models activate after 14 days of data")
        return ("collecting", f"{7 - days} more days needed for reliable baselines")

    def _extract_trend_data(self, daily_files: List[Path]) -> List[Dict[str, Any]]:
        """Extract key metrics from each daily snapshot (compact)."""
        trends = []
        for f in daily_files[-30:]:  # last 30 days max
            try:
                raw = json.loads(f.read_text())
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

    def _extract_intraday_trend(self) -> List[Dict[str, Any]]:
        """Extract today's intraday snapshots as compact trend entries."""
        today = datetime.now().strftime("%Y-%m-%d")
        intraday_dir = self.intel_dir / "intraday" / today
        if not intraday_dir.exists():
            return []

        entries = []
        for f in sorted(intraday_dir.glob("*.json")):
            try:
                raw = json.loads(f.read_text())
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

    def _read_latest_insight(self, insights_dir: Path) -> Optional[Dict[str, Any]]:
        """Read the most recent insight report."""
        if not insights_dir.exists():
            return None
        files = sorted(insights_dir.glob("*.json"))
        if not files:
            return None
        try:
            data = json.loads(files[-1].read_text())
            return {"date": data.get("date", files[-1].stem), "report": data.get("report", "")}
        except Exception as e:
            self.logger.debug(f"Failed to read insight {files[-1].name}: {e}")
            return None

    def _read_ml_models(self) -> Dict[str, Any]:
        """Read ML model training info."""
        log_path = self.intel_dir / "models" / "training_log.json"
        if not log_path.exists():
            return {"count": 0, "last_trained": None, "scores": {}}
        try:
            data = json.loads(log_path.read_text())
            return {
                "count": len(data) if isinstance(data, list) else data.get("count", 0),
                "last_trained": data[-1].get("timestamp") if isinstance(data, list) and data else data.get("last_trained"),
                "scores": data[-1].get("scores", {}) if isinstance(data, list) and data else data.get("scores", {}),
            }
        except Exception:
            return {"count": 0, "last_trained": None, "scores": {}}

    def _read_meta_learning(self) -> Dict[str, Any]:
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
        except Exception:
            return {"applied_count": 0, "last_applied": None, "suggestions": []}

    def _build_run_log(self) -> List[Dict[str, Any]]:
        """Build run history from file mtimes and log file."""
        runs = []

        # Scan daily snapshots
        daily_dir = self.intel_dir / "daily"
        if daily_dir.exists():
            for f in sorted(daily_dir.glob("*.json"), reverse=True)[:5]:
                runs.append({
                    "timestamp": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                    "type": "daily",
                    "status": "ok",
                })

        # Scan today's intraday
        today = datetime.now().strftime("%Y-%m-%d")
        intraday_dir = self.intel_dir / "intraday" / today
        if intraday_dir.exists():
            for f in sorted(intraday_dir.glob("*.json"), reverse=True):
                runs.append({
                    "timestamp": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                    "type": "intraday",
                    "status": "ok",
                })

        # Scan insight reports
        insights_dir = self.intel_dir / "insights"
        if insights_dir.exists():
            for f in sorted(insights_dir.glob("*.json"), reverse=True)[:3]:
                runs.append({
                    "timestamp": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                    "type": "full_pipeline",
                    "status": "ok",
                })

        # Parse log for errors (last 50 lines)
        if self.log_path.exists():
            try:
                lines = self.log_path.read_text().splitlines()[-50:]
                for line in lines:
                    if "ERROR" in line or "FAILED" in line:
                        # Try to extract timestamp from log line
                        ts_match = re.match(r"(\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2})", line)
                        runs.append({
                            "timestamp": ts_match.group(1) if ts_match else None,
                            "type": "error",
                            "status": "error",
                            "message": line.strip()[:200],
                        })
            except Exception:
                pass

        # Sort by timestamp descending, take most recent 15
        runs.sort(key=lambda r: r.get("timestamp") or "", reverse=True)
        return runs[:15]

    def _read_config(self) -> Dict[str, Any]:
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
            "feature_config": feature_config or {
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

    def _read_latest_automation_suggestion(self) -> Optional[Dict[str, Any]]:
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
        for pattern in ["*.json", "daily/*.json", "intraday/*/*.json",
                        "insights/*.json", "models/*.json", "meta-learning/*.json"]:
            count += len(list(self.intel_dir.glob(pattern)))
        return count

    # ------------------------------------------------------------------
    # Daily digest to Telegram
    # ------------------------------------------------------------------

    async def _maybe_send_digest(self, data: Dict[str, Any]):
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
        self._last_digest_date = insight_date
        try:
            message = self._format_digest(data)
            await self._send_telegram(message)
            self.logger.info(f"Daily digest sent for {insight_date}")
        except Exception as e:
            self.logger.warning(f"Failed to send daily digest: {e}")

    def _format_digest(self, data: Dict[str, Any]) -> str:
        """Format intelligence data into a Telegram-friendly digest."""
        maturity = data.get("data_maturity", {})
        trend = data.get("trend_data", [])
        intraday = data.get("intraday_trend", [])
        predictions = data.get("predictions", {})
        insight = data.get("daily_insight", {})

        lines = ["*HA Intelligence Daily Digest*", ""]

        # Phase & maturity
        phase = maturity.get("phase", "unknown")
        days = maturity.get("days_of_data", 0)
        lines.append(f"Phase: *{phase}* ({days} days of data)")

        # Today's latest metrics (from last intraday snapshot)
        if intraday:
            latest = intraday[-1]
            parts = []
            if "power_watts" in latest:
                parts.append(f"Power: {latest['power_watts']}W")
            if "lights_on" in latest:
                parts.append(f"Lights: {latest['lights_on']}")
            if "devices_home" in latest:
                parts.append(f"Devices: {latest['devices_home']}")
            if parts:
                lines.append(f"Now: {' | '.join(parts)}")

        # Trend comparison (today vs yesterday)
        if len(trend) >= 2:
            today_t = trend[-1]
            yesterday_t = trend[-2]
            deltas = []
            for key in ["power_watts", "lights_on", "devices_home", "unavailable"]:
                t_val = today_t.get(key)
                y_val = yesterday_t.get(key)
                if t_val is not None and y_val is not None:
                    diff = t_val - y_val
                    sign = "+" if diff > 0 else ""
                    deltas.append(f"{key}: {sign}{diff:.0f}")
            if deltas:
                lines.append(f"vs yesterday: {', '.join(deltas)}")

        # Predictions summary
        if isinstance(predictions, dict) and predictions.get("power_watts"):
            preds = predictions
            pred_parts = []
            for key in ["power_watts", "lights_on", "devices_home"]:
                pred = preds.get(key, {})
                if isinstance(pred, dict) and "predicted" in pred:
                    conf = pred.get("confidence", "?")
                    pred_parts.append(f"{key}: {pred['predicted']} ({conf})")
            if pred_parts:
                lines.append("")
                lines.append("*Predictions:* " + " | ".join(pred_parts))

        # Insight excerpt (first 300 chars of report)
        report = (insight.get("report") or "")[:300]
        if report:
            # Strip markdown headers for Telegram
            report = re.sub(r"###?\s*", "", report).strip()
            lines.append("")
            lines.append(f"_Insight:_ {report}")
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
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    result = await resp.json()
                    if not result.get("ok"):
                        raise RuntimeError(f"Telegram API error: {result}")
        except aiohttp.ClientError as e:
            raise RuntimeError(f"Telegram request failed: {e}") from e
