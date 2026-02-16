"""HTTP API functions for fetching data from Home Assistant and external services."""

import json
import logging
import re
import subprocess
import urllib.error
import urllib.request

from aria.engine.config import HAConfig, WeatherConfig

logger = logging.getLogger(__name__)


def fetch_ha_states(ha_config: HAConfig) -> list[dict]:
    """Fetch all entity states from HA REST API."""
    if not ha_config.token:
        return []
    try:
        req = urllib.request.Request(
            f"{ha_config.url}/api/states",
            headers={"Authorization": f"Bearer {ha_config.token}"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except Exception as e:
        logger.warning("Failed to fetch HA states from %s: %s", ha_config.url, e)
        return []


def fetch_weather(weather_config: WeatherConfig) -> str:
    """Fetch weather from wttr.in, return raw string."""
    try:
        url = f"https://wttr.in/{weather_config.location}?format=%C+%t+%h+%w"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.read().decode("utf-8", errors="replace").strip()
    except Exception as e:
        logger.warning("Failed to fetch weather from wttr.in: %s", e)
        return ""


def parse_weather(raw: str) -> dict:
    """Parse wttr.in compact format into structured dict."""
    result = {"raw": raw, "condition": "", "temp_f": None, "humidity_pct": None, "wind_mph": None}
    if not raw:
        return result
    m = re.search(r"([+-]?\d+)\s*°F", raw)
    if m:
        result["temp_f"] = int(m.group(1))
    m = re.search(r"(\d+)%", raw)
    if m:
        result["humidity_pct"] = int(m.group(1))
    m = re.search(r"[→←↑↓↗↘↙↖]?\s*(\d+)\s*mph", raw)
    if m:
        result["wind_mph"] = int(m.group(1))
    m = re.match(r"^(.+?)\s*[+-]?\d+°", raw)
    if m:
        result["condition"] = m.group(1).strip()
    return result


def fetch_calendar_events() -> list:
    """Fetch today's calendar events via gog CLI."""
    try:
        result = subprocess.run(
            ["gog", "calendar", "list", "--today", "--all", "--plain"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            logger.warning("Calendar fetch failed (exit %d): %s", result.returncode, result.stderr[:200])
            return []
        lines = result.stdout.strip().split("\n")
        if len(lines) <= 1:
            return []
        events = []
        for line in lines[1:]:
            parts = line.split("\t")
            if len(parts) >= 4:
                events.append({"start": parts[1], "end": parts[2], "summary": parts[3]})
            elif len(parts) >= 5:
                events.append({"start": parts[2], "end": parts[3], "summary": parts[4]})
        return events
    except Exception as e:
        logger.warning("Failed to fetch calendar events: %s", e)
        return []
