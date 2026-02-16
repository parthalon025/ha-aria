"""Naming backends for discovered entity clusters.

Provides both rule-based heuristic naming and LLM-powered naming via local
Ollama. The LLM backend falls back to heuristic on any failure.
"""

from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)


def _classify_time(peak_hours: list[int]) -> str | None:
    """Map peak hours to a time-of-day label.

    Buckets:
        morning   = 5-11
        afternoon = 12-16
        evening   = 17-21
        night     = 22-23, 0-4
    """
    if not peak_hours:
        return None

    buckets = {"morning": 0, "afternoon": 0, "evening": 0, "night": 0}
    for h in peak_hours:
        if 5 <= h <= 11:
            buckets["morning"] += 1
        elif 12 <= h <= 16:
            buckets["afternoon"] += 1
        elif 17 <= h <= 21:
            buckets["evening"] += 1
        else:
            buckets["night"] += 1

    return max(buckets, key=buckets.get)


def _dominant(mapping: dict[str, int], threshold: float) -> str | None:
    """Return the key with the highest count if it meets the threshold fraction."""
    if not mapping:
        return None
    total = sum(mapping.values())
    if total == 0:
        return None
    best_key = max(mapping, key=mapping.get)
    if mapping[best_key] / total >= threshold:
        return best_key
    return None


def _to_snake(text: str) -> str:
    """Normalize a string to snake_case (lowercase, spaces/hyphens to underscores, strip non-alnum)."""
    text = text.lower().strip()
    text = re.sub(r"[\s\-]+", "_", text)
    text = re.sub(r"[^a-z0-9_]", "", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def heuristic_name(cluster_info: dict) -> str:
    """Generate a snake_case name from cluster metadata.

    cluster_info keys:
        entity_ids           — list of entity ID strings
        domains              — dict mapping domain name to count
        areas                — dict mapping area name to count
        device_classes       — (optional) dict mapping device class to count
        temporal_pattern     — (optional) dict with peak_hours list and weekday_bias float

    Naming rules (applied in order, parts joined with underscore):
        1. Time prefix if temporal_pattern has peak_hours
        2. Dominant area (>= 60% of entities in one area)
        3. Device class if present (more specific than domain)
        4. Dominant domain (>= 50%) if no device class
        5. "mixed" if no single domain >= 50%
        6. "cluster" as absolute fallback
    """
    parts: list[str] = []

    # 1. Time prefix
    temporal = cluster_info.get("temporal_pattern")
    if temporal and temporal.get("peak_hours"):
        time_label = _classify_time(temporal["peak_hours"])
        if time_label:
            parts.append(time_label)

    # 2. Dominant area (>= 60%)
    areas = cluster_info.get("areas", {})
    dominant_area = _dominant(areas, 0.6)
    if dominant_area:
        parts.append(_to_snake(dominant_area))

    # 3. Device class or domain
    device_classes = cluster_info.get("device_classes", {})
    domains = cluster_info.get("domains", {})

    dominant_dc = _dominant(device_classes, 0.0) if device_classes else None
    if dominant_dc:
        parts.append(_to_snake(dominant_dc))
    else:
        dominant_domain = _dominant(domains, 0.5)
        if dominant_domain:
            parts.append(_to_snake(dominant_domain))
        elif domains:
            parts.append("mixed")

    # 6. Absolute fallback
    if not parts:
        return "cluster"

    return "_".join(parts)


def heuristic_description(cluster_info: dict) -> str:
    """Generate a human-readable description of a cluster.

    Includes: entity count, domain breakdown, areas, and peak hours if temporal.
    """
    entity_ids = cluster_info.get("entity_ids", [])
    domains = cluster_info.get("domains", {})
    areas = cluster_info.get("areas", {})
    temporal = cluster_info.get("temporal_pattern")

    count = len(entity_ids)
    sentences: list[str] = []

    # Entity count
    sentences.append(f"{count} entities")

    # Domain breakdown
    if domains:
        domain_parts = [f"{d} ({c})" for d, c in sorted(domains.items(), key=lambda x: -x[1])]
        sentences.append("domains: " + ", ".join(domain_parts))

    # Areas
    if areas:
        area_parts = sorted(areas.keys())
        sentences.append("areas: " + ", ".join(area_parts))

    # Temporal info
    if temporal and temporal.get("peak_hours"):
        hours = sorted(temporal["peak_hours"])
        hour_strs = [f"{h}:00" for h in hours]
        sentences.append("peak hours: " + ", ".join(hour_strs))

    return ". ".join(sentences) + "."


# ---------------------------------------------------------------------------
# LLM naming via local Ollama
# ---------------------------------------------------------------------------


async def _call_ollama(prompt: str, model: str = "deepseek-r1:8b") -> str:
    """Call Ollama API directly for real-time LLM inference.

    Routes through ollama-queue proxy (port 7683) to serialize GPU access.
    """
    import aiohttp

    async with aiohttp.ClientSession() as session, session.post(
        "http://127.0.0.1:7683/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=aiohttp.ClientTimeout(total=120),
    ) as resp:
        data = await resp.json()
        return data.get("response", "").strip()


async def ollama_name(cluster_info: dict) -> str:
    """Generate a cluster name using local Ollama LLM. Falls back to heuristic."""
    entities = cluster_info.get("entity_ids", [])[:10]
    domains = cluster_info.get("domains", {})
    areas = cluster_info.get("areas", {})
    temporal = cluster_info.get("temporal_pattern", {})

    prompt = (
        "Name this group of Home Assistant entities in 2-4 words. "
        "Return ONLY the name in snake_case, nothing else.\n\n"
        f"Entities: {', '.join(entities)}\n"
        f"Domains: {json.dumps(domains)}\n"
        f"Areas: {json.dumps(areas)}\n"
        f"Temporal: {json.dumps(temporal)}\n"
    )

    try:
        result = await _call_ollama(prompt)
        name = result.strip().lower().replace(" ", "_").replace("-", "_")
        # Remove any non-alphanumeric/underscore chars
        name = "".join(c for c in name if c.isalnum() or c == "_")
        # Strip leading/trailing underscores
        name = name.strip("_")
        if name and len(name) > 2:
            return name
    except Exception as e:
        logger.warning(f"Ollama naming failed, falling back to heuristic: {e}")

    return heuristic_name(cluster_info)


async def ollama_description(cluster_info: dict) -> str:
    """Generate a description using local Ollama LLM. Falls back to heuristic."""
    entities = cluster_info.get("entity_ids", [])[:10]
    domains = cluster_info.get("domains", {})
    areas = cluster_info.get("areas", {})

    prompt = (
        "Describe this group of Home Assistant entities in one sentence. "
        "Explain what they have in common and why they might be grouped.\n\n"
        f"Entities: {', '.join(entities)}\n"
        f"Domains: {json.dumps(domains)}\n"
        f"Areas: {json.dumps(areas)}\n"
    )

    try:
        result = await _call_ollama(prompt)
        if result and len(result) > 10:
            return result[:200]
    except Exception as e:
        logger.warning(f"Ollama description failed: {e}")

    return heuristic_description(cluster_info)
