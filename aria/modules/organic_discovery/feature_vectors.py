"""Feature vector builder for organic capability discovery.

Transforms HA entity states, device registry, and activity rates into a
numeric matrix suitable for HDBSCAN clustering. Each entity becomes one row.
"""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Capability attribute keys to detect
_CAPABILITY_FLAGS = [
    ("has_brightness", "brightness"),
    ("has_color_temp", "color_temp"),
    ("has_rgb", "rgb_color"),
    ("has_hvac", "hvac_modes"),
    ("has_temperature_target", "temperature"),
]


def _resolve_area(
    entity: dict[str, Any],
    devices: dict[str, dict[str, Any]],
) -> str | None:
    """Resolve area_id: entity direct > device fallback."""
    area = entity.get("area_id")
    if area:
        return area
    device_id = entity.get("device_id")
    if device_id and device_id in devices:
        return devices[device_id].get("area_id")
    return None


def _resolve_manufacturer(
    entity: dict[str, Any],
    devices: dict[str, dict[str, Any]],
) -> str | None:
    """Resolve manufacturer from device registry."""
    device_id = entity.get("device_id")
    if device_id and device_id in devices:
        return devices[device_id].get("manufacturer")
    return None


def _estimate_state_cardinality(entity: dict[str, Any]) -> float:
    """Estimate state cardinality from entity characteristics.

    - binary_sensor domain → 2
    - Entities with unit_of_measurement (numeric sensors) → 100
    - Everything else → 5
    """
    domain = entity.get("domain", "")
    if domain == "binary_sensor":
        return 2.0
    if entity.get("unit_of_measurement"):
        return 100.0
    return 5.0


def _collect_categorical_values(
    entities: list[dict[str, Any]],
    devices: dict[str, dict[str, Any]],
) -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    """Pre-scan entities to collect all unique categorical values.

    Returns sorted lists of unique: domains, device_classes, units, areas, manufacturers.
    """
    domains: set[str] = set()
    device_classes: set[str] = set()
    units: set[str] = set()
    areas: set[str] = set()
    manufacturers: set[str] = set()

    for entity in entities:
        domain = entity.get("domain", "")
        if domain:
            domains.add(domain)

        dc = entity.get("device_class")
        if dc:
            device_classes.add(dc)

        uom = entity.get("unit_of_measurement")
        if uom:
            units.add(uom)

        area = _resolve_area(entity, devices)
        if area:
            areas.add(area)

        mfr = _resolve_manufacturer(entity, devices)
        if mfr:
            manufacturers.add(mfr)

    return (
        sorted(domains),
        sorted(device_classes),
        sorted(units),
        sorted(areas),
        sorted(manufacturers),
    )


def _build_feature_names(
    domains: list[str],
    device_classes: list[str],
    units: list[str],
    areas: list[str],
    manufacturers: list[str],
) -> list[str]:
    """Build ordered feature name list."""
    names: list[str] = []

    # One-hot categoricals
    names.extend(f"domain_{d}" for d in domains)
    names.extend(f"device_class_{dc}" for dc in device_classes)
    names.extend(f"unit_{u}" for u in units)
    names.extend(f"area_{a}" for a in areas)
    names.extend(f"manufacturer_{m}" for m in manufacturers)

    # Scalar features
    names.append("state_cardinality")
    names.append("avg_daily_changes")
    names.append("available")

    # Capability flags
    names.extend(flag_name for flag_name, _ in _CAPABILITY_FLAGS)

    return names


def build_feature_matrix(
    entities: list[dict[str, Any]],
    devices: dict[str, dict[str, Any]],
    entity_registry: dict[str, dict[str, Any]],
    activity_rates: dict[str, float],
) -> tuple[np.ndarray, list[str], list[str]]:
    """Build a numeric feature matrix from HA entity data.

    Each entity becomes one row. Features include domain one-hot, device class
    one-hot, unit one-hot, area one-hot (resolved via device), manufacturer
    one-hot (resolved via device), state cardinality, activity rate, available
    flag, and capability flags.

    Args:
        entities: List of entity dicts from discovery cache.
        devices: Dict mapping device_id → device dict.
        entity_registry: Dict mapping entity_id → registry entry (currently unused,
            reserved for future enrichment).
        activity_rates: Dict mapping entity_id → avg daily state changes.

    Returns:
        Tuple of (matrix, entity_ids, feature_names) where:
        - matrix: np.ndarray of shape (n_entities, n_features)
        - entity_ids: List of entity IDs in row order
        - feature_names: List of feature names in column order
    """
    # Collect categorical values for one-hot encoding
    domains, device_classes, units, areas, manufacturers = _collect_categorical_values(entities, devices)

    # Build feature name index
    feature_names = _build_feature_names(
        domains,
        device_classes,
        units,
        areas,
        manufacturers,
    )

    if not entities:
        return np.empty((0, len(feature_names)), dtype=np.float64), [], feature_names

    # Build name → column index lookup
    col_index = {name: i for i, name in enumerate(feature_names)}
    n_features = len(feature_names)
    n_entities = len(entities)

    # Allocate matrix (zeros — one-hot columns default to 0)
    matrix = np.zeros((n_entities, n_features), dtype=np.float64)

    entity_ids: list[str] = []

    for row, entity in enumerate(entities):
        eid = entity.get("entity_id", "")
        entity_ids.append(eid)
        _fill_entity_row(matrix, row, entity, eid, devices, activity_rates, col_index)

    return matrix, entity_ids, feature_names


def _fill_entity_row(  # noqa: PLR0913 — matrix row builder needs all context
    matrix: np.ndarray,
    row: int,
    entity: dict[str, Any],
    eid: str,
    devices: dict[str, dict[str, Any]],
    activity_rates: dict[str, float],
    col_index: dict[str, int],
) -> None:
    """Fill a single entity row in the feature matrix."""
    # One-hot categorical features
    _set_one_hot(matrix, row, col_index, "domain", entity.get("domain", ""))
    _set_one_hot(matrix, row, col_index, "device_class", entity.get("device_class"))
    _set_one_hot(matrix, row, col_index, "unit", entity.get("unit_of_measurement"))
    _set_one_hot(matrix, row, col_index, "area", _resolve_area(entity, devices))
    _set_one_hot(matrix, row, col_index, "manufacturer", _resolve_manufacturer(entity, devices))

    # Scalar features
    matrix[row, col_index["state_cardinality"]] = _estimate_state_cardinality(entity)
    matrix[row, col_index["avg_daily_changes"]] = activity_rates.get(eid, 0.0)
    state = entity.get("state", "")
    matrix[row, col_index["available"]] = 0.0 if state == "unavailable" else 1.0

    # Capability flags
    attrs = entity.get("attributes", {})
    for flag_name, attr_key in _CAPABILITY_FLAGS:
        if attr_key in attrs:
            matrix[row, col_index[flag_name]] = 1.0


def _set_one_hot(
    matrix: np.ndarray,
    row: int,
    col_index: dict[str, int],
    prefix: str,
    value: str | None,
) -> None:
    """Set a one-hot column in the feature matrix if the value is present."""
    if not value:
        return
    key = f"{prefix}_{value}"
    if key in col_index:
        matrix[row, col_index[key]] = 1.0
