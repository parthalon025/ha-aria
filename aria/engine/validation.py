"""Snapshot validation — catches corrupt/incomplete data before model training.

Prevents the scenario where HA restarts mid-snapshot, producing a snapshot with
0 entities or 90% unavailable, which poisons model training for a week.
"""


# Minimum viable snapshot requirements
MIN_ENTITY_COUNT = 100  # HA has ~3050; anything below 100 means HA was down
MAX_UNAVAILABLE_RATIO = 0.5  # >50% unavailable = HA was likely restarting
REQUIRED_SECTIONS = ["date", "entities"]


def validate_snapshot(snapshot: dict) -> list[str]:
    """Validate a single snapshot for training readiness.

    Returns list of error strings (empty = valid).
    """
    errors = []

    # Required fields
    if not snapshot.get("date"):
        errors.append("Missing required field: date")

    # Entity count sanity check
    entities = snapshot.get("entities", {})
    total = entities.get("total", 0)
    unavailable = entities.get("unavailable", 0)

    if total < MIN_ENTITY_COUNT:
        errors.append(
            f"Entities count too low: {total} (min {MIN_ENTITY_COUNT}). HA may have been down during snapshot."
        )

    if total > 0 and unavailable / total > MAX_UNAVAILABLE_RATIO:
        errors.append(
            f"High unavailable ratio: {unavailable}/{total} ({unavailable / total:.0%}). HA may have been restarting."
        )

    return errors


def validate_snapshot_batch(
    snapshots: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Validate a batch of snapshots, separating valid from rejected.

    Returns (valid_snapshots, rejected_snapshots).
    """
    valid = []
    rejected = []

    for snap in snapshots:
        errors = validate_snapshot(snap)
        if errors:
            rejected.append(snap)
        else:
            valid.append(snap)

    return valid, rejected
