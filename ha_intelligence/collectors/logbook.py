"""Logbook summarization for HA intelligence snapshots."""

# Clock sensors to exclude from "useful events" count
CLOCK_SENSORS = {
    "sensor.date_time_utc", "sensor.date_time_iso", "sensor.time_date",
    "sensor.time_utc", "sensor.time", "sensor.date_time",
}


def summarize_logbook(entries: list) -> dict:
    """Summarize logbook entries into counts by domain and hour."""
    total = len(entries)
    useful = 0
    by_domain = {}
    hourly = {}
    for e in entries:
        eid = e.get("entity_id", "")
        domain = eid.split(".")[0] if "." in eid else e.get("domain", "unknown")
        by_domain[domain] = by_domain.get(domain, 0) + 1
        if eid not in CLOCK_SENSORS:
            useful += 1
        when = e.get("when", "")
        if len(when) >= 13:
            h = when[11:13]
            hourly[h] = hourly.get(h, 0) + 1
    return {
        "total_events": total,
        "useful_events": useful,
        "by_domain": by_domain,
        "hourly": hourly,
    }
