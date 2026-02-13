"""Time-based feature engineering — cyclical encoding, sun-relative features, holidays."""

import math
from datetime import datetime

# US holidays (Florida)
try:
    import holidays as holidays_lib
    US_HOLIDAYS = holidays_lib.US(years=range(2025, 2028))
except ImportError:
    US_HOLIDAYS = {}


def _time_to_minutes(time_str):
    """Convert HH:MM string to minutes since midnight."""
    parts = time_str.split(":")
    return int(parts[0]) * 60 + int(parts[1])


def cyclical_encode(value, max_value):
    """Encode a cyclical feature as sin/cos pair."""
    angle = 2 * math.pi * value / max_value
    return round(math.sin(angle), 6), round(math.cos(angle), 6)


def build_time_features(timestamp_str, sun_data=None, date_str=None):
    """Build all time features from a timestamp and sun data."""
    if timestamp_str:
        dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00")) if "T" in timestamp_str else datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
    else:
        dt = datetime.now()

    hour = dt.hour + dt.minute / 60.0
    dow = dt.weekday()
    month = dt.month
    doy = dt.timetuple().tm_yday

    h_sin, h_cos = cyclical_encode(hour, 24)
    d_sin, d_cos = cyclical_encode(dow, 7)
    m_sin, m_cos = cyclical_encode(month, 12)
    y_sin, y_cos = cyclical_encode(doy, 365)

    current_minutes = dt.hour * 60 + dt.minute

    # Sun-relative features
    sunrise_minutes = 360  # default 6:00
    sunset_minutes = 1080  # default 18:00
    if sun_data:
        try:
            sunrise_minutes = _time_to_minutes(sun_data.get("sunrise", "06:00"))
            sunset_minutes = _time_to_minutes(sun_data.get("sunset", "18:00"))
        except Exception:
            pass

    daylight_total = max(1, sunset_minutes - sunrise_minutes)
    is_night = current_minutes < sunrise_minutes or current_minutes > sunset_minutes

    # Holiday check
    check_date = date_str or dt.strftime("%Y-%m-%d")
    is_holiday = check_date in US_HOLIDAYS if US_HOLIDAYS else False

    return {
        "hour": dt.hour,
        "hour_sin": h_sin, "hour_cos": h_cos,
        "dow": dow,
        "dow_sin": d_sin, "dow_cos": d_cos,
        "month": month,
        "month_sin": m_sin, "month_cos": m_cos,
        "day_of_year": doy,
        "day_of_year_sin": y_sin, "day_of_year_cos": y_cos,
        "is_weekend": dow >= 5,
        "is_holiday": is_holiday,
        "is_night": is_night,
        "is_work_hours": not (dow >= 5) and 480 <= current_minutes <= 1020,
        "minutes_since_midnight": current_minutes,
        "minutes_since_sunrise": max(0, current_minutes - sunrise_minutes),
        "minutes_until_sunset": max(0, sunset_minutes - current_minutes),
        "daylight_remaining_pct": round(max(0, (sunset_minutes - current_minutes) / daylight_total * 100), 1) if not is_night else 0,
        "week_of_year": dt.isocalendar()[1],
    }
