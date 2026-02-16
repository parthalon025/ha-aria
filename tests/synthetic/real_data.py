"""RealDataLoader -- loads actual HA snapshot data for backtesting."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

DEFAULT_INTRADAY_DIR = Path.home() / "ha-logs" / "intelligence" / "intraday"
DEFAULT_DAILY_DIR = Path.home() / "ha-logs" / "intelligence" / "daily"


class RealDataLoader:
    """Load real Home Assistant snapshot data from disk."""

    def __init__(
        self,
        intraday_dir: Path = DEFAULT_INTRADAY_DIR,
        daily_dir: Path = DEFAULT_DAILY_DIR,
    ):
        self.intraday_dir = intraday_dir
        self.daily_dir = daily_dir

    def load_intraday(self, min_days: int = 3) -> list[dict]:
        """Load intraday snapshots sorted chronologically.

        Returns empty list if directory doesn't exist or has < min_days of data.
        """
        if not self.intraday_dir.exists():
            return []
        day_dirs = sorted(d for d in self.intraday_dir.iterdir() if d.is_dir())
        if len(day_dirs) < min_days:
            return []

        snapshots = []
        for day_dir in day_dirs:
            for hour_file in sorted(day_dir.glob("*.json")):
                try:
                    data = json.loads(hour_file.read_text())
                    # Ensure time_features exists
                    if "time_features" not in data:
                        hour = int(hour_file.stem)
                        data["time_features"] = {"hour": float(hour)}
                    if "date" not in data:
                        data["date"] = day_dir.name
                    snapshots.append(data)
                except (json.JSONDecodeError, ValueError):
                    continue
        return snapshots

    def load_daily(self) -> list[dict]:
        """Load daily snapshots sorted chronologically."""
        if not self.daily_dir.exists():
            return []

        snapshots = []
        for path in sorted(self.daily_dir.iterdir()):
            try:
                if path.suffix == ".gz":
                    data = json.loads(gzip.decompress(path.read_bytes()))
                elif path.suffix == ".json":
                    data = json.loads(path.read_text())
                else:
                    continue
                snapshots.append(data)
            except (json.JSONDecodeError, ValueError, gzip.BadGzipFile):
                continue
        return snapshots

    def available_days(self) -> int:
        """Count how many days of intraday data are available."""
        if not self.intraday_dir.exists():
            return 0
        return sum(1 for d in self.intraday_dir.iterdir() if d.is_dir())
