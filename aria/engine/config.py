"""Configuration dataclasses for ARIA engine.

Replaces module-level globals with type-safe, testable config objects.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class HAConfig:
    """Home Assistant connection settings."""
    url: str = "http://192.168.1.35:8123"
    token: str = ""

    @classmethod
    def from_env(cls):
        return cls(
            url=os.environ.get("HA_URL", cls.url),
            token=os.environ.get("HA_TOKEN", ""),
        )


@dataclass
class PathConfig:
    """All data directory paths. Single source of truth for file locations."""
    data_dir: Path = field(default_factory=lambda: Path.home() / "ha-logs" / "intelligence")
    logbook_path: Path = field(default_factory=lambda: Path.home() / "ha-logs" / "current.json")

    @property
    def daily_dir(self) -> Path:
        return self.data_dir / "daily"

    @property
    def intraday_dir(self) -> Path:
        return self.data_dir / "intraday"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def meta_dir(self) -> Path:
        return self.data_dir / "meta-learning"

    @property
    def insights_dir(self) -> Path:
        return self.data_dir / "insights"

    @property
    def baselines_path(self) -> Path:
        return self.data_dir / "baselines.json"

    @property
    def predictions_path(self) -> Path:
        return self.data_dir / "predictions.json"

    @property
    def accuracy_path(self) -> Path:
        return self.data_dir / "accuracy.json"

    @property
    def correlations_path(self) -> Path:
        return self.data_dir / "correlations.json"

    @property
    def feature_config_path(self) -> Path:
        return self.data_dir / "feature_config.json"

    @property
    def snapshot_log_path(self) -> Path:
        return self.data_dir / "snapshot_log.jsonl"

    @property
    def capabilities_path(self) -> Path:
        return self.data_dir / "capabilities.json"

    @property
    def sequence_model_path(self) -> Path:
        return self.models_dir / "sequence_model.json"

    def ensure_dirs(self):
        """Create all required directories."""
        for d in [self.data_dir, self.daily_dir, self.intraday_dir,
                  self.models_dir, self.meta_dir, self.insights_dir]:
            d.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    """sklearn model hyperparameters."""
    n_estimators: int = 100
    max_depth: int = 4
    learning_rate: float = 0.1
    subsample: float = 0.8
    min_samples_ratio: int = 20  # min_samples_leaf = max(3, n // this)
    min_training_samples: int = 14
    validation_split: float = 0.8


@dataclass
class OllamaConfig:
    """Ollama LLM settings."""
    url: str = "http://localhost:11434/api/chat"
    model: str = "deepseek-r1:8b"
    timeout: int = 60


@dataclass
class WeatherConfig:
    """Weather API settings."""
    location: str = "Shalimar+FL"


@dataclass
class SafetyConfig:
    """Entity safety rules."""
    unavailable_exclude_domains: set = field(
        default_factory=lambda: {"update", "tts", "stt"}
    )
    safety_entities: set = field(
        default_factory=lambda: {"lock.", "alarm_", "camera."}
    )


@dataclass
class HolidayConfig:
    """Holiday calendar."""
    country: str = "US"
    years: tuple = (2025, 2026, 2027, 2028)

    def get_holidays(self):
        """Load holiday calendar. Returns empty dict if holidays package unavailable."""
        try:
            import holidays as holidays_lib
            return holidays_lib.country_holidays(self.country, years=list(self.years))
        except ImportError:
            return {}


@dataclass
class AppConfig:
    """Top-level config composing all sub-configs."""
    ha: HAConfig = field(default_factory=HAConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    weather: WeatherConfig = field(default_factory=WeatherConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    holidays: HolidayConfig = field(default_factory=HolidayConfig)

    @classmethod
    def from_env(cls):
        """Create config from environment variables (for production/cron use)."""
        return cls(
            ha=HAConfig.from_env(),
            paths=PathConfig(),
            model=ModelConfig(),
            ollama=OllamaConfig(),
            weather=WeatherConfig(),
            safety=SafetyConfig(),
            holidays=HolidayConfig(),
        )
