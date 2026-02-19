"""Hardware capability scanner for tiered model selection."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Tier thresholds
TIER_4_MIN_RAM_GB = 8.0
TIER_3_MIN_RAM_GB = 8.0
TIER_3_MIN_CORES = 4
TIER_2_MIN_RAM_GB = 2.0
TIER_2_MIN_CORES = 2


@dataclass
class HardwareProfile:
    ram_gb: float
    cpu_cores: int
    gpu_available: bool
    gpu_name: str | None = None
    benchmark_score: float | None = None


def recommend_tier(profile: HardwareProfile) -> int:
    """Recommend ML tier based on hardware profile."""
    if profile.gpu_available and profile.ram_gb >= TIER_4_MIN_RAM_GB:
        return 4
    if profile.ram_gb >= TIER_3_MIN_RAM_GB and profile.cpu_cores >= TIER_3_MIN_CORES:
        return 3
    if profile.ram_gb >= TIER_2_MIN_RAM_GB and profile.cpu_cores >= TIER_2_MIN_CORES:
        return 2
    return 1


def scan_hardware() -> HardwareProfile:
    """Probe system hardware and return a HardwareProfile."""
    try:
        import psutil
    except ImportError:
        logger.warning("psutil not installed — defaulting to Tier 1 profile")
        return HardwareProfile(ram_gb=0, cpu_cores=1, gpu_available=False)

    ram_bytes = psutil.virtual_memory().total
    ram_gb = ram_bytes / (1024**3)
    cpu_cores = psutil.cpu_count(logical=True) or 1

    gpu_available = False
    gpu_name = None
    try:
        import torch

        if torch.cuda.is_available():
            gpu_available = True
            gpu_name = torch.cuda.get_device_name(0)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            gpu_available = True
            gpu_name = "Apple MPS"
    except ImportError:
        logger.warning("torch not installed — GPU detection skipped")
    except Exception as e:
        logger.warning("GPU detection failed: %s", e)

    profile = HardwareProfile(
        ram_gb=round(ram_gb, 1),
        cpu_cores=cpu_cores,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
    )
    tier = recommend_tier(profile)
    logger.info(
        f"Hardware: {profile.ram_gb}GB RAM, {profile.cpu_cores} cores, "
        f"GPU={'yes (' + (gpu_name or '?') + ')' if gpu_available else 'no'} "
        f"→ Recommended tier: {tier}"
    )
    return profile
