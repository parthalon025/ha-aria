from unittest.mock import MagicMock, patch

import pytest

from aria.engine.hardware import HardwareProfile, recommend_tier, scan_hardware


class TestHardwareProfile:
    def test_profile_dataclass_fields(self):
        profile = HardwareProfile(ram_gb=32.0, cpu_cores=8, gpu_available=False, gpu_name=None, benchmark_score=None)
        assert profile.ram_gb == 32.0
        assert profile.cpu_cores == 8
        assert profile.gpu_available is False

    def test_tier_1_low_hardware(self):
        profile = HardwareProfile(ram_gb=1.5, cpu_cores=1, gpu_available=False)
        assert recommend_tier(profile) == 1

    def test_tier_2_moderate_hardware(self):
        profile = HardwareProfile(ram_gb=4.0, cpu_cores=2, gpu_available=False)
        assert recommend_tier(profile) == 2

    def test_tier_3_high_cpu(self):
        profile = HardwareProfile(ram_gb=16.0, cpu_cores=8, gpu_available=False)
        assert recommend_tier(profile) == 3

    def test_tier_4_gpu_available(self):
        profile = HardwareProfile(ram_gb=16.0, cpu_cores=8, gpu_available=True)
        assert recommend_tier(profile) == 4

    def test_tier_4_requires_min_ram(self):
        """GPU present but <8GB RAM should not get Tier 4."""
        profile = HardwareProfile(ram_gb=4.0, cpu_cores=4, gpu_available=True)
        assert recommend_tier(profile) == 2

    def test_tier_boundary_2gb_2cores(self):
        profile = HardwareProfile(ram_gb=2.0, cpu_cores=2, gpu_available=False)
        assert recommend_tier(profile) == 2

    def test_tier_boundary_8gb_4cores(self):
        profile = HardwareProfile(ram_gb=8.0, cpu_cores=4, gpu_available=False)
        assert recommend_tier(profile) == 3


class TestScanHardware:
    def test_scan_returns_profile(self):
        mock_psutil = MagicMock()
        mock_psutil.virtual_memory.return_value = MagicMock(total=32 * 1024**3)
        mock_psutil.cpu_count.return_value = 8
        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            profile = scan_hardware()
        assert isinstance(profile, HardwareProfile)
        assert profile.ram_gb == pytest.approx(32.0, abs=0.5)
        assert profile.cpu_cores == 8

    def test_scan_gpu_detection_no_torch(self):
        mock_psutil = MagicMock()
        mock_psutil.virtual_memory.return_value = MagicMock(total=16 * 1024**3)
        mock_psutil.cpu_count.return_value = 4
        with patch.dict("sys.modules", {"psutil": mock_psutil, "torch": None}):
            profile = scan_hardware()
        assert profile.gpu_available is False
