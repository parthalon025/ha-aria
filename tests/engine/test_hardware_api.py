"""Test hardware profile API response shape — validates in isolation."""

from unittest.mock import MagicMock

from aria.engine.hardware import HardwareProfile, recommend_tier


class TestHardwareAPIResponse:
    def test_hardware_response_shape(self):
        """Verify the hardware endpoint response has all required fields."""
        profile = HardwareProfile(ram_gb=16.0, cpu_cores=8, gpu_available=False)
        tier = recommend_tier(profile)

        # Simulate the API response construction
        ml_module = MagicMock()
        ml_module.current_tier = tier
        ml_module.fallback_tracker.to_dict.return_value = []

        response = {
            "ram_gb": profile.ram_gb,
            "cpu_cores": profile.cpu_cores,
            "gpu_available": profile.gpu_available,
            "gpu_name": profile.gpu_name,
            "recommended_tier": tier,
            "current_tier": ml_module.current_tier if ml_module else tier,
            "tier_override": "auto",
            "active_fallbacks": (ml_module.fallback_tracker.to_dict() if ml_module else []),
        }

        assert "ram_gb" in response
        assert "cpu_cores" in response
        assert "recommended_tier" in response
        assert "current_tier" in response
        assert "active_fallbacks" in response
        assert response["recommended_tier"] == 3
        assert isinstance(response["active_fallbacks"], list)

    def test_hardware_response_without_ml_module(self):
        """When ml_engine module is unavailable, use recommended tier."""
        profile = HardwareProfile(ram_gb=4.0, cpu_cores=2, gpu_available=False)
        tier = recommend_tier(profile)
        ml_module = None

        response = {
            "ram_gb": profile.ram_gb,
            "cpu_cores": profile.cpu_cores,
            "gpu_available": profile.gpu_available,
            "gpu_name": profile.gpu_name,
            "recommended_tier": tier,
            "current_tier": ml_module.current_tier if ml_module else tier,
            "tier_override": "auto",
            "active_fallbacks": (ml_module.fallback_tracker.to_dict() if ml_module else []),
        }

        assert response["current_tier"] == response["recommended_tier"]
        assert response["active_fallbacks"] == []
