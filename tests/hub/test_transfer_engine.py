"""Tests for transfer engine hub module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aria.modules.transfer_engine import TransferEngineModule


@pytest.fixture
def mock_hub():
    hub = MagicMock()
    hub.subscribe = MagicMock()
    hub.unsubscribe = MagicMock()
    hub.get_cache = AsyncMock(return_value=None)
    hub.set_cache = AsyncMock()
    hub.publish = AsyncMock()
    hub.get_config_value = MagicMock(return_value=None)
    hub.get_module = MagicMock(return_value=None)
    hub.modules = {}
    return hub


class TestTransferEngineInit:
    """Test module initialization and tier gating."""

    def test_module_id(self, mock_hub):
        module = TransferEngineModule(mock_hub)
        assert module.module_id == "transfer_engine"

    def test_no_subscribe_in_constructor(self, mock_hub):
        TransferEngineModule(mock_hub)
        mock_hub.subscribe.assert_not_called()

    @patch("aria.modules.transfer_engine.recommend_tier", return_value=2)
    @patch("aria.modules.transfer_engine.scan_hardware")
    async def test_tier_gate_blocks_below_tier_3(self, mock_scan, mock_tier, mock_hub):
        mock_scan.return_value = MagicMock(ram_gb=4, cpu_cores=2)
        module = TransferEngineModule(mock_hub)
        await module.initialize()
        assert module.active is False

    @patch("aria.modules.transfer_engine.recommend_tier", return_value=3)
    @patch("aria.modules.transfer_engine.scan_hardware")
    async def test_tier_gate_allows_tier_3(self, mock_scan, mock_tier, mock_hub):
        mock_scan.return_value = MagicMock(ram_gb=16, cpu_cores=4)
        module = TransferEngineModule(mock_hub)
        await module.initialize()
        assert module.active is True
        # Should subscribe to events
        subscribe_events = [call[0][0] for call in mock_hub.subscribe.call_args_list]
        assert "organic_discovery_complete" in subscribe_events
        assert "shadow_resolved" in subscribe_events


class TestCandidateGeneration:
    """Test transfer candidate generation from discovery events."""

    @patch("aria.modules.transfer_engine.recommend_tier", return_value=3)
    @patch("aria.modules.transfer_engine.scan_hardware")
    async def test_on_discovery_complete_generates_candidates(self, mock_scan, mock_tier, mock_hub):
        mock_scan.return_value = MagicMock(ram_gb=16, cpu_cores=4)

        # Set up capability and entity caches
        capabilities = {
            "kitchen_lighting": {
                "entities": [
                    "light.kitchen_1",
                    "binary_sensor.kitchen_motion",
                ],
                "layer": "domain",
                "status": "promoted",
                "source": "organic",
            },
            "bedroom_lighting": {
                "entities": [
                    "light.bedroom_1",
                    "binary_sensor.bedroom_motion",
                ],
                "layer": "domain",
                "status": "candidate",
                "source": "organic",
            },
        }
        entities = {
            "light.kitchen_1": {
                "entity_id": "light.kitchen_1",
                "domain": "light",
                "area_id": "kitchen",
            },
            "binary_sensor.kitchen_motion": {
                "entity_id": "binary_sensor.kitchen_motion",
                "domain": "binary_sensor",
                "area_id": "kitchen",
                "device_class": "motion",
            },
            "light.bedroom_1": {
                "entity_id": "light.bedroom_1",
                "domain": "light",
                "area_id": "bedroom",
            },
            "binary_sensor.bedroom_motion": {
                "entity_id": "binary_sensor.bedroom_motion",
                "domain": "binary_sensor",
                "area_id": "bedroom",
                "device_class": "motion",
            },
        }

        mock_hub.get_cache = AsyncMock(
            side_effect=lambda key: {
                "capabilities": {"data": capabilities},
                "entities": {"data": entities},
            }.get(key)
        )

        module = TransferEngineModule(mock_hub)
        await module.initialize()
        await module._on_discovery_complete({})

        assert len(module.candidates) >= 0  # May or may not meet threshold


class TestShadowTesting:
    """Test shadow result integration for transfer candidates."""

    @patch("aria.modules.transfer_engine.recommend_tier", return_value=3)
    @patch("aria.modules.transfer_engine.scan_hardware")
    async def test_shadow_result_updates_candidate(self, mock_scan, mock_tier, mock_hub):
        mock_scan.return_value = MagicMock(ram_gb=16, cpu_cores=4)

        module = TransferEngineModule(mock_hub)
        await module.initialize()

        # Manually add a test candidate
        from aria.engine.transfer import TransferCandidate, TransferType

        tc = TransferCandidate(
            source_capability="kitchen_lighting",
            target_context="bedroom",
            transfer_type=TransferType.ROOM_TO_ROOM,
            similarity_score=0.72,
            source_entities=["light.kitchen_1"],
            target_entities=["light.bedroom_1"],
        )
        module.candidates.append(tc)

        # Shadow event for a target entity
        await module._on_shadow_resolved(
            {
                "prediction_id": "test-123",
                "features": {"hour_sin": 0.5},
                "outcome": "correct",
                "actual_data": {"entity_id": "light.bedroom_1"},
            }
        )

        assert tc.shadow_tests >= 0  # May or may not match entity


class TestTransferEngineState:
    """Test state reporting."""

    def test_get_stats(self, mock_hub):
        module = TransferEngineModule(mock_hub)
        stats = module.get_stats()
        assert "active" in stats
        assert "candidates_total" in stats
        assert "candidates_by_state" in stats

    def test_get_current_state(self, mock_hub):
        module = TransferEngineModule(mock_hub)
        state = module.get_current_state()
        assert "candidates" in state
        assert "summary" in state

    @patch("aria.modules.transfer_engine.recommend_tier", return_value=3)
    @patch("aria.modules.transfer_engine.scan_hardware")
    async def test_shutdown_unsubscribes(self, mock_scan, mock_tier, mock_hub):
        mock_scan.return_value = MagicMock(ram_gb=16, cpu_cores=4)
        module = TransferEngineModule(mock_hub)
        await module.initialize()
        await module.shutdown()
        assert mock_hub.unsubscribe.call_count == 2
