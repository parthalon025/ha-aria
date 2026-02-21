"""Tests for action builder — service selection, area targeting, restricted domains."""

from unittest.mock import MagicMock

from aria.automation.action_builder import (
    DOMAIN_SERVICE_MAP,
    RESTRICTED_DOMAINS,
    build_actions,
)
from aria.automation.models import ChainLink, DetectionResult


def _detection(  # noqa: PLR0913 — test helper
    trigger_entity="binary_sensor.bedroom_motion",
    action_entities=None,
    area_id="bedroom",
    day_type="workday",
    confidence=0.85,
    entity_chain=None,
):
    if action_entities is None:
        action_entities = ["light.bedroom"]
    if entity_chain is None:
        entity_chain = [
            ChainLink(entity_id=trigger_entity, state="on", offset_seconds=0),
            ChainLink(entity_id=action_entities[0], state="on", offset_seconds=30),
        ]
    return DetectionResult(
        source="pattern",
        trigger_entity=trigger_entity,
        action_entities=action_entities,
        entity_chain=entity_chain,
        area_id=area_id,
        confidence=confidence,
        recency_weight=0.9,
        observation_count=47,
        first_seen="2026-01-01T06:30:00",
        last_seen="2026-02-19T06:45:00",
        day_type=day_type,
        combined_score=0.8,
    )


def _mock_entity_graph():
    graph = MagicMock()
    graph.entities_in_area.return_value = [
        "light.bedroom",
        "light.bedroom_lamp",
        "binary_sensor.bedroom_motion",
        "person.alice",
    ]
    return graph


class TestServiceSelection:
    """Test domain → service mapping for action entities."""

    def test_light_turn_on(self):
        actions = build_actions(_detection(), _mock_entity_graph())
        assert len(actions) >= 1
        assert actions[0]["action"] == "light.turn_on"

    def test_light_turn_off(self):
        det = _detection()
        det.entity_chain[1] = ChainLink(entity_id="light.bedroom", state="off", offset_seconds=30)
        actions = build_actions(det, _mock_entity_graph())
        assert actions[0]["action"] == "light.turn_off"

    def test_switch_service(self):
        det = _detection(action_entities=["switch.coffee_maker"])
        det.entity_chain[1] = ChainLink(entity_id="switch.coffee_maker", state="on", offset_seconds=30)
        actions = build_actions(det, _mock_entity_graph())
        assert actions[0]["action"] == "switch.turn_on"

    def test_fan_service(self):
        det = _detection(action_entities=["fan.bedroom"])
        det.entity_chain[1] = ChainLink(entity_id="fan.bedroom", state="on", offset_seconds=30)
        actions = build_actions(det, _mock_entity_graph())
        assert actions[0]["action"] == "fan.turn_on"

    def test_media_player_service(self):
        det = _detection(action_entities=["media_player.living_room"])
        det.entity_chain[1] = ChainLink(entity_id="media_player.living_room", state="playing", offset_seconds=30)
        actions = build_actions(det, _mock_entity_graph())
        assert actions[0]["action"] == "media_player.media_play"


class TestAreaTargeting:
    """Test area-based targeting preference."""

    def test_uses_area_target_when_all_actions_in_area(self):
        """When all action entities are in the same area, use area targeting."""
        actions = build_actions(_detection(), _mock_entity_graph())
        assert len(actions) >= 1
        target = actions[0].get("target", {})
        assert target.get("area_id") == "bedroom"

    def test_falls_back_to_entity_list_without_area(self):
        """When no area_id, target entities directly."""
        det = _detection(area_id=None)
        graph = MagicMock()
        graph.entities_in_area.return_value = []
        actions = build_actions(det, graph)
        assert len(actions) >= 1
        target = actions[0].get("target", {})
        assert "entity_id" in target
        assert target["entity_id"] == "light.bedroom"

    def test_multiple_action_entities_same_domain(self):
        """Multiple action entities of the same domain produce one action."""
        det = _detection(action_entities=["light.bedroom", "light.bedroom_lamp"])
        det.entity_chain = [
            ChainLink(entity_id="binary_sensor.bedroom_motion", state="on", offset_seconds=0),
            ChainLink(entity_id="light.bedroom", state="on", offset_seconds=30),
            ChainLink(entity_id="light.bedroom_lamp", state="on", offset_seconds=35),
        ]
        actions = build_actions(det, _mock_entity_graph())
        light_actions = [a for a in actions if a["action"].startswith("light.")]
        assert len(light_actions) == 1  # Grouped under area target


class TestRestrictedDomains:
    """Test restricted domain detection and flagging."""

    def test_restricted_domains_defined(self):
        assert "lock" in RESTRICTED_DOMAINS
        assert "alarm_control_panel" in RESTRICTED_DOMAINS
        assert "cover" in RESTRICTED_DOMAINS

    def test_lock_action_flagged_restricted(self):
        det = _detection(action_entities=["lock.front_door"])
        det.entity_chain[1] = ChainLink(entity_id="lock.front_door", state="locked", offset_seconds=30)
        actions = build_actions(det, _mock_entity_graph())
        assert len(actions) >= 1
        assert actions[0].get("_restricted") is True

    def test_light_action_not_restricted(self):
        actions = build_actions(_detection(), _mock_entity_graph())
        for action in actions:
            assert action.get("_restricted") is not True


class TestActionStructure:
    """Test the action dict structure matches HA schema."""

    def test_has_required_keys(self):
        actions = build_actions(_detection(), _mock_entity_graph())
        assert len(actions) >= 1
        action = actions[0]
        assert "action" in action
        assert "target" in action

    def test_returns_list(self):
        actions = build_actions(_detection(), _mock_entity_graph())
        assert isinstance(actions, list)
        for a in actions:
            assert isinstance(a, dict)


class TestDomainServiceMap:
    """Test service map completeness."""

    def test_common_domains_mapped(self):
        assert "light" in DOMAIN_SERVICE_MAP
        assert "switch" in DOMAIN_SERVICE_MAP
        assert "fan" in DOMAIN_SERVICE_MAP
        assert "media_player" in DOMAIN_SERVICE_MAP
        assert "climate" in DOMAIN_SERVICE_MAP
