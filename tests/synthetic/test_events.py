"""Tests for synthetic event stream generator."""

from tests.synthetic.events import EventStreamGenerator
from tests.synthetic.simulator import HouseholdSimulator


class TestEventStreamGenerator:
    def test_generates_events_from_snapshots(self):
        sim = HouseholdSimulator(scenario="stable_couple", days=3, seed=42)
        snapshots = sim.generate()
        gen = EventStreamGenerator(snapshots)
        events = gen.generate()
        assert len(events) > 0, "Should produce events"

    def test_events_have_required_fields(self):
        sim = HouseholdSimulator(scenario="stable_couple", days=3, seed=42)
        snapshots = sim.generate()
        gen = EventStreamGenerator(snapshots)
        events = gen.generate()
        event = events[0]
        assert "entity_id" in event
        assert "new_state" in event
        assert "old_state" in event
        assert "timestamp" in event

    def test_events_in_chronological_order(self):
        sim = HouseholdSimulator(scenario="stable_couple", days=3, seed=42)
        snapshots = sim.generate()
        gen = EventStreamGenerator(snapshots)
        events = gen.generate()
        timestamps = [e["timestamp"] for e in events]
        assert timestamps == sorted(timestamps), "Events must be chronological"

    def test_events_include_tracked_domains(self):
        sim = HouseholdSimulator(scenario="stable_couple", days=3, seed=42)
        snapshots = sim.generate()
        gen = EventStreamGenerator(snapshots)
        events = gen.generate()
        domains = {e["entity_id"].split(".")[0] for e in events}
        assert "light" in domains or "binary_sensor" in domains

    def test_vacation_has_fewer_events(self):
        sim_stable = HouseholdSimulator(scenario="stable_couple", days=7, seed=42)
        sim_vacation = HouseholdSimulator(scenario="vacation", days=7, seed=42)
        events_stable = EventStreamGenerator(sim_stable.generate()).generate()
        events_vacation = EventStreamGenerator(sim_vacation.generate()).generate()
        active_stable = [e for e in events_stable if e["new_state"] not in ("off", "not_home", "unavailable")]
        active_vacation = [e for e in events_vacation if e["new_state"] not in ("off", "not_home", "unavailable")]
        assert len(active_vacation) <= len(active_stable), "Vacation should have fewer active events"

    def test_event_count_scales_with_days(self):
        sim_3 = HouseholdSimulator(scenario="stable_couple", days=3, seed=42)
        sim_7 = HouseholdSimulator(scenario="stable_couple", days=7, seed=42)
        events_3 = EventStreamGenerator(sim_3.generate()).generate()
        events_7 = EventStreamGenerator(sim_7.generate()).generate()
        assert len(events_7) > len(events_3), "More days = more events"
