"""Presence module validation — Bayesian occupancy from synthetic sensor events."""

from datetime import datetime

from aria.engine.analysis.occupancy import BayesianOccupancy
from tests.synthetic.events import EventStreamGenerator
from tests.synthetic.simulator import HouseholdSimulator


class TestPresenceWithSyntheticData:
    """Presence detection should produce room probabilities from synthetic events."""

    def test_bayesian_occupancy_from_synthetic_snapshots(self):
        """BayesianOccupancy should estimate occupancy from synthetic snapshots."""
        sim = HouseholdSimulator(scenario="stable_couple", days=7, seed=42)
        snapshots = sim.generate()
        occ = BayesianOccupancy()
        results_with_probability = 0
        for snap in snapshots[:30]:
            ts = datetime.strptime(snap.get("date", "2026-02-01"), "%Y-%m-%d")
            tf = snap.get("time_features", {})
            hour = int(tf.get("hour", 12))
            ts = ts.replace(hour=hour)
            result = occ.estimate(snap, timestamp=ts)
            assert isinstance(result, dict)
            assert "overall" in result
            overall = result["overall"]
            assert "probability" in overall
            assert isinstance(overall["probability"], int | float)
            if overall["probability"] > 0:
                results_with_probability += 1
        assert results_with_probability > 0, "Should have at least one non-zero probability"

    def test_presence_differs_by_scenario(self):
        """Vacation scenario should show lower presence signals than stable."""
        sim_stable = HouseholdSimulator(scenario="stable_couple", days=14, seed=42)
        sim_vacation = HouseholdSimulator(scenario="vacation", days=14, seed=42)
        events_stable = EventStreamGenerator(sim_stable.generate()).generate()
        events_vacation = EventStreamGenerator(sim_vacation.generate()).generate()
        home_stable = [
            e for e in events_stable if e["domain"] in ("person", "device_tracker") and e["new_state"] == "home"
        ]
        home_vacation = [
            e for e in events_vacation if e["domain"] in ("person", "device_tracker") and e["new_state"] == "home"
        ]
        assert len(home_stable) >= len(home_vacation), (
            f"Stable ({len(home_stable)}) should have >= home events than vacation ({len(home_vacation)})"
        )

    def test_all_scenarios_produce_presence_signals(self, all_scenario_results):
        """Every scenario should produce motion or person events."""
        for scenario, data in all_scenario_results.items():
            events = EventStreamGenerator(data["snapshots"]).generate()
            presence_events = [e for e in events if e["domain"] in ("binary_sensor", "person", "device_tracker")]
            assert len(presence_events) > 0, f"{scenario}: no presence-relevant events"
