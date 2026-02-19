"""Activity Labeler validation — LLM prediction with synthetic context."""

from aria.modules.activity_labeler import ACTIVITY_PROMPT_TEMPLATE, CLASSIFIER_THRESHOLD
from tests.synthetic.simulator import HouseholdSimulator


class TestActivityLabelerValidation:
    """Activity labeler should produce predictions from synthetic context."""

    def test_prompt_template_renders_with_synthetic_data(self):
        """Prompt template should render with realistic synthetic values."""
        prompt = ACTIVITY_PROMPT_TEMPLATE.format(
            power_watts=450,
            lights_on=3,
            motion_rooms="living_room, kitchen",
            time_of_day="evening",
            hour=19,
            minute=30,
            occupancy="home",
            recent_events="light.living_room turned on, switch.kitchen turned on",
        )
        assert "450W" in prompt
        assert "evening" in prompt
        assert "living_room" in prompt

    def test_classifier_threshold_is_reasonable(self):
        """Classifier should require minimum labels before training."""
        assert CLASSIFIER_THRESHOLD == 50, "Threshold should be 50 labels"

    def test_synthetic_snapshots_have_labeler_inputs(self):
        """Every scenario should provide data the labeler needs."""
        sim = HouseholdSimulator(scenario="stable_couple", days=7, seed=42)
        snapshots = sim.generate()
        for snap in snapshots[-5:]:
            power = snap.get("power", {})
            assert "total_watts" in power, (
                f"Snapshot should have power.total_watts, got power keys: {list(power.keys())}"
            )

    def test_all_scenarios_have_labeler_context(self, all_scenario_results):
        """Every scenario should provide sufficient context for labeling."""
        for scenario, data in all_scenario_results.items():
            last_snap = data["snapshots"][-1]
            has_power = "power" in last_snap and "total_watts" in last_snap.get("power", {})
            has_lights = "lights" in last_snap and "on" in last_snap.get("lights", {})
            has_time = "time_features" in last_snap or "date" in last_snap
            assert has_power and has_lights and has_time, (
                f"{scenario}: missing labeler context (power={has_power}, lights={has_lights}, time={has_time})"
            )
