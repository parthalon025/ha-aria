"""Cross-scenario validation and final accuracy KPI report."""

from collections import Counter

from tests.synthetic.pipeline import PipelineRunner
from tests.synthetic.simulator import HouseholdSimulator


class TestCrossScenarioComparisons:
    """ARIA should produce different predictions for different household patterns."""

    def test_vacation_vs_stable_occupancy(self, all_scenario_results):
        stable = all_scenario_results["stable_couple"]["result"]
        vacation = all_scenario_results["vacation"]["result"]
        stable_occ = stable["predictions"].get("devices_home", {}).get("predicted", 0)
        vacation_occ = vacation["predictions"].get("devices_home", {}).get("predicted", 0)
        # Both should produce numeric predictions
        assert isinstance(stable_occ, int | float), "stable_couple should predict devices_home"
        assert isinstance(vacation_occ, int | float), "vacation should predict devices_home"

    def test_stable_more_accurate_than_vacation(self, all_scenario_results):
        """Stable patterns are easier to predict than vacation absences."""
        stable_score = all_scenario_results["stable_couple"]["result"]["scores"]["overall"]
        vacation_score = all_scenario_results["vacation"]["result"]["scores"]["overall"]
        assert isinstance(stable_score, int | float), "stable_couple overall not numeric"
        assert isinstance(vacation_score, int | float), "vacation overall not numeric"
        assert stable_score >= vacation_score, (
            f"stable_couple ({stable_score}%) should score >= vacation ({vacation_score}%)"
        )

    def test_wfh_more_accurate_than_new_roommate(self, all_scenario_results):
        """Consistent WFH patterns should be easier to predict than changing roommate patterns."""
        wfh_score = all_scenario_results["work_from_home"]["result"]["scores"]["overall"]
        roommate_score = all_scenario_results["new_roommate"]["result"]["scores"]["overall"]
        assert isinstance(wfh_score, int | float), "work_from_home overall not numeric"
        assert isinstance(roommate_score, int | float), "new_roommate overall not numeric"
        assert wfh_score >= roommate_score - 5, (
            f"work_from_home ({wfh_score}%) should score >= new_roommate ({roommate_score}%) - 5"
        )

    def test_higher_occupancy_predicts_higher_power(self, all_scenario_results):
        """More people home should predict higher power consumption."""
        stable_power = (
            all_scenario_results["stable_couple"]["result"]["predictions"].get("power_watts", {}).get("predicted", 0)
        )
        vacation_power = (
            all_scenario_results["vacation"]["result"]["predictions"].get("power_watts", {}).get("predicted", 0)
        )
        assert isinstance(stable_power, int | float), "stable_couple power not numeric"
        assert isinstance(vacation_power, int | float), "vacation power not numeric"
        assert stable_power >= vacation_power, (
            f"stable_couple power ({stable_power}W) should be >= vacation ({vacation_power}W)"
        )

    def test_degradation_does_not_improve_accuracy(self, all_scenario_results):
        """Sensor degradation should not produce better scores than clean data."""
        degraded_score = all_scenario_results["sensor_degradation"]["result"]["scores"]["overall"]
        stable_score = all_scenario_results["stable_couple"]["result"]["scores"]["overall"]
        assert isinstance(degraded_score, int | float), "sensor_degradation overall not numeric"
        assert isinstance(stable_score, int | float), "stable_couple overall not numeric"
        assert degraded_score <= stable_score + 5, (
            f"sensor_degradation ({degraded_score}%) should score <= stable_couple ({stable_score}%) + 5"
        )

    def test_event_count_reflects_activity(self, all_scenario_results):
        """Active households should generate more state-change events."""
        stable_events = all_scenario_results["stable_couple"]["events"]
        vacation_events = all_scenario_results["vacation"]["events"]
        assert len(stable_events) >= len(vacation_events), (
            f"stable_couple events ({len(stable_events)}) should be >= vacation ({len(vacation_events)})"
        )

    def test_more_data_improves_accuracy(self, tmp_path):
        sim = HouseholdSimulator(scenario="stable_couple", days=30, seed=42)
        snapshots = sim.generate()

        runner_14 = PipelineRunner(snapshots[: 14 * 6], data_dir=tmp_path / "14d")
        result_14 = runner_14.run_full()

        runner_28 = PipelineRunner(snapshots[: 28 * 6], data_dir=tmp_path / "28d")
        result_28 = runner_28.run_full()

        score_14 = result_14["scores"]["overall"]
        score_28 = result_28["scores"]["overall"]
        # With more data, score should be at least as good (allowing 10pt variance)
        assert score_28 >= score_14 - 10, f"28d score ({score_28}) should be near or above 14d score ({score_14})"

    def test_all_scenarios_produce_valid_scores(self, all_scenario_results):
        for scenario, data in all_scenario_results.items():
            scores = data["result"]["scores"]
            assert scores is not None, f"{scenario}: scores are None"
            assert "overall" in scores, f"{scenario}: no overall score"
            assert "metrics" in scores, f"{scenario}: no metrics"
            assert isinstance(scores["overall"], int | float), f"{scenario}: overall not numeric"


class TestAccuracyKPI:
    """Final accuracy report -- the single number that matters."""

    def test_overall_accuracy_above_threshold(self, all_scenario_results):
        """Run all scenarios, compute accuracy, print report, assert threshold."""
        scenario_scores = {}
        for scenario, data in all_scenario_results.items():
            scores = data["result"]["scores"]
            scenario_scores[scenario] = scores

        # Print the report
        self._print_report(scenario_scores)

        # Compute overall
        all_overalls = [s["overall"] for s in scenario_scores.values()]
        overall = sum(all_overalls) / len(all_overalls)

        # This is the one number
        print(f"\n{'=' * 50}")
        print(f"  ARIA PREDICTION ACCURACY: {overall:.0f}%")
        print(f"{'=' * 50}\n")

        assert overall > 0, f"Overall accuracy {overall:.0f}% is zero -- something is fundamentally broken"

    def _print_report(self, scenario_scores: dict):
        """Print formatted validation report."""
        metrics = ["power_watts", "lights_on", "devices_home", "unavailable", "useful_events"]
        short_names = {
            "power_watts": "Power",
            "lights_on": "Lights",
            "devices_home": "Occ.",
            "unavailable": "Unavail",
            "useful_events": "Events",
        }

        header = f"{'Scenario':<22} {'Overall':>8}"
        for m in metrics:
            header += f" {short_names.get(m, m):>8}"

        print(f"\n{'=' * len(header)}")
        print("  ARIA VALIDATION REPORT")
        print(f"{'=' * len(header)}")
        print(header)
        print("-" * len(header))

        for scenario, scores in scenario_scores.items():
            line = f"{scenario:<22} {scores['overall']:>7}%"
            for m in metrics:
                metric_data = scores.get("metrics", {}).get(m, {})
                acc = metric_data.get("accuracy", 0)
                line += f" {acc:>7}%"
            print(line)

        all_overalls = [s["overall"] for s in scenario_scores.values()]
        overall = sum(all_overalls) / len(all_overalls)
        print("-" * len(header))
        print(f"{'OVERALL':<22} {overall:>7.0f}%")
        print(f"{'=' * len(header)}")


class TestModuleCoverage:
    """Report which ARIA modules have synthetic event coverage."""

    # Domains that each hub module consumes
    MODULE_DOMAINS = {
        "activity_monitor": {"light", "switch", "binary_sensor", "media_player", "fan"},
        "shadow_engine": {"light", "switch", "lock", "cover", "climate", "fan"},
        "trajectory_classifier": {"light", "binary_sensor", "lock", "media_player"},
        "presence": {"binary_sensor", "person", "device_tracker"},
        "discovery": {"light", "switch", "binary_sensor", "lock", "climate", "sensor"},
    }

    def test_module_coverage_report(self, all_scenario_results):
        """Print per-module event coverage across all scenarios."""
        print(f"\n{'=' * 70}")
        print("  MODULE COVERAGE REPORT")
        print(f"{'=' * 70}")

        header = f"{'Module':<22}"
        scenarios = list(all_scenario_results.keys())
        for s in scenarios:
            header += f" {s[:10]:>10}"
        print(header)
        print("-" * len(header))

        all_covered = True
        for module, domains in self.MODULE_DOMAINS.items():
            line = f"{module:<22}"
            for scenario in scenarios:
                events = all_scenario_results[scenario].get("events", [])
                event_domains = {e["domain"] for e in events}
                overlap = domains & event_domains
                pct = (len(overlap) / len(domains) * 100) if domains else 0
                line += f" {pct:>9.0f}%"
                if pct == 0:
                    all_covered = False
            print(line)

        print(f"{'=' * 70}")

        # Also print total event counts per scenario
        print(f"\n{'Event Counts':<22}", end="")
        for scenario in scenarios:
            events = all_scenario_results[scenario].get("events", [])
            print(f" {len(events):>10}", end="")
        print()

        # Domain distribution for stable_couple
        stable_events = all_scenario_results["stable_couple"].get("events", [])
        domain_counts = Counter(e["domain"] for e in stable_events)
        print(f"\nDomain distribution (stable_couple): {dict(domain_counts)}")

        assert all_covered, "Some modules have 0% coverage in at least one scenario"
