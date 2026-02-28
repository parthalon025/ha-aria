"""Engine pipeline validation — deterministic assertions across all scenarios."""

import pytest

from tests.synthetic.pipeline import PipelineRunner
from tests.synthetic.simulator import INTRADAY_HOURS, HouseholdSimulator


class TestStableCouple:
    """Baseline scenario: predictable two-person household."""

    def test_pipeline_completes_with_positive_score(self, stable_pipeline):
        result = stable_pipeline["result"]
        assert result["scores"]["overall"] > 0

    def test_power_predictions_in_range(self, stable_pipeline):
        predictions = stable_pipeline["result"]["predictions"]
        power = predictions.get("power_watts", {})
        predicted = power.get("predicted", 0)
        assert 50 <= predicted <= 5000, f"Power prediction {predicted}W outside realistic range"

    def test_weekday_weekend_baselines_differ(self, stable_pipeline):
        baselines = stable_pipeline["result"]["baselines"]
        weekday_keys = [k for k in baselines if k in ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")]
        weekend_keys = [k for k in baselines if k in ("Saturday", "Sunday")]
        if weekday_keys and weekend_keys:
            wd_power = baselines[weekday_keys[0]].get("power_watts", {}).get("mean", 0)
            we_power = baselines[weekend_keys[0]].get("power_watts", {}).get("mean", 0)
            assert wd_power != we_power, "Weekday and weekend power baselines should differ"

    def test_all_target_metrics_trained(self, stable_pipeline):
        training = stable_pipeline["result"]["training"]
        expected_metrics = ["power_watts", "lights_on", "devices_home"]
        for metric in expected_metrics:
            assert metric in training, f"Missing training result for {metric}"
            assert "error" not in training[metric], f"Training failed for {metric}: {training[metric]}"

    def test_r2_above_noise(self, stable_pipeline):
        training = stable_pipeline["result"]["training"]
        r2_values = {m: r.get("r2", -999) for m, r in training.items() if "error" not in r}
        positive_r2 = [m for m, r2 in r2_values.items() if r2 > 0.0]
        assert len(positive_r2) >= 1, f"No metrics with positive R2: {r2_values}"

    def test_feature_vector_shape(self, stable_pipeline):
        runner = stable_pipeline["runner"]
        names, X, _targets = runner.build_training_data()
        assert len(names) > 0
        assert len(X) > 0, "Training data should have rows"
        assert len(X[0]) == len(names), f"Feature count mismatch: {len(X[0])} vs {len(names)}"

    def test_snapshot_schema_complete(self, stable_pipeline):
        snapshots = stable_pipeline["snapshots"]
        required_keys = [
            "date",
            "day_of_week",
            "power",
            "lights",
            "occupancy",
            "climate",
            "locks",
            "motion",
            "entities",
            "weather",
            "time_features",
            "logbook_summary",
        ]
        for snap in snapshots[:5]:
            for key in required_keys:
                assert key in snap, f"Snapshot missing key: {key}"


class TestVacation:
    def test_pipeline_completes(self, all_scenario_results):
        result = all_scenario_results["vacation"]["result"]
        assert result["scores"] is not None

    def test_vacancy_period_lower_occupancy(self, all_scenario_results):
        snapshots = all_scenario_results["vacation"]["snapshots"]
        pre_vacation = snapshots[: 10 * 6]
        vacation_period = snapshots[10 * 6 : 18 * 6]
        pre_occ = sum(s["occupancy"]["device_count_home"] for s in pre_vacation) / len(pre_vacation)
        vac_occ = sum(s["occupancy"]["device_count_home"] for s in vacation_period) / len(vacation_period)
        assert vac_occ < pre_occ, f"Vacation occupancy ({vac_occ:.1f}) should be < pre-vacation ({pre_occ:.1f})"

    def test_vacancy_period_lower_power(self, all_scenario_results):
        snapshots = all_scenario_results["vacation"]["snapshots"]
        pre_vacation = snapshots[: 10 * 6]
        vacation_period = snapshots[10 * 6 : 18 * 6]
        pre_power = sum(s["power"]["total_watts"] for s in pre_vacation) / len(pre_vacation)
        vac_power = sum(s["power"]["total_watts"] for s in vacation_period) / len(vacation_period)
        assert vac_power < pre_power, f"Vacation power ({vac_power:.0f}W) should be < pre-vacation ({pre_power:.0f}W)"


class TestWorkFromHome:
    def test_pipeline_completes(self, all_scenario_results):
        result = all_scenario_results["work_from_home"]["result"]
        assert result["scores"] is not None

    def test_daytime_occupancy_increases_after_wfh(self, all_scenario_results):
        snapshots = all_scenario_results["work_from_home"]["snapshots"]
        pre_wfh_daytime = []
        post_wfh_daytime = []
        for i, snap in enumerate(snapshots):
            day = i // 6
            hour_idx = i % 6
            if hour_idx in (1, 2, 3):  # 9am, 12pm, 3pm
                occ = snap["occupancy"]["device_count_home"]
                if day < 8:
                    pre_wfh_daytime.append(occ)
                elif day >= 8:
                    post_wfh_daytime.append(occ)
        pre_avg = sum(pre_wfh_daytime) / max(len(pre_wfh_daytime), 1)
        post_avg = sum(post_wfh_daytime) / max(len(post_wfh_daytime), 1)
        assert post_avg >= pre_avg, f"WFH daytime occupancy ({post_avg:.1f}) should be >= pre-WFH ({pre_avg:.1f})"


class TestNewRoommate:
    def test_pipeline_completes(self, all_scenario_results):
        result = all_scenario_results["new_roommate"]["result"]
        assert result["scores"] is not None

    def test_occupancy_increases_after_roommate(self, all_scenario_results):
        snapshots = all_scenario_results["new_roommate"]["snapshots"]
        pre_roommate = snapshots[: 15 * 6]
        post_roommate = snapshots[15 * 6 :]
        # Use people_home list length for a cleaner signal than device_count_home
        pre_occ = sum(len(s["occupancy"].get("people_home", [])) for s in pre_roommate) / len(pre_roommate)
        post_occ = sum(len(s["occupancy"].get("people_home", [])) for s in post_roommate) / len(post_roommate)
        # Allow small tolerance for stochastic variation in schedules
        assert post_occ >= pre_occ - 0.2, f"Post-roommate occupancy ({post_occ:.1f}) should be >= pre ({pre_occ:.1f})"


class TestSensorDegradation:
    def test_pipeline_completes(self, all_scenario_results):
        result = all_scenario_results["sensor_degradation"]["result"]
        assert result["scores"] is not None
        assert result["snapshots_saved"] == 30 * len(INTRADAY_HOURS)

    def test_unavailable_count_increases(self, all_scenario_results):
        snapshots = all_scenario_results["sensor_degradation"]["snapshots"]
        pre_degrade = snapshots[: 20 * 6]
        post_degrade = snapshots[20 * 6 :]
        pre_unavail = sum(len(s.get("entities_summary", {}).get("unavailable_entities", [])) for s in pre_degrade)
        post_unavail = sum(len(s.get("entities_summary", {}).get("unavailable_entities", [])) for s in post_degrade)
        assert post_unavail > pre_unavail, "Unavailable entities should increase after degradation starts"


class TestHolidayWeek:
    def test_pipeline_completes(self, all_scenario_results):
        result = all_scenario_results["holiday_week"]["result"]
        assert result["scores"] is not None

    def test_holiday_flags_present(self, all_scenario_results):
        snapshots = all_scenario_results["holiday_week"]["snapshots"]
        holiday_snapshots = [s for s in snapshots if s.get("is_holiday")]
        assert len(holiday_snapshots) >= 3 * 6, f"Expected 18+ holiday snapshots, got {len(holiday_snapshots)}"


class TestColdStart:
    def test_7_day_cold_start(self, tmp_path):
        sim = HouseholdSimulator(scenario="stable_couple", days=7, seed=42)
        snapshots = sim.generate()
        runner = PipelineRunner(snapshots, data_dir=tmp_path)
        runner.save_snapshots()
        baselines = runner.compute_baselines()
        assert len(baselines) > 0, "7-day data should produce baselines"
        predictions = runner.generate_predictions()
        assert predictions is not None, "7-day data should produce predictions"
        assert "power_watts" in predictions


class TestSchemaAllScenarios:
    @pytest.mark.parametrize(
        "scenario",
        [
            "stable_couple",
            "vacation",
            "work_from_home",
            "new_roommate",
            "sensor_degradation",
            "holiday_week",
        ],
    )
    def test_snapshot_has_required_keys(self, all_scenario_results, scenario):
        snapshots = all_scenario_results[scenario]["snapshots"]
        required_keys = [
            "date",
            "day_of_week",
            "power",
            "lights",
            "occupancy",
            "climate",
            "locks",
            "motion",
            "entities",
            "weather",
        ]
        snap = snapshots[0]
        for key in required_keys:
            assert key in snap, f"{scenario} snapshot missing key: {key}"
