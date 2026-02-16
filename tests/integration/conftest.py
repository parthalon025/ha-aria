"""Shared fixtures for integration tests."""

import pytest

from tests.synthetic.events import EventStreamGenerator
from tests.synthetic.pipeline import PipelineRunner
from tests.synthetic.simulator import HouseholdSimulator


@pytest.fixture(scope="module")
def stable_30d_runner(tmp_path_factory):
    """30-day stable household with full pipeline run. Module-scoped for performance."""
    tmp = tmp_path_factory.mktemp("stable_30d")
    sim = HouseholdSimulator(scenario="stable_couple", days=30, seed=42)
    snapshots = sim.generate()
    runner = PipelineRunner(snapshots, data_dir=tmp)
    runner.save_snapshots()
    return runner


@pytest.fixture(scope="module")
def stable_30d_snapshots():
    """30-day stable household snapshots."""
    sim = HouseholdSimulator(scenario="stable_couple", days=30, seed=42)
    return sim.generate()


# --- Validation suite fixtures ---

VALIDATION_SCENARIOS = [
    ("stable_couple", 30),
    ("vacation", 30),
    ("work_from_home", 30),
    ("new_roommate", 30),
    ("sensor_degradation", 30),
    ("holiday_week", 30),
]


@pytest.fixture(scope="module")
def stable_pipeline(tmp_path_factory):
    """30-day stable_couple with full pipeline. Module-scoped."""
    tmp = tmp_path_factory.mktemp("val_stable")
    sim = HouseholdSimulator(scenario="stable_couple", days=30, seed=42)
    snapshots = sim.generate()
    runner = PipelineRunner(snapshots, data_dir=tmp)
    result = runner.run_full()
    return {"runner": runner, "snapshots": snapshots, "result": result}


@pytest.fixture(scope="module")
def all_scenario_results(tmp_path_factory):
    """All 6 scenarios with full pipeline runs. Module-scoped."""
    results = {}
    for scenario, days in VALIDATION_SCENARIOS:
        tmp = tmp_path_factory.mktemp(f"val_{scenario}")
        sim = HouseholdSimulator(scenario=scenario, days=days, seed=42)
        snapshots = sim.generate()
        runner = PipelineRunner(snapshots, data_dir=tmp)
        result = runner.run_full()
        events = EventStreamGenerator(snapshots).generate()
        results[scenario] = {
            "runner": runner,
            "snapshots": snapshots,
            "result": result,
            "days": days,
            "events": events,
        }
    return results
