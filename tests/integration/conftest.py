"""Shared fixtures for integration tests."""

import pytest

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
