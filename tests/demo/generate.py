"""Generate frozen demo checkpoints from household simulator."""

from __future__ import annotations

from pathlib import Path

from tests.synthetic.pipeline import PipelineRunner
from tests.synthetic.simulator import HouseholdSimulator


def generate_checkpoint(
    scenario: str = "stable_couple",
    days: int = 30,
    seed: int = 42,
    output_dir: Path | str = None,
) -> dict:
    """Generate a frozen demo checkpoint."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sim = HouseholdSimulator(scenario=scenario, days=days, seed=seed)
    snapshots = sim.generate()

    runner = PipelineRunner(snapshots, data_dir=output_dir)
    return runner.run_full()


def generate_all_checkpoints(base_dir: Path | str = None):
    """Generate all standard demo checkpoints."""
    if base_dir is None:
        base_dir = Path(__file__).parent / "fixtures"
    base_dir = Path(base_dir)

    checkpoints = [
        ("day_07", 7),
        ("day_14", 14),
        ("day_30", 30),
        ("day_45", 45),
    ]

    for name, days in checkpoints:
        print(f"Generating {name} ({days} days)...")
        generate_checkpoint(
            scenario="stable_couple",
            days=days,
            seed=42,
            output_dir=base_dir / name,
        )
        print(f"  Done: {base_dir / name}")


if __name__ == "__main__":
    generate_all_checkpoints()
