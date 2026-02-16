"""Run ARIA validation suite and parse results into structured JSON."""

import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Cache last result in memory (survives across requests, reset on restart)
_last_result = None


def run_validation() -> dict:
    """Execute validation suite via pytest subprocess, return structured results."""
    global _last_result

    project_root = Path(__file__).resolve().parent.parent.parent
    venv_pytest = project_root / ".venv" / "bin" / "python"

    cmd = [
        str(venv_pytest),
        "-m",
        "pytest",
        "tests/integration/test_validation_scenarios.py",
        "tests/integration/test_validation_backtest.py",
        "-v",
        "--timeout=120",
        "--tb=short",
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(project_root),
        )

        result = _parse_pytest_output(proc.stdout, proc.stderr, proc.returncode)
        result["timestamp"] = datetime.now(UTC).isoformat()
        result["duration_seconds"] = _extract_duration(proc.stdout)
        _last_result = result
        return result

    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "error": "Validation suite exceeded 300s timeout",
            "timestamp": datetime.now(UTC).isoformat(),
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "timestamp": datetime.now(UTC).isoformat()}


def get_latest() -> dict | None:
    """Return the most recent validation result, or None."""
    return _last_result


def _parse_pytest_output(stdout: str, stderr: str, returncode: int) -> dict:
    """Parse pytest verbose output into structured result."""
    lines = stdout.splitlines()
    tests = []
    passed = 0
    failed = 0
    skipped = 0

    # Parse PASSED/FAILED lines from -v output
    for line in lines:
        if " PASSED" in line:
            name = line.split(" PASSED")[0].strip().split("::")[-1]
            tests.append({"name": name, "status": "passed"})
            passed += 1
        elif " FAILED" in line:
            name = line.split(" FAILED")[0].strip().split("::")[-1]
            tests.append({"name": name, "status": "failed"})
            failed += 1
        elif " SKIPPED" in line:
            name = line.split(" SKIPPED")[0].strip().split("::")[-1]
            tests.append({"name": name, "status": "skipped"})
            skipped += 1

    # Extract the accuracy report from captured stdout
    report = _extract_accuracy_report(stdout)

    return {
        "status": "passed" if returncode == 0 else "failed",
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "total": passed + failed + skipped,
        "tests": tests,
        "report": report,
        "raw_output": stdout[-2000:] if len(stdout) > 2000 else stdout,
    }


def _extract_accuracy_report(stdout: str) -> dict:
    """Extract scenario accuracy scores from the validation report printed to stdout."""
    scenarios, overall = _extract_scenario_scores(stdout.splitlines())
    backtest = _extract_backtest_scores(stdout.splitlines())
    return {"scenarios": scenarios, "overall": overall, "backtest": backtest}


def _extract_scenario_scores(lines: list[str]) -> tuple[dict, float | None]:
    """Parse scenario scores and overall accuracy from ARIA VALIDATION REPORT section."""
    scenarios = {}
    overall = None
    in_report = False

    for line in lines:
        if "ARIA VALIDATION REPORT" in line:
            in_report = True
            continue
        if not in_report:
            continue
        if line.strip().startswith("OVERALL"):
            for p in line.split():
                if p.endswith("%"):
                    overall = float(p.rstrip("%"))
                    break
            break
        if "%" in line and not line.startswith("-") and not line.startswith("="):
            parts = line.split()
            if len(parts) >= 2:
                pcts = [float(p.rstrip("%")) for p in parts[1:] if p.endswith("%")]
                if pcts:
                    scenarios[parts[0]] = {
                        "overall": pcts[0],
                        "metrics": pcts[1:] if len(pcts) > 1 else [],
                    }

    return scenarios, overall


def _extract_backtest_scores(lines: list[str]) -> dict:
    """Parse real-data backtest accuracy from REAL-DATA BACKTEST section."""
    backtest = {}
    in_backtest = False

    for line in lines:
        if "REAL-DATA BACKTEST" in line or "Real vs Synthetic" in line:
            in_backtest = True
            continue
        if in_backtest and ("real data" in line.lower() or "real:" in line.lower()):
            pcts = [float(p.rstrip("%")) for p in line.split() if p.endswith("%")]
            if pcts:
                backtest["overall"] = pcts[0]
                backtest["metrics"] = pcts[1:] if len(pcts) > 1 else []
            break

    return backtest


def _extract_duration(stdout: str) -> float | None:
    """Extract test duration from pytest summary line like '= 17 passed in 34.11s ='."""
    import re

    for line in reversed(stdout.splitlines()):
        if "passed" in line and " in " in line:
            # Match patterns like "34.11s" or "1:02:03" surrounded by any decoration
            m = re.search(r"in\s+([\d:.]+)s?\b", line)
            if not m:
                continue
            time_str = m.group(1)
            if ":" in time_str:
                segments = time_str.split(":")
                return sum(float(s) * (60**i) for i, s in enumerate(reversed(segments)))
            try:
                return float(time_str)
            except ValueError:
                continue
    return None
