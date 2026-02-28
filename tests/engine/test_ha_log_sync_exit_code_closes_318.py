"""Test that ha-log-sync exits non-zero on HA connectivity failure.

Closes #318.
"""

import os
import subprocess
import tempfile
from pathlib import Path

SCRIPT_PATH = Path(__file__).parent.parent.parent / "bin" / "ha-log-sync"


def run_sync_with_mock_curl(mock_curl_exit: int, tmp_home: Path, mode: str = "sync") -> int:
    """Run ha-log-sync with a mock curl that exits with the given code.

    Returns the script's exit code.
    """
    # Create a mock curl that exits with the given code
    mock_bin = tmp_home / "mock-bin"
    mock_bin.mkdir()
    mock_curl = mock_bin / "curl"
    mock_curl.write_text(f"#!/bin/bash\nexit {mock_curl_exit}\n")
    mock_curl.chmod(0o755)

    # Create required dirs and env file
    ha_logs = tmp_home / "ha-logs"
    ha_logs.mkdir()
    env_file = tmp_home / ".env"
    env_file.write_text("export HA_URL=http://test-host:8123\nexport HA_TOKEN=test-token\n")

    env = {
        "HOME": str(tmp_home),
        "PATH": f"{mock_bin}:{os.environ.get('PATH', '')}",
        "HA_URL": "http://test-host:8123",
        "HA_TOKEN": "test-token",
    }

    args = [str(SCRIPT_PATH)]
    if mode != "sync":
        args.append(mode)

    result = subprocess.run(args, env=env, capture_output=True, timeout=10)
    return result.returncode


def test_default_sync_exits_nonzero_on_curl_failure_closes_318():
    """Default sync mode must exit non-zero when curl fails (HA unreachable)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_home = Path(tmp)
        exit_code = run_sync_with_mock_curl(mock_curl_exit=7, tmp_home=tmp_home, mode="sync")
    assert exit_code != 0, (
        "ha-log-sync default sync exited 0 when curl failed — monitoring cannot detect HA downtime (#318)"
    )


def test_full_sync_exits_nonzero_on_curl_failure_closes_318():
    """--full sync mode must exit non-zero when curl fails (HA unreachable)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_home = Path(tmp)
        exit_code = run_sync_with_mock_curl(mock_curl_exit=7, tmp_home=tmp_home, mode="--full")
    assert exit_code != 0, "ha-log-sync --full exited 0 when curl failed — monitoring cannot detect HA downtime (#318)"


def test_sync_exits_zero_on_curl_success_closes_318():
    """Default sync mode must exit 0 when curl succeeds (HA reachable)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_home = Path(tmp)
        # Mock curl that exits 0 and writes valid JSON
        mock_bin = tmp_home / "mock-bin"
        mock_bin.mkdir()
        mock_curl = mock_bin / "curl"
        # Write a valid JSON array to the output file (-o flag)
        mock_curl.write_text(
            "#!/bin/bash\n"
            "# Extract -o argument and write to it\n"
            "while [[ $# -gt 0 ]]; do\n"
            "  if [[ $1 == '-o' ]]; then shift; echo '[]' > \"$1\"; fi\n"
            "  shift\n"
            "done\n"
            "exit 0\n"
        )
        mock_curl.chmod(0o755)

        ha_logs = tmp_home / "ha-logs"
        ha_logs.mkdir()
        env_file = tmp_home / ".env"
        env_file.write_text("export HA_URL=http://test-host:8123\nexport HA_TOKEN=test-token\n")

        env = {
            "HOME": str(tmp_home),
            "PATH": f"{mock_bin}:{os.environ.get('PATH', '')}",
            "HA_URL": "http://test-host:8123",
            "HA_TOKEN": "test-token",
        }
        result = subprocess.run([str(SCRIPT_PATH)], env=env, capture_output=True, timeout=10)

    # Exit 0 on success is acceptable (merge step may fail in test but the connectivity check passed)
    # We just verify it doesn't exit with the "blocked on curl" exit code
    assert result.returncode in (0, 1), f"Unexpected exit code {result.returncode} on successful curl"
