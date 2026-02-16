"""Tests for aria status CLI command and hub module status tracking."""

import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio

sys.path.insert(0, str(Path(__file__).parent.parent))

from aria.hub.core import IntelligenceHub

# ============================================================================
# IntelligenceHub module status tracking
# ============================================================================


@pytest.fixture
def temp_cache():
    """Create a temporary cache path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield str(Path(tmpdir) / "test.db")


@pytest_asyncio.fixture
async def hub(temp_cache):
    """Create an initialized hub."""
    h = IntelligenceHub(temp_cache)
    await h.initialize()
    yield h
    await h.shutdown()


@pytest.mark.asyncio
async def test_module_status_starts_registered(hub):
    """Module status is 'registered' after register_module."""
    from aria.hub.core import Module

    mod = Module("test_mod", hub)
    hub.register_module(mod)
    assert hub.module_status["test_mod"] == "registered"


@pytest.mark.asyncio
async def test_mark_module_running(hub):
    """mark_module_running sets status to 'running'."""
    from aria.hub.core import Module

    mod = Module("test_mod", hub)
    hub.register_module(mod)
    hub.mark_module_running("test_mod")
    assert hub.module_status["test_mod"] == "running"


@pytest.mark.asyncio
async def test_mark_module_failed(hub):
    """mark_module_failed sets status to 'failed'."""
    from aria.hub.core import Module

    mod = Module("test_mod", hub)
    hub.register_module(mod)
    hub.mark_module_failed("test_mod")
    assert hub.module_status["test_mod"] == "failed"


@pytest.mark.asyncio
async def test_uptime_tracking(hub):
    """get_uptime_seconds returns positive value after init."""
    uptime = hub.get_uptime_seconds()
    assert uptime >= 0


@pytest.mark.asyncio
async def test_start_time_set_on_init(hub):
    """_start_time is set during initialize()."""
    assert hub._start_time is not None
    assert isinstance(hub._start_time, datetime)


@pytest.mark.asyncio
async def test_health_check_includes_status(hub):
    """Health check includes 'status' and 'uptime_seconds' fields."""
    health = await hub.health_check()
    assert health["status"] == "ok"
    assert "uptime_seconds" in health
    assert isinstance(health["uptime_seconds"], int)


@pytest.mark.asyncio
async def test_health_check_module_status(hub):
    """Health check reflects module status correctly."""
    from aria.hub.core import Module

    mod1 = Module("mod_a", hub)
    mod2 = Module("mod_b", hub)
    hub.register_module(mod1)
    hub.register_module(mod2)
    hub.mark_module_running("mod_a")
    hub.mark_module_failed("mod_b")

    health = await hub.health_check()
    assert health["modules"]["mod_a"] == "running"
    assert health["modules"]["mod_b"] == "failed"


# ============================================================================
# aria status command
# ============================================================================


class TestStatusCommand:
    def test_status_json_no_hub(self, capsys, tmp_path):
        """status --json works when hub is not running."""
        from aria.cli import _status

        with (
            patch("os.path.expanduser", return_value=str(tmp_path)),
            patch("urllib.request.urlopen", side_effect=ConnectionRefusedError),
        ):
            _status(json_output=True)

        output = capsys.readouterr().out
        data = json.loads(output)
        assert data["hub_running"] is False
        assert data["hub_health"] is None
        assert "version" in data

    def test_status_pretty_no_hub(self, capsys, tmp_path):
        """status pretty-prints when hub is not running."""
        from aria.cli import _status

        with (
            patch("os.path.expanduser", return_value=str(tmp_path)),
            patch("urllib.request.urlopen", side_effect=ConnectionRefusedError),
        ):
            _status(json_output=False)

        output = capsys.readouterr().out
        assert "ARIA Status" in output
        assert "stopped" in output

    def test_status_shows_snapshot_date(self, capsys, tmp_path):
        """status detects last snapshot file."""
        from aria.cli import _status

        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()
        (daily_dir / "2026-02-13.jsonl").write_text("{}\n")

        with (
            patch("os.path.expanduser", return_value=str(tmp_path)),
            patch("urllib.request.urlopen", side_effect=ConnectionRefusedError),
        ):
            _status(json_output=True)

        output = capsys.readouterr().out
        data = json.loads(output)
        assert data["last_snapshot"] is not None
        assert "2026" in data["last_snapshot"]

    def test_status_shows_training_date(self, capsys, tmp_path):
        """status detects last model file."""
        from aria.cli import _status

        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "gradient_boosting.joblib").write_bytes(b"\x00")

        with (
            patch("os.path.expanduser", return_value=str(tmp_path)),
            patch("urllib.request.urlopen", side_effect=ConnectionRefusedError),
        ):
            _status(json_output=True)

        output = capsys.readouterr().out
        data = json.loads(output)
        assert data["last_training"] is not None

    def test_status_json_with_hub_running(self, capsys, tmp_path):
        """status shows hub info when hub is running."""
        from aria.cli import _status

        health_response = json.dumps(
            {
                "status": "ok",
                "uptime_seconds": 7200,
                "modules": {"discovery": "running", "ml_engine": "running"},
                "cache": {"categories": ["entities", "areas", "capabilities"]},
                "timestamp": "2026-02-13T10:00:00",
            }
        ).encode()

        mock_resp = MagicMock()
        mock_resp.read.return_value = health_response
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with (
            patch("os.path.expanduser", return_value=str(tmp_path)),
            patch("urllib.request.urlopen", return_value=mock_resp),
        ):
            _status(json_output=True)

        output = capsys.readouterr().out
        data = json.loads(output)
        assert data["hub_running"] is True
        assert data["cache_categories"] == 3
