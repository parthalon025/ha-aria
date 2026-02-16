"""Shared fixtures for tests/hub/ test suite.

Provides the common API hub mock and test client fixtures used by the
test_api_*.py files. Module-specific MockHub classes and hub fixtures
remain in their respective test files — they shadow these fixtures
via pytest's scoping rules.
"""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from aria.hub.api import create_api
from aria.hub.core import IntelligenceHub


@pytest.fixture
def api_hub():
    """Create a mock IntelligenceHub for API endpoint tests.

    Used by test_api_shadow.py, test_api_config.py, test_api_ml.py,
    test_api_features.py, and test_api_organic_discovery.py.

    Module tests (test_orchestrator.py, test_patterns.py, etc.) define their
    own hub fixtures with custom MockHub classes that shadow this one.
    """
    mock_hub = MagicMock(spec=IntelligenceHub)
    mock_hub.cache = MagicMock()
    mock_hub.modules = {}
    mock_hub.module_status = {}
    mock_hub.subscribers = {}
    mock_hub.subscribe = MagicMock()
    mock_hub._request_count = 0
    mock_hub.get_uptime_seconds = MagicMock(return_value=0)
    return mock_hub


@pytest.fixture
def api_client(api_hub):
    """Create a FastAPI TestClient backed by api_hub."""
    app = create_api(api_hub)
    return TestClient(app)
