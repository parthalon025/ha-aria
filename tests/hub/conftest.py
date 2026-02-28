"""Shared fixtures for tests/hub/ test suite.

Provides the common API hub mock and test client fixtures used by the
test_api_*.py files. Module-specific MockHub classes and hub fixtures
remain in their respective test files — they shadow these fixtures
via pytest's scoping rules.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

import aria.hub.api as _api_module
from aria.hub.api import create_api
from aria.hub.core import IntelligenceHub

# Test API key injected for all hub test clients.  Tests that exercise
# auth behaviour set _api_module._ARIA_API_KEY themselves and restore it.
_TEST_API_KEY = "test-aria-key"


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
    mock_hub._audit_logger = None
    mock_hub.set_cache = AsyncMock()
    mock_hub.get_uptime_seconds = MagicMock(return_value=0)
    mock_hub.publish = AsyncMock()
    return mock_hub


@pytest.fixture
def api_client(api_hub):
    """Create a FastAPI TestClient backed by api_hub.

    The test key is injected into the api module for the duration of each
    test and restored on teardown.  All requests carry the X-API-Key header
    so that endpoints guarded by verify_api_key() pass auth.
    """
    original_key = _api_module._ARIA_API_KEY
    _api_module._ARIA_API_KEY = _TEST_API_KEY
    try:
        app = create_api(api_hub)
        yield TestClient(app, headers={"X-API-Key": _TEST_API_KEY})
    finally:
        _api_module._ARIA_API_KEY = original_key
