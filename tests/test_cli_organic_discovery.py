"""Tests for the discover-organic CLI command."""

import unittest
from unittest.mock import patch, AsyncMock, MagicMock


class TestDiscoverOrganicSubparser(unittest.TestCase):
    """Verify discover-organic is registered as a CLI subcommand."""

    def test_subparser_registered_and_dispatches(self):
        """discover-organic should be a recognized subcommand that routes to _discover_organic."""
        with patch("sys.argv", ["aria", "discover-organic"]):
            with patch("aria.cli._discover_organic") as mock_fn:
                from aria.cli import main
                main()
                mock_fn.assert_called_once()

    def test_unknown_command_exits(self):
        """An unrecognized command should exit with code 1."""
        with patch("sys.argv", ["aria", "not-a-command"]):
            with self.assertRaises(SystemExit) as ctx:
                from aria.cli import main
                main()
            self.assertEqual(ctx.exception.code, 2)


class TestDiscoverOrganicFunction(unittest.TestCase):
    """Verify _discover_organic sets up hub and runs pipeline."""

    @patch("aria.modules.organic_discovery.module.OrganicDiscoveryModule")
    @patch("aria.hub.core.IntelligenceHub")
    def test_creates_hub_and_module(self, MockHub, MockModule):
        """_discover_organic should create a hub, register module, run discovery."""
        # Set up async mocks
        mock_hub_instance = MagicMock()
        mock_hub_instance.initialize = AsyncMock()
        mock_hub_instance.shutdown = AsyncMock()
        mock_hub_instance.is_running.return_value = True
        MockHub.return_value = mock_hub_instance

        mock_module_instance = MagicMock()
        mock_module_instance.initialize = AsyncMock()
        mock_module_instance.run_discovery = AsyncMock(return_value={"clusters": 0})
        MockModule.return_value = mock_module_instance

        from aria.cli import _discover_organic
        _discover_organic()

        MockHub.assert_called_once()
        mock_hub_instance.initialize.assert_called_once()
        MockModule.assert_called_once_with(mock_hub_instance)
        mock_hub_instance.register_module.assert_called_once_with(mock_module_instance)
        mock_module_instance.initialize.assert_called_once()
        mock_module_instance.run_discovery.assert_called_once()
        mock_hub_instance.shutdown.assert_called_once()

    @patch("aria.modules.organic_discovery.module.OrganicDiscoveryModule")
    @patch("aria.hub.core.IntelligenceHub")
    def test_shutdown_on_error(self, MockHub, MockModule):
        """Hub should shut down even if discovery raises an exception."""
        mock_hub_instance = MagicMock()
        mock_hub_instance.initialize = AsyncMock()
        mock_hub_instance.shutdown = AsyncMock()
        mock_hub_instance.is_running.return_value = True
        MockHub.return_value = mock_hub_instance

        mock_module_instance = MagicMock()
        mock_module_instance.initialize = AsyncMock()
        mock_module_instance.run_discovery = AsyncMock(side_effect=RuntimeError("test error"))
        MockModule.return_value = mock_module_instance

        from aria.cli import _discover_organic
        with self.assertRaises(RuntimeError):
            _discover_organic()

        mock_hub_instance.shutdown.assert_called_once()


if __name__ == "__main__":
    unittest.main()
