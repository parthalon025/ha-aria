#!/usr/bin/env python3
"""HA Intelligence Hub - Main entry point.

Starts the Intelligence Hub with FastAPI server, cache management,
and module orchestration.

Usage:
    ha-hub.py [--port PORT] [--host HOST] [--log-level LEVEL]

Options:
    --port PORT         FastAPI server port (default: 8000)
    --host HOST         FastAPI server host (default: 127.0.0.1)
    --log-level LEVEL   Logging level (default: INFO)
    --cache-dir DIR     Cache directory path (default: ~/ha-logs/intelligence/cache)
"""

import sys
import os
import asyncio
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn
from hub.core import IntelligenceHub
from hub.api import create_api
from modules.discovery import DiscoveryModule
from modules.ml_engine import MLEngine
from modules.orchestrator import OrchestratorModule
from modules.patterns import PatternRecognition
from modules.intelligence import IntelligenceModule
from modules.activity_monitor import ActivityMonitor


# Global hub instance for signal handling
hub_instance = None


def setup_logging(log_level: str = "INFO"):
    """Configure logging for hub and modules."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


async def start_hub(cache_path: str) -> IntelligenceHub:
    """Initialize and start the intelligence hub.

    Args:
        cache_path: Path to cache database

    Returns:
        Initialized IntelligenceHub instance
    """
    hub = IntelligenceHub(cache_path)
    await hub.initialize()
    return hub


async def shutdown_hub(hub: IntelligenceHub):
    """Gracefully shutdown the hub (idempotent)."""
    if not hub.is_running():
        return
    logging.info("Shutting down hub...")
    await hub.shutdown()
    logging.info("Hub shutdown complete")


def parse_args():
    """Parse command-line arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="HA Intelligence Hub - Adaptive home automation intelligence"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="FastAPI server port (default: 8000)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="FastAPI server host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=os.path.expanduser("~/ha-logs/intelligence/cache"),
        help="Cache directory path"
    )

    return parser.parse_args()


async def main():
    """Main entry point."""
    global hub_instance

    # Parse arguments
    args = parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger("main")

    # Setup cache path
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "hub.db"

    logger.info("=" * 70)
    logger.info("HA Intelligence Hub v0.1.0")
    logger.info("=" * 70)
    logger.info(f"Cache: {cache_path}")
    logger.info(f"Server: http://{args.host}:{args.port}")
    logger.info(f"WebSocket: ws://{args.host}:{args.port}/ws")
    logger.info(f"Log level: {args.log_level}")
    logger.info("=" * 70)

    # Initialize hub
    try:
        hub_instance = await start_hub(str(cache_path))
        logger.info("Hub initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize hub: {e}")
        return 1

    # Get HA credentials from environment
    ha_url = os.environ.get("HA_URL")
    ha_token = os.environ.get("HA_TOKEN")

    if not ha_url or not ha_token:
        logger.error("HA_URL and HA_TOKEN environment variables required")
        logger.error("Source ~/.env before running or export them manually")
        await shutdown_hub(hub_instance)
        return 1

    # Register and initialize discovery module
    try:
        logger.info("Initializing discovery module...")
        discovery = DiscoveryModule(hub_instance, ha_url, ha_token)
        hub_instance.register_module(discovery)
        await discovery.initialize()

        # Schedule periodic discovery (every 24 hours)
        await discovery.schedule_periodic_discovery(interval_hours=24)

        # Start event-driven discovery (WebSocket listener for registry changes)
        try:
            await discovery.start_event_listener()
        except Exception as e:
            logger.warning(f"Event listener failed to start (non-fatal): {e}")

        logger.info("Discovery module ready")
    except Exception as e:
        logger.error(f"Failed to initialize discovery module: {e}")
        await shutdown_hub(hub_instance)
        return 1

    # Register and initialize ML engine
    try:
        logger.info("Initializing ML engine...")
        models_dir = os.path.join(args.cache_dir, "..", "models")
        training_data_dir = os.path.join(args.cache_dir, "..", "daily")

        ml_engine = MLEngine(hub_instance, models_dir, training_data_dir)
        hub_instance.register_module(ml_engine)
        await ml_engine.initialize()

        # Schedule periodic training (every 7 days)
        await ml_engine.schedule_periodic_training(interval_days=7)

        logger.info("ML engine ready")
    except Exception as e:
        logger.error(f"Failed to initialize ML engine: {e}")
        await shutdown_hub(hub_instance)
        return 1

    # Register and initialize pattern recognition
    try:
        logger.info("Initializing pattern recognition...")
        log_dir = Path(os.path.join(args.cache_dir, ".."))
        patterns = PatternRecognition(hub_instance, log_dir)
        hub_instance.register_module(patterns)
        await patterns.initialize()

        logger.info("Pattern recognition ready")
    except Exception as e:
        logger.error(f"Failed to initialize pattern recognition: {e}")
        await shutdown_hub(hub_instance)
        return 1

    # Register and initialize orchestrator
    try:
        logger.info("Initializing orchestrator...")
        orchestrator = OrchestratorModule(hub_instance, ha_url, ha_token)
        hub_instance.register_module(orchestrator)
        await orchestrator.initialize()

        logger.info("Orchestrator ready")
    except Exception as e:
        logger.error(f"Failed to initialize orchestrator: {e}")
        await shutdown_hub(hub_instance)
        return 1

    # Register intelligence module (non-fatal — hub works without it)
    try:
        logger.info("Initializing intelligence module...")
        intelligence_dir = str(Path(args.cache_dir).parent)
        intel_mod = IntelligenceModule(hub_instance, intelligence_dir)
        hub_instance.register_module(intel_mod)
        await intel_mod.initialize()
        await intel_mod.schedule_refresh()
        logger.info("Intelligence module ready")
    except Exception as e:
        logger.warning(f"Intelligence module failed (non-fatal): {e}")

    # Register activity monitor (non-fatal — hub works without it)
    try:
        logger.info("Initializing activity monitor...")
        activity_monitor = ActivityMonitor(hub_instance, ha_url, ha_token)
        hub_instance.register_module(activity_monitor)
        await activity_monitor.initialize()
        logger.info("Activity monitor ready")
    except Exception as e:
        logger.warning(f"Activity monitor failed (non-fatal): {e}")

    # Create FastAPI app
    # Note: uvicorn handles SIGINT/SIGTERM internally — no custom signal handlers needed
    app = create_api(hub_instance)

    # Configure uvicorn
    config = uvicorn.Config(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
        access_log=True
    )
    server = uvicorn.Server(config)

    # Start server
    try:
        logger.info("Starting FastAPI server...")
        await server.serve()
    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt")
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        # Shutdown hub
        await shutdown_hub(hub_instance)

    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nShutdown by user")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
