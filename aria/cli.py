"""ARIA CLI — unified entry point for batch engine and real-time hub."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="aria",
        description="ARIA — Adaptive Residence Intelligence Architecture",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Batch engine commands
    subparsers.add_parser("snapshot", help="Collect current HA state snapshot")
    subparsers.add_parser("predict", help="Generate predictions from latest snapshot")
    subparsers.add_parser("full", help="Full daily pipeline: snapshot -> predict -> report")
    subparsers.add_parser("score", help="Score yesterday's predictions against actuals")
    subparsers.add_parser("retrain", help="Retrain ML models from accumulated data")
    subparsers.add_parser("meta-learn", help="LLM meta-learning to tune feature config")
    subparsers.add_parser("check-drift", help="Detect concept drift in predictions")
    subparsers.add_parser("correlations", help="Compute entity co-occurrence correlations")
    subparsers.add_parser("suggest-automations", help="Generate HA automation YAML via LLM")
    subparsers.add_parser("prophet", help="Train Prophet seasonal forecasters")
    subparsers.add_parser("occupancy", help="Bayesian occupancy estimation")
    subparsers.add_parser("power-profiles", help="Analyze per-outlet power consumption")

    # Sequence sub-commands
    seq_parser = subparsers.add_parser("sequences", help="Markov chain sequence analysis")
    seq_sub = seq_parser.add_subparsers(dest="seq_command")
    seq_sub.add_parser("train", help="Train Markov chain model from logbook sequences")
    seq_sub.add_parser("detect", help="Detect anomalous event sequences")

    # Intraday snapshot (used by hub subprocess)
    subparsers.add_parser("snapshot-intraday", help="Collect intraday snapshot (internal)")

    # Hub serve command
    serve_parser = subparsers.add_parser("serve", help="Start real-time hub and dashboard")
    serve_parser.add_argument("--port", type=int, default=8001, help="Port (default: 8001)")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Host (default: 127.0.0.1)")
    serve_parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")
    serve_parser.add_argument("--quiet", action="store_true", help="Only show WARNING and above")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Generate synthetic demo data for visual testing")
    demo_parser.add_argument("--scenario", default="stable_couple", help="Household scenario (default: stable_couple)")
    demo_parser.add_argument("--days", type=int, default=30, help="Days to simulate (default: 30)")
    demo_parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    demo_parser.add_argument("--checkpoint", type=str, default=None, help="Load a frozen checkpoint directory instead of generating")
    demo_parser.add_argument("--port", type=int, default=8001, help="Port for hub (default: 8001)")
    demo_parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: temp directory)")

    # Status command
    status_parser = subparsers.add_parser("status", help="Show ARIA hub status")
    status_parser.add_argument("--json", action="store_true", dest="json_output", help="Output as JSON")

    # Organic discovery
    subparsers.add_parser("discover-organic", help="Run organic capability discovery pipeline")

    # Log sync
    subparsers.add_parser("sync-logs", help="Sync HA logbook to local JSON")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Dispatch to engine commands
    _dispatch(args)


def _dispatch(args):
    """Route CLI commands to engine or hub functions."""
    # Engine commands reuse the existing engine CLI logic
    ENGINE_COMMANDS = {
        "snapshot": "--snapshot",
        "predict": "--predict",
        "full": "--full",
        "score": "--score",
        "retrain": "--retrain",
        "meta-learn": "--meta-learn",
        "check-drift": "--check-drift",
        "correlations": "--entity-correlations",
        "suggest-automations": "--suggest-automations",
        "prophet": "--train-prophet",
        "occupancy": "--occupancy",
        "power-profiles": "--power-profiles",
        "snapshot-intraday": "--snapshot-intraday",
    }

    if args.command in ENGINE_COMMANDS:
        # Delegate to engine CLI with the old-style flag
        from aria.engine.cli import main as engine_main

        sys.argv = ["aria", ENGINE_COMMANDS[args.command]]
        engine_main()

    elif args.command == "sequences":
        if args.seq_command == "train":
            sys.argv = ["aria", "--train-sequences"]
        elif args.seq_command == "detect":
            sys.argv = ["aria", "--sequence-anomalies"]
        else:
            print("Usage: aria sequences {train|detect}")
            sys.exit(1)
        from aria.engine.cli import main as engine_main

        engine_main()

    elif args.command == "serve":
        log_level = "INFO"
        if args.verbose:
            log_level = "DEBUG"
        elif args.quiet:
            log_level = "WARNING"
        _serve(args.host, args.port, log_level)

    elif args.command == "demo":
        _demo(args)

    elif args.command == "status":
        _status(json_output=args.json_output)

    elif args.command == "discover-organic":
        _discover_organic()

    elif args.command == "sync-logs":
        _sync_logs()

    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


def _serve(host: str, port: int, log_level: str = "INFO"):
    """Start the ARIA real-time hub."""
    import asyncio
    import logging
    import os
    from pathlib import Path

    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("aria.serve")

    import uvicorn
    from aria.hub.core import IntelligenceHub
    from aria.hub.api import create_api
    from aria.modules.discovery import DiscoveryModule
    from aria.modules.ml_engine import MLEngine
    from aria.modules.orchestrator import OrchestratorModule
    from aria.modules.patterns import PatternRecognition
    from aria.modules.intelligence import IntelligenceModule
    from aria.modules.activity_monitor import ActivityMonitor
    from aria.modules.shadow_engine import ShadowEngine

    async def start():
        # Setup cache
        cache_dir = Path(os.path.expanduser("~/ha-logs/intelligence/cache"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = str(cache_dir / "hub.db")

        logger.info("=" * 70)
        logger.info("ARIA — Adaptive Residence Intelligence Architecture")
        logger.info("=" * 70)
        logger.info(f"Cache: {cache_path}")
        logger.info(f"Server: http://{host}:{port}")
        logger.info(f"WebSocket: ws://{host}:{port}/ws")
        logger.info("=" * 70)

        hub = IntelligenceHub(cache_path)
        await hub.initialize()

        # Seed config defaults
        try:
            from aria.hub.config_defaults import seed_config_defaults

            seeded = await seed_config_defaults(hub.cache)
            if seeded:
                logger.info(f"Seeded {seeded} new config parameter(s)")
        except Exception as e:
            logger.warning(f"Config seeding failed (non-fatal): {e}")

        # HA credentials
        ha_url = os.environ.get("HA_URL")
        ha_token = os.environ.get("HA_TOKEN")
        if not ha_url or not ha_token:
            logger.error("HA_URL and HA_TOKEN environment variables required")
            await hub.shutdown()
            return

        intelligence_dir = str(cache_dir.parent)
        models_dir = os.path.join(intelligence_dir, "models")
        training_data_dir = os.path.join(intelligence_dir, "daily")

        # Register and initialize modules, tracking success/failure
        def _init_module(module, name):
            """Register and mark module status after init attempt."""

            async def _do():
                try:
                    await module.initialize()
                    hub.mark_module_running(name)
                except Exception:
                    hub.mark_module_failed(name)
                    raise

            return _do

        # discovery
        discovery = DiscoveryModule(hub, ha_url, ha_token)
        hub.register_module(discovery)
        await _init_module(discovery, "discovery")()
        await discovery.schedule_periodic_discovery(interval_hours=24)
        try:
            await discovery.start_event_listener()
        except Exception as e:
            logger.warning(f"Event listener failed to start (non-fatal): {e}")

        # ml_engine
        ml_engine = MLEngine(hub, models_dir, training_data_dir)
        hub.register_module(ml_engine)
        await _init_module(ml_engine, "ml_engine")()
        await ml_engine.schedule_periodic_training(interval_days=7)

        # pattern_recognition
        log_dir = Path(intelligence_dir)
        patterns = PatternRecognition(hub, log_dir)
        hub.register_module(patterns)
        await _init_module(patterns, "pattern_recognition")()

        # orchestrator
        orchestrator = OrchestratorModule(hub, ha_url, ha_token)
        hub.register_module(orchestrator)
        await _init_module(orchestrator, "orchestrator")()

        # shadow_engine (non-fatal)
        try:
            shadow_engine = ShadowEngine(hub)
            hub.register_module(shadow_engine)
            await _init_module(shadow_engine, "shadow_engine")()
        except Exception as e:
            logger.error(f"Shadow engine failed (hub continues without it): {e}")

        # data_quality (non-fatal)
        try:
            from aria.modules.data_quality import DataQualityModule

            data_quality = DataQualityModule(hub)
            hub.register_module(data_quality)
            await _init_module(data_quality, "data_quality")()
        except Exception as e:
            logger.warning(f"Data quality module failed (non-fatal): {e}")

        # organic_discovery (non-fatal)
        try:
            from aria.modules.organic_discovery.module import OrganicDiscoveryModule
            organic_discovery = OrganicDiscoveryModule(hub)
            hub.register_module(organic_discovery)
            await _init_module(organic_discovery, "organic_discovery")()
        except Exception as e:
            logger.warning(f"Organic discovery module failed (non-fatal): {e}")

        # intelligence (non-fatal)
        intel_mod = IntelligenceModule(hub, intelligence_dir)
        hub.register_module(intel_mod)
        try:
            await _init_module(intel_mod, "intelligence")()
            await intel_mod.schedule_refresh()
        except Exception as e:
            logger.warning(f"Intelligence module failed (non-fatal): {e}")

        # activity_monitor (non-fatal)
        try:
            activity_monitor = ActivityMonitor(hub, ha_url, ha_token)
            hub.register_module(activity_monitor)
            await _init_module(activity_monitor, "activity_monitor")()
        except Exception as e:
            logger.warning(f"Activity monitor failed (non-fatal): {e}")

        # Module load summary
        total = len(hub.module_status)
        running = sum(1 for s in hub.module_status.values() if s == "running")
        failed = [mid for mid, s in hub.module_status.items() if s == "failed"]
        if failed:
            logger.warning(f"Loaded {running}/{total} modules ({', '.join(failed)} failed)")
        else:
            logger.info(f"Loaded {running}/{total} modules (all healthy)")

        app = create_api(hub)

        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level=log_level.lower(),
            access_log=(log_level != "WARNING"),
        )
        server = uvicorn.Server(config)

        try:
            await server.serve()
        finally:
            if hub.is_running():
                await hub.shutdown()

    asyncio.run(start())


def _demo(args):
    """Generate or load synthetic demo data for visual testing."""
    from pathlib import Path
    import tempfile

    if args.checkpoint:
        data_dir = Path(args.checkpoint)
        if not data_dir.exists():
            print(f"Error: checkpoint directory not found: {data_dir}")
            sys.exit(1)
        print(f"Using checkpoint: {data_dir}")
    else:
        output_dir = Path(args.output_dir) if args.output_dir else Path(tempfile.mkdtemp(prefix="aria-demo-"))
        print(f"Generating {args.days}-day '{args.scenario}' scenario (seed={args.seed})...")

        from tests.demo.generate import generate_checkpoint

        result = generate_checkpoint(
            scenario=args.scenario,
            days=args.days,
            seed=args.seed,
            output_dir=output_dir,
        )
        data_dir = output_dir
        print(f"Generated {result['snapshots_saved']} snapshots")
        print(f"  Training metrics: {len(result.get('training', {}))}")
        print(f"  Predictions: {len(result.get('predictions', {}))}")

    print(f"\nDemo data directory: {data_dir}")
    print(f"\nTo start the hub with this data, point intelligence_dir at:")
    print(f"  {data_dir}")
    print(f"\nContents:")
    for child in sorted(data_dir.iterdir()):
        if child.is_dir():
            count = len(list(child.iterdir()))
            print(f"  {child.name}/ ({count} files)")
        else:
            print(f"  {child.name}")


def _status(json_output: bool = False):
    """Show ARIA hub status — checks running hub, cache, snapshots, models."""
    import json
    import os
    from datetime import datetime
    from pathlib import Path

    from aria import __version__

    intelligence_dir = Path(os.path.expanduser("~/ha-logs/intelligence"))
    result = {
        "version": __version__,
        "hub_running": False,
        "hub_health": None,
        "cache_categories": 0,
        "last_snapshot": None,
        "last_training": None,
    }

    # Check if hub is running by hitting /health
    try:
        import urllib.request

        req = urllib.request.Request("http://127.0.0.1:8001/health", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            health = json.loads(resp.read())
            result["hub_running"] = True
            result["hub_health"] = health
    except Exception:
        pass

    # Count cache categories from hub if running, else check DB file
    if result["hub_health"]:
        cats = result["hub_health"].get("cache", {}).get("categories", [])
        result["cache_categories"] = len(cats)
    else:
        db_path = intelligence_dir / "cache" / "hub.db"
        if db_path.exists():
            try:
                import sqlite3

                conn = sqlite3.connect(str(db_path))
                cursor = conn.execute("SELECT COUNT(*) FROM cache")
                result["cache_categories"] = cursor.fetchone()[0]
                conn.close()
            except Exception:
                pass

    # Last snapshot (newest file in daily/)
    daily_dir = intelligence_dir / "daily"
    if daily_dir.exists():
        files = sorted(daily_dir.glob("*.jsonl"), key=lambda f: f.stat().st_mtime, reverse=True)
        if files:
            mtime = datetime.fromtimestamp(files[0].stat().st_mtime)
            result["last_snapshot"] = mtime.isoformat()

    # Last training (newest model file)
    models_dir = intelligence_dir / "models"
    if models_dir.exists():
        model_files = sorted(models_dir.glob("*.joblib"), key=lambda f: f.stat().st_mtime, reverse=True)
        if model_files:
            mtime = datetime.fromtimestamp(model_files[0].stat().st_mtime)
            result["last_training"] = mtime.isoformat()

    if json_output:
        print(json.dumps(result, indent=2))
        return

    # Pretty table output
    print("ARIA Status")
    print("=" * 40)
    print(f"  Version:          {result['version']}")
    hub_status = "running" if result["hub_running"] else "stopped"
    print(f"  Hub:              {hub_status}")
    if result["hub_health"]:
        modules = result["hub_health"].get("modules", {})
        running = sum(1 for s in modules.values() if s == "running")
        total = len(modules)
        print(f"  Modules:          {running}/{total} running")
        uptime = result["hub_health"].get("uptime_seconds", 0)
        hours, remainder = divmod(int(uptime), 3600)
        minutes, secs = divmod(remainder, 60)
        print(f"  Uptime:           {hours}h {minutes}m {secs}s")
    print(f"  Cache categories: {result['cache_categories']}")
    print(f"  Last snapshot:    {result['last_snapshot'] or 'none'}")
    print(f"  Last training:    {result['last_training'] or 'none'}")


def _discover_organic():
    """Run the organic capability discovery pipeline (batch mode)."""
    import asyncio
    import logging
    import os
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("aria.discover-organic")

    async def run():
        cache_dir = Path(os.path.expanduser("~/ha-logs/intelligence/cache"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = str(cache_dir / "hub.db")

        from aria.hub.core import IntelligenceHub
        from aria.modules.organic_discovery.module import OrganicDiscoveryModule

        hub = IntelligenceHub(cache_path)
        await hub.initialize()

        try:
            module = OrganicDiscoveryModule(hub)
            hub.register_module(module)
            await module.initialize()

            result = await module.run_discovery()
            logger.info(f"Organic discovery complete: {result}")
        finally:
            if hub.is_running():
                await hub.shutdown()

    asyncio.run(run())


def _sync_logs():
    """Run ha-log-sync."""
    import subprocess
    import os

    bin_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sync_script = os.path.join(bin_dir, "bin", "ha-log-sync")
    subprocess.run([sys.executable, sync_script], check=True)


if __name__ == "__main__":
    main()
