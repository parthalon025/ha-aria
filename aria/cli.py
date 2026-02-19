"""ARIA CLI — unified entry point for batch engine and real-time hub."""

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser with all subcommands."""
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

    _add_serve_parser(subparsers)
    _add_demo_parser(subparsers)
    _add_status_parser(subparsers)

    # Simple commands
    subparsers.add_parser("sync-logs", help="Sync HA logbook to local JSON")

    _add_watchdog_parser(subparsers)
    _add_capabilities_parser(subparsers)
    _add_audit_parser(subparsers)

    return parser


def _add_serve_parser(subparsers):
    """Add serve subcommand parser."""
    serve_parser = subparsers.add_parser("serve", help="Start real-time hub and dashboard")
    serve_parser.add_argument("--port", type=int, default=8001, help="Port (default: 8001)")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Host (default: 127.0.0.1)")
    serve_parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")
    serve_parser.add_argument("--quiet", action="store_true", help="Only show WARNING and above")


def _add_demo_parser(subparsers):
    """Add demo subcommand parser."""
    demo_parser = subparsers.add_parser("demo", help="Generate synthetic demo data for visual testing")
    demo_parser.add_argument("--scenario", default="stable_couple", help="Household scenario (default: stable_couple)")
    demo_parser.add_argument("--days", type=int, default=30, help="Days to simulate (default: 30)")
    demo_parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    demo_parser.add_argument(
        "--checkpoint", type=str, default=None, help="Load a frozen checkpoint directory instead of generating"
    )
    demo_parser.add_argument("--port", type=int, default=8001, help="Port for hub (default: 8001)")
    demo_parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: temp directory)")


def _add_status_parser(subparsers):
    """Add status subcommand parser."""
    status_parser = subparsers.add_parser("status", help="Show ARIA hub status")
    status_parser.add_argument("--json", action="store_true", dest="json_output", help="Output as JSON")


def _add_watchdog_parser(subparsers):
    """Add watchdog subcommand parser."""
    wd_parser = subparsers.add_parser("watchdog", help="Run health checks and alert on failures")
    wd_parser.add_argument("--quiet", action="store_true", help="Log + alert only (no stdout)")
    wd_parser.add_argument("--no-alert", action="store_true", help="Skip Telegram alerts")
    wd_parser.add_argument("--json", action="store_true", dest="json_output", help="JSON output")


def _add_capabilities_parser(subparsers):
    """Add capabilities subcommand group parser."""
    cap_parser = subparsers.add_parser("capabilities", help="Capability registry management")
    cap_sub = cap_parser.add_subparsers(dest="cap_command")
    cap_list = cap_sub.add_parser("list", help="List all registered capabilities")
    cap_list.add_argument("--layer", choices=["hub", "engine", "dashboard", "cross-cutting"], help="Filter by layer")
    cap_list.add_argument("--status", choices=["stable", "experimental", "planned"], help="Filter by status")
    cap_list.add_argument("--verbose", action="store_true", help="Show full details")
    cap_verify = cap_sub.add_parser("verify", help="Validate capability declarations")
    cap_verify.add_argument("--strict", action="store_true", help="Fail on any issue (for CI)")
    cap_sub.add_parser("export", help="Export capabilities as JSON")


def _add_audit_parser(subparsers):
    """Add audit subcommand group parser."""
    import os

    audit_parser = subparsers.add_parser("audit", help="Query audit log")
    audit_sub = audit_parser.add_subparsers(dest="audit_command")

    ev_parser = audit_sub.add_parser("events", help="Query audit events")
    ev_parser.add_argument("--type", dest="event_type")
    ev_parser.add_argument("--source")
    ev_parser.add_argument("--subject")
    ev_parser.add_argument("--severity")
    ev_parser.add_argument("--since")
    ev_parser.add_argument("--until", dest="until_date")
    ev_parser.add_argument("--request-id")
    ev_parser.add_argument("--limit", type=int, default=50)
    ev_parser.add_argument("--json", action="store_true", dest="json_output")

    req_parser = audit_sub.add_parser("requests", help="Query API request log")
    req_parser.add_argument("--path")
    req_parser.add_argument("--method")
    req_parser.add_argument("--status", type=int, dest="status_min")
    req_parser.add_argument("--since")
    req_parser.add_argument("--limit", type=int, default=50)
    req_parser.add_argument("--json", action="store_true", dest="json_output")

    tl_parser = audit_sub.add_parser("timeline", help="Timeline for a subject")
    tl_parser.add_argument("subject")
    tl_parser.add_argument("--since")
    tl_parser.add_argument("--json", action="store_true", dest="json_output")

    stats_parser = audit_sub.add_parser("stats", help="Audit statistics")
    stats_parser.add_argument("--since")

    startups_parser = audit_sub.add_parser("startups", help="Recent hub startups")
    startups_parser.add_argument("--limit", type=int, default=10)

    cur_parser = audit_sub.add_parser("curation", help="Entity curation history")
    cur_parser.add_argument("entity_id")
    cur_parser.add_argument("--limit", type=int, default=50)

    verify_parser = audit_sub.add_parser("verify", help="Integrity check")
    verify_parser.add_argument("--since")

    export_parser = audit_sub.add_parser("export", help="Export archive")
    export_parser.add_argument("--before", required=True)
    export_parser.add_argument("--output", default=os.path.expanduser("~/ha-logs/intelligence/audit-archive"))

    tail_parser = audit_sub.add_parser("tail", help="Live tail audit events")
    tail_parser.add_argument("--types")
    tail_parser.add_argument("--severity-min", default="info")


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    _dispatch(args)


# Engine commands map subcommand names to old-style --flags
_ENGINE_COMMANDS = {
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


def _dispatch(args):
    """Route CLI commands to engine or hub functions."""
    if args.command in _ENGINE_COMMANDS:
        _dispatch_engine(args.command)
    elif args.command == "sequences":
        _dispatch_sequences(args)
    elif args.command == "serve":
        _dispatch_serve(args)
    elif args.command == "demo":
        _demo(args)
    elif args.command == "status":
        _status(json_output=args.json_output)
    elif args.command == "sync-logs":
        _sync_logs()
    elif args.command == "capabilities":
        _capabilities(args)
    elif args.command == "watchdog":
        _watchdog(args)
    elif args.command == "audit":
        _audit(args)
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


def _dispatch_engine(command):
    """Delegate to engine CLI with the old-style flag."""
    from aria.engine.cli import main as engine_main

    sys.argv = ["aria", _ENGINE_COMMANDS[command]]
    engine_main()


def _dispatch_sequences(args):
    """Handle sequences subcommand."""
    if args.seq_command == "train":
        sys.argv = ["aria", "--train-sequences"]
    elif args.seq_command == "detect":
        sys.argv = ["aria", "--sequence-anomalies"]
    else:
        print("Usage: aria sequences {train|detect}")
        sys.exit(1)
    from aria.engine.cli import main as engine_main

    engine_main()


def _dispatch_serve(args):
    """Resolve log level and start serve."""
    log_level = "INFO"
    if args.verbose:
        log_level = "DEBUG"
    elif args.quiet:
        log_level = "WARNING"
    _serve(args.host, args.port, log_level)


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

    from aria.hub.api import create_api
    from aria.hub.core import IntelligenceHub

    async def start():
        import time as _time

        # Setup cache
        cache_dir = Path(os.path.expanduser("~/ha-logs/intelligence/cache"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = str(cache_dir / "hub.db")

        _log_startup_banner(logger, cache_path, host, port)

        _start_time = _time.monotonic()

        hub = IntelligenceHub(cache_path)
        await hub.initialize()

        # Seed config defaults
        await _seed_config(hub, logger)

        # Audit logger
        audit_logger = await _init_audit_logger(hub, cache_dir, logger)

        # HA credentials
        ha_url = os.environ.get("HA_URL")
        ha_token = os.environ.get("HA_TOKEN")
        if not ha_url or not ha_token:
            logger.error("HA_URL and HA_TOKEN environment variables required")
            if audit_logger:
                await audit_logger.shutdown()
            await hub.shutdown()
            return

        intelligence_dir = str(cache_dir.parent)
        await _register_modules(hub, ha_url, ha_token, intelligence_dir, logger)

        # Module load summary
        _log_module_summary(hub, logger)

        # Log startup snapshot and schedule pruning
        if audit_logger:
            await _log_startup_snapshot(hub, audit_logger, _start_time, logger)
            await _schedule_audit_pruning(hub, audit_logger, logger)

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
            if audit_logger:
                await audit_logger.shutdown()
            if hub.is_running():
                await hub.shutdown()

    asyncio.run(start())


def _log_startup_banner(logger, cache_path: str, host: str, port: int) -> None:
    """Log the ARIA startup banner."""
    logger.info("=" * 70)
    logger.info("ARIA — Adaptive Residence Intelligence Architecture")
    logger.info("=" * 70)
    logger.info(f"Cache: {cache_path}")
    logger.info(f"Server: http://{host}:{port}")
    logger.info(f"WebSocket: ws://{host}:{port}/ws")
    logger.info("=" * 70)


async def _init_audit_logger(hub, cache_dir, logger):
    """Initialize AuditLogger, attach to hub, and return it (non-fatal)."""
    from aria.hub.audit import AuditLogger

    audit_logger = AuditLogger()
    audit_db_path = str(cache_dir / "audit.db")
    try:
        await audit_logger.initialize(audit_db_path)
        hub.set_audit_logger(audit_logger)
        logger.info(f"Audit: {audit_db_path}")
    except Exception as e:
        logger.warning(f"Audit logger failed to initialize (non-fatal): {e}")
        return None
    return audit_logger


async def _log_startup_snapshot(hub, audit_logger, start_time: float, logger) -> None:
    """Log a startup snapshot with modules, config, and elapsed duration."""
    import time as _time

    startup_duration = (_time.monotonic() - start_time) * 1000
    config_snapshot = {}
    try:
        all_config = await hub.cache.get_all_config()
        config_snapshot = {c["key"]: c["value"] for c in all_config}
    except Exception as e:
        logger.debug(f"Could not collect config snapshot for startup audit: {e}")
    await audit_logger.log_startup(
        modules=hub.module_status,
        config_snapshot=config_snapshot,
        duration_ms=startup_duration,
    )


async def _schedule_audit_pruning(hub, audit_logger, logger) -> None:
    """Register a 24h audit pruning task on the hub scheduler."""
    from datetime import timedelta

    async def _prune_audit():
        retention = 90
        try:
            config_val = await hub.cache.get_config_value("audit.retention_days")
            if config_val is not None:
                retention = int(config_val)
        except Exception:
            pass
        deleted = await audit_logger.prune(retention)
        if deleted:
            logger.info(f"Audit pruning: {deleted} records removed")

    await hub.schedule_task("prune_audit", _prune_audit, interval=timedelta(hours=24))


async def _seed_config(hub, logger):
    """Seed config defaults into the database (non-fatal)."""
    try:
        from aria.hub.config_defaults import seed_config_defaults

        seeded = await seed_config_defaults(hub.cache)
        if seeded:
            logger.info(f"Seeded {seeded} new config parameter(s)")
    except Exception as e:
        logger.warning(f"Config seeding failed (non-fatal): {e}")


def _make_init_module(hub):
    """Create a module initializer that tracks success/failure."""

    def _init_module(module, name):
        async def _do():
            try:
                await module.initialize()
                hub.mark_module_running(name)
            except Exception:
                hub.mark_module_failed(name)
                raise

        return _do

    return _init_module


async def _register_modules(hub, ha_url, ha_token, intelligence_dir, logger):
    """Register and initialize all hub modules in dependency tiers.

    Tier 0: discovery (must be first — other modules depend on its cache data)
    Tier 1: ml_engine, patterns, orchestrator (parallel — independent of each other)
    Tier 2: optional modules (parallel — non-fatal)
    """
    import asyncio
    import os
    from pathlib import Path

    from aria.modules.discovery import DiscoveryModule
    from aria.modules.ml_engine import MLEngine
    from aria.modules.orchestrator import OrchestratorModule
    from aria.modules.patterns import PatternRecognition

    models_dir = os.path.join(intelligence_dir, "models")
    training_data_dir = os.path.join(intelligence_dir, "daily")
    _init = _make_init_module(hub)

    # --- Tier 0: discovery (must complete before others) ---
    discovery = DiscoveryModule(hub, ha_url, ha_token)
    hub.register_module(discovery)
    await _init(discovery, "discovery")()
    await discovery.schedule_periodic_discovery(interval_hours=24)
    try:
        await discovery.start_event_listener()
    except Exception as e:
        logger.warning(f"Event listener failed to start (non-fatal): {e}")

    # --- Tier 1: parallel core modules ---
    ml_engine = MLEngine(hub, models_dir, training_data_dir)
    hub.register_module(ml_engine)

    log_dir = Path(intelligence_dir)
    patterns = PatternRecognition(hub, log_dir)
    hub.register_module(patterns)

    orchestrator = OrchestratorModule(hub, ha_url, ha_token)
    hub.register_module(orchestrator)

    async def _init_ml():
        await _init(ml_engine, "ml_engine")()
        await ml_engine.schedule_periodic_training(interval_days=7)

    tier1_results = await asyncio.gather(
        _init_ml(),
        _init(patterns, "pattern_recognition")(),
        _init(orchestrator, "orchestrator")(),
        return_exceptions=True,
    )
    for i, result in enumerate(tier1_results):
        if isinstance(result, Exception):
            names = ["ml_engine", "pattern_recognition", "orchestrator"]
            logger.error(f"Tier 1 module {names[i]} failed: {result}")

    # --- Tier 2: optional modules (parallel, non-fatal) ---
    await _register_optional_modules(hub, ha_url, ha_token, intelligence_dir, _init, logger)


async def _register_optional_modules(hub, ha_url, ha_token, intelligence_dir, _init, logger):
    """Register optional modules that are non-fatal on failure (parallel)."""
    import asyncio

    tier2_results = await asyncio.gather(
        _register_analysis_modules(hub, intelligence_dir, _init, logger),
        _register_monitor_modules(hub, ha_url, ha_token, _init, logger),
        return_exceptions=True,
    )
    for i, result in enumerate(tier2_results):
        if isinstance(result, Exception):
            names = ["analysis_modules", "monitor_modules"]
            logger.error(f"Tier 2 module group {names[i]} failed: {result}")


async def _register_analysis_modules(hub, intelligence_dir, _init, logger):
    """Register analysis modules (shadow, data quality, organic discovery, intelligence)."""
    from aria.modules.intelligence import IntelligenceModule
    from aria.modules.shadow_engine import ShadowEngine

    # shadow_engine
    try:
        shadow_engine = ShadowEngine(hub)
        hub.register_module(shadow_engine)
        await _init(shadow_engine, "shadow_engine")()
    except Exception as e:
        logger.error(f"Shadow engine failed (hub continues without it): {e}")

    # intelligence
    intel_mod = IntelligenceModule(hub, intelligence_dir)
    hub.register_module(intel_mod)
    try:
        await _init(intel_mod, "intelligence")()
        await intel_mod.schedule_refresh()
    except Exception as e:
        logger.warning(f"Intelligence module failed (non-fatal): {e}")

    await _register_ml_modules(hub, _init, logger)


async def _register_ml_modules(hub, _init, logger):
    """Register ML modules (Tier 3+ — each self-gates on hardware tier)."""
    # trajectory_classifier
    try:
        from aria.modules.trajectory_classifier import TrajectoryClassifier

        trajectory_classifier = TrajectoryClassifier(hub)
        hub.register_module(trajectory_classifier)
        await _init(trajectory_classifier, "trajectory_classifier")()
    except Exception as e:
        logger.warning(f"Trajectory classifier module failed (non-fatal): {e}")


async def _register_monitor_modules(hub, ha_url, ha_token, _init, logger):
    """Register monitoring modules (activity monitor, labeler, presence)."""
    import os

    # activity_monitor
    try:
        from aria.modules.activity_monitor import ActivityMonitor

        activity_monitor = ActivityMonitor(hub, ha_url, ha_token)
        hub.register_module(activity_monitor)
        await _init(activity_monitor, "activity_monitor")()
    except Exception as e:
        logger.warning(f"Activity monitor failed (non-fatal): {e}")

    # presence
    try:
        from aria.modules.presence import PresenceModule

        presence = PresenceModule(
            hub,
            ha_url,
            ha_token,
            mqtt_host=os.environ.get("MQTT_HOST", ""),
            mqtt_user=os.environ.get("MQTT_USER", ""),
            mqtt_password=os.environ.get("MQTT_PASSWORD", ""),
        )
        hub.register_module(presence)
        await _init(presence, "presence")()
    except Exception as e:
        logger.warning(f"Presence module failed (non-fatal): {e}")


def _log_module_summary(hub, logger):
    """Log a summary of module load status."""
    total = len(hub.module_status)
    running = sum(1 for s in hub.module_status.values() if s == "running")
    failed = [mid for mid, s in hub.module_status.items() if s == "failed"]
    if failed:
        logger.warning(f"Loaded {running}/{total} modules ({', '.join(failed)} failed)")
    else:
        logger.info(f"Loaded {running}/{total} modules (all healthy)")


def _demo(args):
    """Generate or load synthetic demo data for visual testing."""
    import tempfile
    from pathlib import Path

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
    print("\nTo start the hub with this data, point intelligence_dir at:")
    print(f"  {data_dir}")
    print("\nContents:")
    for child in sorted(data_dir.iterdir()):
        if child.is_dir():
            count = len(list(child.iterdir()))
            print(f"  {child.name}/ ({count} files)")
        else:
            print(f"  {child.name}")


def _status(json_output: bool = False):
    """Show ARIA hub status — checks running hub, cache, snapshots, models."""
    import json

    result = _collect_status_data()

    if json_output:
        print(json.dumps(result, indent=2))
    else:
        _print_status(result)


def _collect_status_data() -> dict:
    """Collect all status data from hub, cache, snapshots, and models."""
    import json
    import os
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
    result["cache_categories"] = _get_cache_category_count(result["hub_health"], intelligence_dir)

    # Last snapshot (newest file in daily/)
    result["last_snapshot"] = _newest_file_mtime(intelligence_dir / "daily", "*.jsonl")

    # Last training (newest model file)
    result["last_training"] = _newest_file_mtime(intelligence_dir / "models", "*.joblib")

    return result


def _get_cache_category_count(hub_health, intelligence_dir) -> int:
    """Get cache category count from hub health or DB file."""
    if hub_health:
        cats = hub_health.get("cache", {}).get("categories", [])
        return len(cats)

    db_path = intelligence_dir / "cache" / "hub.db"
    if db_path.exists():
        try:
            import sqlite3

            conn = sqlite3.connect(str(db_path))
            cursor = conn.execute("SELECT COUNT(*) FROM cache")
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except Exception:
            pass
    return 0


def _newest_file_mtime(directory, glob_pattern: str) -> str | None:
    """Get ISO timestamp of the newest file matching the glob in directory."""
    from datetime import datetime

    if not directory.exists():
        return None
    files = sorted(directory.glob(glob_pattern), key=lambda f: f.stat().st_mtime, reverse=True)
    if files:
        mtime = datetime.fromtimestamp(files[0].stat().st_mtime)
        return mtime.isoformat()
    return None


def _print_status(result: dict):
    """Print human-readable status output."""
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


def _sync_logs():
    """Run ha-log-sync."""
    import os
    import subprocess

    bin_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sync_script = os.path.join(bin_dir, "bin", "ha-log-sync")
    subprocess.run([sys.executable, sync_script], check=True)


def _watchdog(args):
    """Run watchdog health checks."""
    from aria.watchdog import run_watchdog

    ret = run_watchdog(
        quiet=args.quiet,
        no_alert=args.no_alert,
        json_output=args.json_output,
    )
    sys.exit(ret)


def _capabilities(args):
    """Handle the capabilities subcommand group (list, verify, export)."""
    from aria.capabilities import CapabilityRegistry

    registry = CapabilityRegistry()
    registry.collect_from_modules()

    cap_command = getattr(args, "cap_command", None)

    if cap_command == "list":
        _capabilities_list(registry, args)
    elif cap_command == "verify":
        _capabilities_verify(registry, args)
    elif cap_command == "export":
        _capabilities_export(registry)
    else:
        print("Usage: aria capabilities {list|verify|export}")
        sys.exit(1)


def _capabilities_list(registry, args):
    """List capabilities, optionally filtered by layer/status."""
    caps = registry.list_all()

    if args.layer:
        caps = [c for c in caps if c.layer == args.layer]
    if args.status:
        caps = [c for c in caps if c.status == args.status]

    # Group by layer
    by_layer = {}
    for cap in caps:
        by_layer.setdefault(cap.layer, []).append(cap)

    layer_order = ["hub", "engine", "dashboard", "cross-cutting"]
    total = 0

    for layer in layer_order:
        layer_caps = by_layer.get(layer, [])
        if not layer_caps:
            continue
        layer_label = layer.replace("-", " ").title()
        print(f"\n{layer_label} Layer ({len(layer_caps)})")
        _print_layer_caps(layer_caps, args.verbose)
        total += len(layer_caps)

    print(f"\nTotal: {total} capabilities")


def _print_layer_caps(layer_caps, verbose: bool):
    """Print capabilities for a single layer."""
    if verbose:
        for cap in sorted(layer_caps, key=lambda c: c.id):
            _print_cap_detail(cap)
    else:
        print(f"  {'ID':<30}{'Name':<38}{'Status':<14}{'Tests':<7}{'Config'}")
        print(f"  {'─' * 28}  {'─' * 36}  {'─' * 12}  {'─' * 5}  {'─' * 6}")
        for cap in sorted(layer_caps, key=lambda c: c.id):
            print(f"  {cap.id:<30}{cap.name:<38}{cap.status:<14}{len(cap.test_paths):<7}{len(cap.config_keys)}")


def _print_cap_detail(cap):
    """Print verbose detail for a single capability."""
    print(f"  {cap.id}")
    print(f"    Name:        {cap.name}")
    print(f"    Description: {cap.description}")
    print(f"    Status:      {cap.status}")
    print(f"    Module:      {cap.module}")
    print(f"    Tests:       {len(cap.test_paths)}")
    print(f"    Config keys: {len(cap.config_keys)}")
    if cap.depends_on:
        print(f"    Depends on:  {', '.join(cap.depends_on)}")
    if cap.pipeline_stage:
        print(f"    Pipeline:    {cap.pipeline_stage}")


def _capabilities_verify(registry, args):
    """Validate all capability declarations."""
    issues = registry.validate_all()

    if issues:
        for issue in issues:
            print(f"  \u2717 {issue}")
        print(f"\n{len(issues)} issue(s) found")
        if args.strict:
            sys.exit(1)
    else:
        total = len(registry.list_all())
        print(f"All {total} capabilities pass validation.")


def _capabilities_export(registry):
    """Export all capabilities as JSON."""
    import json
    from collections import Counter
    from dataclasses import asdict

    caps = registry.list_all()
    by_layer = Counter(c.layer for c in caps)
    by_status = Counter(c.status for c in caps)

    output = {
        "capabilities": [asdict(c) for c in caps],
        "total": len(caps),
        "by_layer": dict(by_layer),
        "by_status": dict(by_status),
    }
    print(json.dumps(output, indent=2))


def _audit(args):
    """Handle audit subcommands — open audit DB read-only and query."""
    import asyncio
    import os

    from aria.hub.audit import AuditLogger

    audit_db = os.path.expanduser("~/ha-logs/intelligence/cache/audit.db")
    if not os.path.exists(audit_db):
        print("No audit database found. Start the hub first to initialize audit logging.")
        sys.exit(1)

    async def run():
        al = AuditLogger()
        await al.initialize(audit_db)
        try:
            await _dispatch_audit(al, args)
        except Exception as e:
            print(f"Audit query failed: {e}")
            sys.exit(1)
        finally:
            await al.shutdown()

    asyncio.run(run())


async def _dispatch_audit(al, args):  # noqa: PLR0915
    """Route audit subcommands to query methods."""
    import json

    cmd = getattr(args, "audit_command", None)

    if cmd == "events":
        results = await al.query_events(
            event_type=getattr(args, "event_type", None),
            source=getattr(args, "source", None),
            subject=getattr(args, "subject", None),
            severity=getattr(args, "severity", None),
            since=getattr(args, "since", None),
            until=getattr(args, "until_date", None),
            request_id=getattr(args, "request_id", None),
            limit=args.limit,
        )
        _print_audit_results(results, getattr(args, "json_output", False))

    elif cmd == "requests":
        results = await al.query_requests(
            path=getattr(args, "path", None),
            method=getattr(args, "method", None),
            status_min=getattr(args, "status_min", None),
            since=getattr(args, "since", None),
            limit=args.limit,
        )
        _print_audit_results(results, getattr(args, "json_output", False))

    elif cmd == "timeline":
        results = await al.query_timeline(
            args.subject,
            since=getattr(args, "since", None),
        )
        _print_audit_results(results, getattr(args, "json_output", False))

    elif cmd == "stats":
        result = await al.get_stats(since=getattr(args, "since", None))
        print(json.dumps(result, indent=2))

    elif cmd == "startups":
        results = await al.query_startups(limit=args.limit)
        _print_audit_results(results, json_output=True)  # JSON-only: nested modules/config fields

    elif cmd == "curation":
        results = await al.query_curation(args.entity_id, limit=args.limit)
        _print_audit_results(results, json_output=True)  # JSON-only: mixed field types

    elif cmd == "verify":
        result = await al.verify_integrity(since=getattr(args, "since", None))
        total = result["total"]
        invalid = result["invalid"]
        if invalid:
            print(f"INTEGRITY CHECK FAILED: {invalid}/{total} events with invalid checksums")
            for e in result.get("details", []):
                exp = e.get("expected", "?")[:12]
                act = e.get("actual", "?")[:12]
                print(f"  ID {e.get('id', '?')}: expected {exp}... actual {act}...")
        else:
            print(f"Integrity OK: {total} events verified")

    elif cmd == "export":
        from datetime import UTC, datetime

        try:
            parsed = datetime.fromisoformat(args.before)
            before_dt = parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        except ValueError:
            print(f"Invalid date format for --before: {args.before!r}. Use ISO format, e.g. 2026-01-01T00:00:00")
            sys.exit(1)

        import os

        output_dir = args.output
        os.makedirs(output_dir, exist_ok=True)
        files = await al.export_archive(before_dt, output_dir)
        print(f"Exported {len(files)} file(s) to {output_dir}")
        for f in files:
            print(f"  {f}")

    elif cmd == "tail":
        print("Live tail requires the hub to be running. Use the WebSocket endpoint instead.")
        print("  ws://127.0.0.1:8001/ws/audit")

    else:
        print("Usage: aria audit {events|requests|timeline|stats|startups|curation|verify|export|tail}")
        sys.exit(1)


def _print_audit_results(results, json_output: bool = False):
    """Print audit query results as JSON or simple table."""
    import json

    if json_output or not results:
        print(json.dumps(results, indent=2, default=str))
        return

    keys = list(results[0].keys())
    # Print header (up to first 5 columns)
    header = " | ".join(f"{k:<20}" for k in keys[:5])
    print(header)
    print("-" * len(header))
    for row in results:
        vals = []
        for k in keys[:5]:
            v = row.get(k, "")
            s = str(v)[:20]
            vals.append(f"{s:<20}")
        print(" | ".join(vals))
    if len(keys) > 5:
        print(f"\n({len(keys) - 5} more columns not shown. Use --json for full output.)")


if __name__ == "__main__":
    main()
