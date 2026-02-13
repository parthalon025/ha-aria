"""CLI entry point for ARIA engine (batch ML pipeline).

Wires all modules together and dispatches commands via argparse-style flags.
Called internally by ``aria <subcommand>`` — see aria/cli.py for the public CLI.

Usage (internal flag-style, called by aria CLI dispatcher):
  aria snapshot           # Collect today's daily snapshot
  aria snapshot-intraday  # Collect intra-day snapshot (current hour)
  aria predict            # Generate predictions for tomorrow
  aria score              # Score yesterday's predictions
  aria retrain            # Retrain sklearn ML models
  aria meta-learn         # LLM meta-learning to tune feature config
  aria check-drift        # Check for concept drift, conditionally retrain
  aria correlations       # Compute entity co-occurrence patterns
  aria suggest-automations # Generate HA automation YAML from patterns
  aria prophet            # Train Prophet seasonal forecasters
  aria occupancy          # Bayesian occupancy estimation
  aria power-profiles     # Analyze per-outlet power profiles
  aria sequences train    # Train Markov chain from logbook events
  aria sequences detect   # Detect anomalous event sequences
  aria full               # Full daily pipeline: snapshot → predict → report
"""

import json
import sys
from datetime import datetime, timedelta

from aria.engine.config import AppConfig
from aria.engine.storage.data_store import DataStore

# sklearn availability check (same pattern as v2)
HAS_SKLEARN = True
try:
    import numpy as np  # noqa: F401
except ImportError:
    HAS_SKLEARN = False


def _init():
    """Initialize config and data store from environment."""
    config = AppConfig.from_env()
    store = DataStore(config.paths)
    store.ensure_dirs()
    return config, store


# ── Commands ──────────────────────────────────────────────────────────


def cmd_snapshot_intraday():
    """Collect and save an intra-day snapshot."""
    config, store = _init()

    from aria.engine.collectors.snapshot import build_intraday_snapshot
    from aria.engine.features.time_features import build_time_features

    snapshot = build_intraday_snapshot(hour=None, date_str=None,
                                       config=config, store=store)
    # Add time features (wiring that was deferred during migration)
    timestamp = snapshot.get("timestamp", datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))
    snapshot["time_features"] = build_time_features(
        timestamp, snapshot.get("sun"), snapshot.get("date"))

    path = store.save_intraday_snapshot(snapshot)
    entities = snapshot.get("entities", {}).get("total", 0)
    hour = snapshot.get("hour", "?")
    print(f"Intraday snapshot saved: {path} ({entities} entities, hour={hour})")
    return snapshot


def cmd_snapshot():
    """Collect and save today's snapshot."""
    config, store = _init()

    from aria.engine.collectors.snapshot import build_snapshot

    snapshot = build_snapshot(date_str=None, config=config, store=store)
    path = store.save_snapshot(snapshot)
    print(f"Snapshot saved: {path} ({snapshot['entities']['total']} entities)")
    return snapshot


def cmd_analyze():
    """Run full analysis on latest data, including ML contextual anomaly detection."""
    config, store = _init()

    from aria.engine.collectors.snapshot import build_snapshot
    from aria.engine.analysis.baselines import compute_baselines
    from aria.engine.analysis.anomalies import detect_anomalies
    from aria.engine.analysis.correlations import cross_correlate
    from aria.engine.analysis.reliability import compute_device_reliability
    from aria.engine.features.vector_builder import build_feature_vector, get_feature_names
    from aria.engine.features.feature_config import load_feature_config
    from aria.engine.models.device_failure import detect_contextual_anomalies
    from aria.engine.models.training import count_days_of_data

    today = datetime.now().strftime("%Y-%m-%d")
    snapshot = store.load_snapshot(today)
    if not snapshot:
        snapshot = build_snapshot(date_str=today, config=config, store=store)

    recent = store.load_recent_snapshots(30)
    baselines = compute_baselines(recent)
    store.save_baselines(baselines)

    anomalies = detect_anomalies(snapshot, baselines)

    reliability = compute_device_reliability(recent)

    correlations = cross_correlate(recent)
    store.save_correlations(correlations)

    # ML contextual anomaly detection
    if HAS_SKLEARN and count_days_of_data(store) >= 14:
        feature_config = load_feature_config(store)
        fv = build_feature_vector(snapshot, feature_config)
        feature_names = get_feature_names(feature_config)
        features_list = [fv.get(name, 0) for name in feature_names]
        ctx_anomaly = detect_contextual_anomalies(
            features_list, str(config.paths.models_dir))
        if ctx_anomaly and ctx_anomaly.get("is_anomaly"):
            anomalies.append({
                "metric": "contextual",
                "description": (
                    f"ML anomaly detected (score: {ctx_anomaly['anomaly_score']}, "
                    f"severity: {ctx_anomaly['severity']})"
                ),
                "z_score": abs(ctx_anomaly["anomaly_score"]) * 5,
                "source": "isolation_forest",
            })

    print(f"Analysis complete: {len(anomalies)} anomalies, {len(correlations)} correlations")
    if anomalies:
        for a in anomalies:
            print(f"  ! {a['description']}")
    return anomalies, correlations, reliability


def cmd_predict():
    """Generate predictions for tomorrow with ML blending."""
    config, store = _init()

    from aria.engine.collectors.snapshot import build_empty_snapshot
    from aria.engine.collectors.ha_api import fetch_weather, parse_weather
    from aria.engine.features.time_features import build_time_features
    from aria.engine.features.vector_builder import build_feature_vector, get_feature_names
    from aria.engine.features.feature_config import load_feature_config
    from aria.engine.models.training import predict_with_ml, count_days_of_data
    from aria.engine.models.device_failure import (
        predict_device_failures, detect_contextual_anomalies,
    )
    from aria.engine.predictions.predictor import generate_predictions

    baselines = store.load_baselines()
    correlations = store.load_correlations()
    if isinstance(correlations, dict):
        correlations = correlations.get("correlations", [])

    weather = parse_weather(fetch_weather(config.weather))
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    # ML predictions (if models exist and enough data)
    ml_preds = None
    device_failures = None
    ctx_anomalies = None
    days = count_days_of_data(store)

    if HAS_SKLEARN and days >= 14:
        # Build a synthetic snapshot for tomorrow using baselines
        tomorrow_snap = build_empty_snapshot(tomorrow, config.holidays)
        dow = datetime.strptime(tomorrow, "%Y-%m-%d").strftime("%A")
        bl = baselines.get(dow, {})
        tomorrow_snap["power"]["total_watts"] = bl.get("power_watts", {}).get("mean", 0)
        tomorrow_snap["lights"]["on"] = bl.get("lights_on", {}).get("mean", 0)
        tomorrow_snap["occupancy"]["device_count_home"] = bl.get("devices_home", {}).get("mean", 0)
        if weather:
            tomorrow_snap["weather"] = weather
        tomorrow_snap["time_features"] = build_time_features(
            f"{tomorrow}T12:00:00", None, tomorrow)
        tomorrow_snap["media"] = {"total_active": 0}
        tomorrow_snap["motion"] = {"active_count": 0}
        tomorrow_snap["ev"] = {}

        ml_preds = predict_with_ml(tomorrow_snap, store=store,
                                    models_dir=str(config.paths.models_dir))
        if ml_preds:
            print(f"ML predictions available ({len(ml_preds)} metrics)")

        # Device failure predictions
        recent = store.load_recent_snapshots(90)
        device_failures = predict_device_failures(recent, str(config.paths.models_dir))
        if device_failures:
            print(f"Device failure warnings: {len(device_failures)}")
            for df in device_failures[:3]:
                print(f"  ! {df['entity_id']}: {df['failure_probability']:.0%} ({df['risk']})")

        # Contextual anomaly detection on today's snapshot
        today = datetime.now().strftime("%Y-%m-%d")
        today_snap = store.load_snapshot(today)
        if today_snap:
            feature_config = load_feature_config(store)
            fv = build_feature_vector(today_snap, feature_config)
            feature_names = get_feature_names(feature_config)
            features_list = [fv.get(name, 0) for name in feature_names]
            ctx_anomalies = detect_contextual_anomalies(
                features_list, str(config.paths.models_dir))
            if ctx_anomalies and ctx_anomalies.get("is_anomaly"):
                print(f"  ! Contextual anomaly detected "
                      f"(score: {ctx_anomalies['anomaly_score']}, "
                      f"severity: {ctx_anomalies['severity']})")

    predictions = generate_predictions(
        tomorrow, baselines, correlations, weather,
        ml_predictions=ml_preds, device_failures=device_failures,
        contextual_anomalies=ctx_anomalies, paths=config.paths,
    )
    store.save_predictions(predictions)
    print(f"Predictions for {tomorrow} ({predictions.get('prediction_method', 'statistical')}):")
    for k, v in predictions.items():
        if isinstance(v, dict) and "predicted" in v:
            print(f"  {k}: {v['predicted']} ({v['confidence']} confidence)")
    return predictions


def cmd_score():
    """Score yesterday's predictions against actual data."""
    config, store = _init()

    from aria.engine.predictions.scoring import score_all_predictions, accuracy_trend

    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    predictions = store.load_predictions()
    actual = store.load_snapshot(yesterday)
    if not actual:
        print(f"No snapshot for {yesterday}, cannot score.")
        return None
    if predictions.get("target_date") != yesterday:
        print(f"No predictions for {yesterday}.")
        return None
    result = score_all_predictions(predictions, actual)
    history = store.update_accuracy_history(result)
    trend = accuracy_trend(history)
    print(f"Accuracy for {yesterday}: {result['overall']}% (trend: {trend})")
    for metric, data in result.get("metrics", {}).items():
        print(f"  {metric}: predicted={data['predicted']}, actual={data['actual']}, accuracy={data['accuracy']}%")
    return result


def cmd_report(dry_run=False):
    """Generate full Ollama insight report."""
    config, store = _init()

    from aria.engine.collectors.snapshot import build_snapshot
    from aria.engine.analysis.baselines import compute_baselines
    from aria.engine.analysis.anomalies import detect_anomalies
    from aria.engine.analysis.reliability import compute_device_reliability
    from aria.engine.llm.reports import generate_insight_report

    today = datetime.now().strftime("%Y-%m-%d")
    snapshot = store.load_snapshot(today)
    if not snapshot:
        snapshot = build_snapshot(date_str=today, config=config, store=store)
        store.save_snapshot(snapshot)

    recent = store.load_recent_snapshots(30)
    baselines = compute_baselines(recent)
    anomalies = detect_anomalies(snapshot, baselines)
    reliability = compute_device_reliability(recent)
    correlations = store.load_correlations()
    if isinstance(correlations, dict):
        correlations = correlations.get("correlations", [])
    predictions = store.load_predictions()
    accuracy = store.load_accuracy_history()

    report = generate_insight_report(
        snapshot, anomalies, predictions, reliability,
        correlations, accuracy, config=config.ollama,
    )
    if dry_run:
        print(report)
    else:
        path = config.paths.insights_dir / f"{today}.json"
        with open(path, "w") as f:
            json.dump({"date": today, "report": report}, f, indent=2)
        print(f"Report saved: {path}")
    return report


def cmd_retrain():
    """Retrain all sklearn ML models."""
    config, store = _init()

    from aria.engine.models.training import train_all_models

    print("Retraining ML models...")
    results = train_all_models(days=90, config=config, store=store)
    if "error" in results:
        print(f"Training failed: {results['error']}")
    else:
        model_count = len([m for m in results.get("models", {}).values() if "error" not in m])
        print(f"Training complete: {model_count} models trained")
    return results


def cmd_meta_learn():
    """Run meta-learning analysis (weekly)."""
    config, store = _init()

    from aria.engine.llm.meta_learning import run_meta_learning

    return run_meta_learning(config=config, store=store)


def cmd_brief():
    """Print one-liner for telegram-brief integration."""
    config, store = _init()

    from aria.engine.collectors.snapshot import build_snapshot
    from aria.engine.analysis.anomalies import detect_anomalies
    from aria.engine.llm.reports import generate_brief_line

    today = datetime.now().strftime("%Y-%m-%d")
    snapshot = store.load_snapshot(today)
    if not snapshot:
        snapshot = build_snapshot(date_str=today, config=config, store=store)
    baselines = store.load_baselines()
    anomalies = detect_anomalies(snapshot, baselines)
    predictions = store.load_predictions()
    accuracy = store.load_accuracy_history()
    print(generate_brief_line(snapshot, anomalies, predictions, accuracy))


def cmd_check_drift():
    """Check for concept drift and conditionally trigger retraining."""
    config, store = _init()

    from aria.engine.analysis.drift import DriftDetector

    accuracy = store.load_accuracy_history()
    detector = DriftDetector()
    result = detector.check(accuracy)

    print(f"Drift check: {result['reason']}")
    if result.get("rolling_mae"):
        for metric, mae in result["rolling_mae"].items():
            current = result.get("current_mae", {}).get(metric, "?")
            threshold = result.get("threshold", {}).get(metric, "?")
            print(f"  {metric}: MAE={current} (rolling median={mae}, threshold={threshold})")

    if result["needs_retrain"]:
        print("Drift detected — triggering retrain...")
        cmd_retrain()
    else:
        if detector.should_skip_scheduled_retrain(accuracy):
            print("Models stable — scheduled retrain can be skipped.")

    return result


def cmd_entity_correlations():
    """Compute entity co-occurrence correlations from logbook data."""
    config, store = _init()

    from aria.engine.analysis.entity_correlations import (
        compute_co_occurrences, compute_hourly_patterns,
        summarize_entity_correlations,
    )

    entries = store.load_logbook()
    if not entries:
        print("No logbook data available.")
        return None

    co_occurrences = compute_co_occurrences(entries, window_minutes=15)
    hourly_patterns = compute_hourly_patterns(entries)
    summary = summarize_entity_correlations(co_occurrences, hourly_patterns)

    # Save enriched correlations
    store.save_entity_correlations(summary)

    print(f"Entity correlations: {summary['total_pairs_found']} pairs found, "
          f"{len(summary['automation_worthy_pairs'])} automation-worthy")
    for pair in summary["top_co_occurrences"][:5]:
        print(f"  {pair['entity_a']} ↔ {pair['entity_b']}: "
              f"{pair['count']}x ({pair['strength']}, "
              f"P={max(pair['conditional_prob_a_given_b'], pair['conditional_prob_b_given_a']):.0%})")

    return summary


def cmd_suggest_automations():
    """Generate HA automation YAML suggestions from learned patterns."""
    config, store = _init()

    from aria.engine.llm.automation_suggestions import generate_automation_suggestions

    result = generate_automation_suggestions(config=config, store=store)
    if not result or "error" in result:
        print(f"Automation suggestions: {result.get('error', 'no result')}")
        return result

    suggestions = result.get("suggestions", [])
    print(f"Generated {len(suggestions)} automation suggestions:")
    for i, s in enumerate(suggestions, 1):
        print(f"\n  [{i}] {s.get('description', 'No description')}")
        if s.get("yaml"):
            for line in s["yaml"].split("\n")[:5]:
                print(f"      {line}")

    return result


def cmd_train_prophet():
    """Train Prophet seasonal forecasters on daily snapshot time series."""
    config, store = _init()

    from aria.engine.models.prophet_forecaster import (
        train_prophet_models, predict_with_prophet, HAS_PROPHET,
    )

    if not HAS_PROPHET:
        print("Prophet not installed. Install with: python3 -m pip install prophet")
        return None

    # Load daily snapshots as (date_str, snapshot) tuples
    daily_dir = config.paths.daily_dir
    if not daily_dir.is_dir():
        print("No daily snapshots available.")
        return None

    import os
    snapshots = []
    for fname in sorted(os.listdir(daily_dir)):
        if not fname.endswith(".json"):
            continue
        date_str = fname.replace(".json", "")
        snap = store.load_snapshot(date_str)
        if snap:
            snapshots.append((date_str, snap))

    if len(snapshots) < 14:
        print(f"Insufficient data for Prophet ({len(snapshots)} days, need 14+)")
        return None

    print(f"Training Prophet on {len(snapshots)} daily snapshots...")
    results = train_prophet_models(snapshots, str(config.paths.models_dir))

    # Generate next-day forecasts
    forecasts = predict_with_prophet(str(config.paths.models_dir))
    if forecasts:
        print("Prophet forecasts for tomorrow:")
        for metric, value in forecasts.items():
            print(f"  {metric}: {value}")

    return results


def cmd_occupancy():
    """Estimate Bayesian occupancy from the current HA snapshot."""
    config, store = _init()

    from aria.engine.analysis.occupancy import (
        BayesianOccupancy, occupancy_to_features,
    )
    from aria.engine.collectors.snapshot import build_snapshot

    today = datetime.now().strftime("%Y-%m-%d")
    snapshot = store.load_snapshot(today)
    if not snapshot:
        snapshot = build_snapshot(date_str=today, config=config, store=store)

    estimator = BayesianOccupancy()
    result = estimator.estimate(snapshot)

    overall = result.get("overall", {})
    print(f"Occupancy: {overall.get('probability', 0):.0%} "
          f"(confidence: {overall.get('confidence', 'none')})")
    for signal in overall.get("signals", []):
        print(f"  {signal['type']}: {signal['value']:.0%} — {signal['detail']}")

    features = occupancy_to_features(result)
    print(f"\nFeature values: {json.dumps(features, indent=2)}")
    return result


def cmd_power_profiles():
    """Analyze per-outlet power consumption profiles."""
    config, store = _init()

    from aria.engine.analysis.power_profiles import ApplianceProfiler

    # Load daily snapshots
    import os
    daily_dir = config.paths.daily_dir
    if not daily_dir.is_dir():
        print("No daily snapshots available.")
        return None

    snapshots = []
    for fname in sorted(os.listdir(daily_dir)):
        if not fname.endswith(".json"):
            continue
        date_str = fname.replace(".json", "")
        snap = store.load_snapshot(date_str)
        if snap:
            snapshots.append((date_str, snap))

    # Also include intraday snapshots for finer resolution
    intraday = store.load_all_intraday_snapshots(days=30)
    for snap in intraday:
        ts = snap.get("timestamp", snap.get("metadata", {}).get("date", ""))
        if ts:
            snapshots.append((ts, snap))

    if len(snapshots) < 2:
        print("Insufficient power data.")
        return None

    profiler = ApplianceProfiler()
    result = profiler.analyze_snapshot_outlets(snapshots)

    print(f"Power analysis: {result['active_count']} active outlets, "
          f"{result['profiles_learned']} profiles learned")
    for name, info in result["outlets"].items():
        if info["is_active"]:
            health = info.get("health", {})
            health_str = f", health={health['score']}" if health.get("score") is not None else ""
            print(f"  {name}: avg={info['avg_watts']}W, "
                  f"max={info['max_watts']}W, "
                  f"cycles={info['cycles_detected']}{health_str}")
            for alert in health.get("alerts", []):
                print(f"    ! {alert}")

    # Save results
    output = config.paths.insights_dir / "power-profiles.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Saved to {output}")

    return result


def cmd_train_sequences():
    """Train Markov chain sequence model from logbook data."""
    config, store = _init()

    from aria.engine.analysis.sequence_anomalies import MarkovChainDetector

    entries = store.load_logbook()
    if not entries:
        print("No logbook data available.")
        return None

    detector = MarkovChainDetector(window_seconds=300, min_transitions=50)
    result = detector.train(entries)

    store.save_sequence_model(detector.to_dict())
    print(f"Sequence model: {result['status']} "
          f"({result['transitions']} transitions, "
          f"{result['unique_entities']} entities)")
    if result["threshold"] is not None:
        print(f"  Anomaly threshold: {result['threshold']:.4f}")
    return result


def cmd_sequence_anomalies():
    """Detect anomalous event sequences using trained Markov chain."""
    config, store = _init()

    from aria.engine.analysis.sequence_anomalies import (
        MarkovChainDetector,
        summarize_sequence_anomalies,
    )

    model_data = store.load_sequence_model()
    if not model_data:
        print("No sequence model found. Run --train-sequences first.")
        return None

    detector = MarkovChainDetector.from_dict(model_data)
    if detector.threshold is None:
        print("Sequence model not fully trained (no threshold). Need more data.")
        return None

    entries = store.load_logbook()
    if not entries:
        print("No logbook data available.")
        return None

    anomalies = detector.detect(entries)
    summary = summarize_sequence_anomalies(
        anomalies, total_windows_checked=max(1, len(entries) // 5))
    store.save_sequence_anomalies(summary)

    print(f"Sequence anomalies: {summary['anomalies_found']} found "
          f"({summary['total_windows_checked']} windows checked)")
    for a in anomalies[:5]:
        print(f"  {a['time_start']} -> {a['time_end']}: "
              f"score={a['score']} ({a['severity']})")
        print(f"    entities: {', '.join(a['entities'][:5])}")
    return summary


def cmd_full(dry_run=False):
    """Run the full daily pipeline: snapshot, score, analyze, predict, report."""
    cmd_snapshot()
    cmd_score()
    cmd_analyze()
    cmd_predict()
    cmd_report(dry_run=dry_run)


# ── Main Dispatch ─────────────────────────────────────────────────────


def main():
    args = sys.argv[1:]
    dry_run = "--dry-run" in args

    if "--snapshot-intraday" in args:
        cmd_snapshot_intraday()
    elif "--snapshot" in args:
        cmd_snapshot()
    elif "--analyze" in args:
        cmd_analyze()
    elif "--predict" in args:
        cmd_predict()
    elif "--score" in args:
        cmd_score()
    elif "--check-drift" in args:
        cmd_check_drift()
    elif "--entity-correlations" in args:
        cmd_entity_correlations()
    elif "--suggest-automations" in args:
        cmd_suggest_automations()
    elif "--train-prophet" in args:
        cmd_train_prophet()
    elif "--occupancy" in args:
        cmd_occupancy()
    elif "--power-profiles" in args:
        cmd_power_profiles()
    elif "--train-sequences" in args:
        cmd_train_sequences()
    elif "--sequence-anomalies" in args:
        cmd_sequence_anomalies()
    elif "--retrain" in args:
        cmd_retrain()
    elif "--meta-learn" in args:
        cmd_meta_learn()
    elif "--report" in args:
        cmd_report(dry_run=dry_run)
    elif "--brief" in args:
        cmd_brief()
    elif "--full" in args:
        cmd_full(dry_run=dry_run)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
