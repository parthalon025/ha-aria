"""Meta-learning — LLM-driven feature config optimization.

Analyzes prediction accuracy, queries LLM for improvement suggestions,
validates each suggestion by retraining, and auto-applies guardrailed changes.
"""

import copy
import json
import re
import shutil
import tempfile
from datetime import datetime

from aria.engine.config import AppConfig, OllamaConfig
from aria.engine.features.feature_config import DEFAULT_FEATURE_CONFIG
from aria.engine.llm.client import ollama_chat
from aria.engine.storage.data_store import DataStore

# --- Constants ---

MAX_META_CHANGES_PER_WEEK = 3

# --- Safety Guards (research-backed) ---
# Khritankov et al. (2024) "Hidden Feedback Loop" + Shumailov et al. (Nature 2024)
# These prevent feedback loop oscillation in the meta-learner.

# Change budget: max config changes per run (prevents over-modification)
MAX_CHANGES_PER_RUN = 3

# Hysteresis thresholds: require improvement > ACCEPT_THRESHOLD before accepting,
# and degradation > REVERT_THRESHOLD before reverting. The asymmetry prevents
# oscillation where small noise causes accept/revert/accept cycles.
ACCEPT_IMPROVEMENT_PCT = 2.0  # Must improve accuracy by >2% to accept
REVERT_DEGRADATION_PCT = 5.0  # Must degrade by >5% to trigger revert

META_LEARNING_PROMPT = """You are a data scientist analyzing a home automation prediction system.
Your job is to find patterns in prediction errors and suggest improvements.

## Current System Performance (last 7 days)
{accuracy_data}

## Feature Importance (from current sklearn models)
{feature_importance}

## Current Feature Configuration
{feature_config}

## Available Data Fields (from daily snapshots)
{available_fields}

## Known Correlations
{correlations}

## Previous Suggestions and Outcomes
{previous_suggestions}

## Task
Analyze the prediction accuracy data and suggest specific improvements.

For each suggestion, provide:
1. "action": "enable_feature" | "disable_feature" | "add_interaction" | "adjust_hyperparameter"
2. "target": the specific feature or parameter name
3. "reason": evidence from the accuracy data (cite specific numbers)
4. "expected_impact": which metric should improve and by roughly how much
5. "confidence": "high" | "medium" | "low"

Rules:
- Maximum 3 suggestions per analysis
- Only suggest changes with clear evidence from the data
- Do NOT suggest changes to safety-critical features
- Prefer enabling existing disabled features over creating new ones
- If accuracy is already >85%, focus on the weakest metric only

Output as a JSON array of suggestion objects. Example:
[{{"action": "enable_feature", "target": "is_weekend_x_temp",
"reason": "weekend power predictions off by 15%",
"expected_impact": "power_watts MAE -5%", "confidence": "medium"}}]
"""


# --- Suggestion Parsing ---


def parse_suggestions(llm_response):
    """Parse JSON suggestion array from LLM response.

    Handles deepseek-r1 <think>...</think> blocks by stripping them first.
    Returns list of suggestion dicts, or empty list on parse failure.
    """
    text = llm_response

    # Strip <think>...</think> blocks (deepseek-r1 reasoning)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Find JSON array in response
    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if not match:
        return []

    try:
        suggestions = json.loads(match.group())
        if not isinstance(suggestions, list):
            return []
        # Validate required fields
        valid = []
        for s in suggestions[:MAX_META_CHANGES_PER_WEEK]:
            if isinstance(s, dict) and "action" in s and "target" in s:
                valid.append(s)
        return valid
    except (json.JSONDecodeError, ValueError):
        return []


# --- Config Application ---


def apply_suggestion_to_config(suggestion, config):
    """Apply a single suggestion to a feature config (in-place).

    Returns True if applied, False if not applicable.
    """
    action = suggestion.get("action")
    target = suggestion.get("target")

    if action == "enable_feature":
        # Check interaction_features (only toggleable features)
        if target in config.get("interaction_features", {}):
            config["interaction_features"][target] = True
            return True
        # Check lag_features
        if target in config.get("lag_features", {}):
            config["lag_features"][target] = True
            return True
    elif action == "disable_feature":
        if target in config.get("interaction_features", {}):
            config["interaction_features"][target] = False
            return True
        if target in config.get("lag_features", {}):
            config["lag_features"][target] = False
            return True
    elif action == "add_interaction":
        if "interaction_features" in config:
            config["interaction_features"][target] = True
            return True

    return False


# --- Validation ---


def _can_validate(snapshots):
    """Check if validation prerequisites are met."""
    try:
        from sklearn.ensemble import GradientBoostingRegressor  # noqa: F401
    except ImportError:
        return False
    return len(snapshots) >= 14


def validate_suggestion(suggestion, snapshots, config):
    """Validate a suggestion by retraining with modified config and comparing accuracy.

    Returns (improvement_pct, modified_config) or (None, None) on failure.
    """
    fail = (None, None)
    if not _can_validate(snapshots):
        return fail

    # Deferred imports — these modules may still be migrating
    from aria.engine.features.vector_builder import build_training_data
    from aria.engine.models.training import train_continuous_model

    modified_config = copy.deepcopy(config)
    if not apply_suggestion_to_config(suggestion, modified_config):
        return fail

    # Build training data with both configs
    orig_names, orig_X, orig_targets = build_training_data(snapshots, config)
    mod_names, mod_X, mod_targets = build_training_data(snapshots, modified_config)

    metric = "power_watts"
    if len(orig_X) < 14 or len(mod_X) < 14 or metric not in orig_targets or metric not in mod_targets:
        return fail

    tmpdir_orig = tempfile.mkdtemp()
    tmpdir_mod = tempfile.mkdtemp()
    try:
        orig_result = train_continuous_model(metric, orig_names, orig_X, orig_targets[metric], tmpdir_orig)
        mod_result = train_continuous_model(metric, mod_names, mod_X, mod_targets[metric], tmpdir_mod)

        if "error" in orig_result or "error" in mod_result or orig_result["mae"] == 0:
            return fail

        improvement_pct = ((orig_result["mae"] - mod_result["mae"]) / orig_result["mae"]) * 100
        return round(improvement_pct, 2), modified_config
    finally:
        shutil.rmtree(tmpdir_orig, ignore_errors=True)
        shutil.rmtree(tmpdir_mod, ignore_errors=True)


# --- Safety Guards ---


def check_revert_needed(accuracy_history: dict, applied_history: dict) -> dict:
    """Check if recent meta-learner changes should be reverted.

    Uses hysteresis: only reverts when degradation exceeds REVERT_DEGRADATION_PCT.
    This prevents oscillation from noisy accuracy measurements.

    Args:
        accuracy_history: Dict with "scores" list (date, overall, metrics).
        applied_history: Dict with "applied" list of past changes.

    Returns:
        Dict with revert_needed (bool), reason, and degradation_pct.
    """
    applied = applied_history.get("applied", [])
    if not applied:
        return {"revert_needed": False, "reason": "no changes to revert"}

    scores = accuracy_history.get("scores", [])
    if len(scores) < 7:
        return {"revert_needed": False, "reason": "insufficient data for revert check"}

    # Compare accuracy before vs after the most recent change
    last_change = applied[-1]
    change_date = last_change.get("date", "")

    before_scores = [s["overall"] for s in scores if s.get("date", "") < change_date]
    after_scores = [s["overall"] for s in scores if s.get("date", "") >= change_date]

    if len(before_scores) < 3 or len(after_scores) < 3:
        return {"revert_needed": False, "reason": "not enough data around change point"}

    import statistics

    before_mean = statistics.mean(before_scores[-5:])
    after_mean = statistics.mean(after_scores[-5:])
    degradation = before_mean - after_mean

    if degradation > REVERT_DEGRADATION_PCT:
        return {
            "revert_needed": True,
            "reason": f"accuracy degraded {degradation:.1f}% since last change (threshold: {REVERT_DEGRADATION_PCT}%)",
            "degradation_pct": round(degradation, 2),
            "before_accuracy": round(before_mean, 2),
            "after_accuracy": round(after_mean, 2),
        }

    return {
        "revert_needed": False,
        "reason": f"degradation {degradation:.1f}% within tolerance (threshold: {REVERT_DEGRADATION_PCT}%)",
        "degradation_pct": round(degradation, 2),
    }


# --- Main Pipeline ---


def _gather_meta_context(config, store):
    """Gather all context needed for meta-learning LLM prompt."""
    accuracy_history = store.load_accuracy_history()
    recent_scores = accuracy_history.get("scores", [])[-7:]

    feature_importance = {}
    training_log_path = config.paths.models_dir / "training_log.json"
    if training_log_path.is_file():
        with open(training_log_path) as f:
            training_log = json.load(f)
        for model_name, model_data in training_log.get("models", {}).items():
            if isinstance(model_data, dict) and "feature_importance" in model_data:
                fi = model_data["feature_importance"]
                top = sorted(fi.items(), key=lambda x: -x[1])[:10]
                feature_importance[model_name] = dict(top)

    feature_config = store.load_feature_config()
    if feature_config is None:
        feature_config = DEFAULT_FEATURE_CONFIG.copy()

    correlations = store.load_correlations()
    if isinstance(correlations, dict):
        correlations = correlations.get("correlations", [])

    applied_history = store.load_applied_suggestions()
    available_fields = list(DEFAULT_FEATURE_CONFIG.get("interaction_features", {}).keys())

    return (
        accuracy_history, recent_scores, feature_importance,
        feature_config, correlations, applied_history, available_fields,
    )


def _handle_revert(store, config, revert_check, recent_scores, week_str):
    """Handle config revert when degradation detected."""
    print(f"  SAFETY: {revert_check['reason']}")
    print("  Reverting to default config and skipping new suggestions.")
    default_config = DEFAULT_FEATURE_CONFIG.copy()
    default_config["last_modified"] = datetime.now().isoformat()
    store.save_feature_config(default_config)

    weekly_report = {
        "week": week_str,
        "generated_at": datetime.now().isoformat(),
        "suggestions": [],
        "applied_count": 0,
        "reverted": True,
        "revert_reason": revert_check["reason"],
        "accuracy_context": recent_scores,
    }
    _save_weekly_report(config, week_str, weekly_report)
    print("Meta-learning complete: reverted to default config")
    return weekly_report


def _apply_suggestions(suggestions, snapshots, feature_config, store):
    """Validate and apply suggestions, returning results and applied count."""
    results = []
    applied_count = 0
    for suggestion in suggestions:
        if applied_count >= MAX_CHANGES_PER_RUN:
            results.append({"suggestion": suggestion, "applied": False, "reason": "per-run change budget reached"})
            continue

        improvement, modified_config = validate_suggestion(suggestion, snapshots, feature_config)
        if improvement is None:
            results.append({"suggestion": suggestion, "applied": False, "reason": "validation failed"})
            continue

        if improvement >= ACCEPT_IMPROVEMENT_PCT:
            modified_config["last_modified"] = datetime.now().isoformat()
            store.save_feature_config(modified_config)
            feature_config = modified_config
            applied_count += 1
            results.append({"suggestion": suggestion, "applied": True, "improvement": improvement})
            print(f"  Applied: {suggestion.get('action')} {suggestion.get('target')} (+{improvement:.1f}%)")
        else:
            results.append({
                "suggestion": suggestion,
                "applied": False,
                "reason": f"improvement {improvement:.1f}% < {ACCEPT_IMPROVEMENT_PCT}% threshold",
                "accuracy_delta": improvement,
            })
            print(f"  Rejected: {suggestion.get('target')} ({improvement:+.1f}%, need >={ACCEPT_IMPROVEMENT_PCT}%)")

    return results, applied_count


def _save_weekly_report(config, week_str, report):
    """Save weekly meta-learning report to disk."""
    weekly_dir = config.paths.meta_dir / "weekly"
    weekly_dir.mkdir(parents=True, exist_ok=True)
    with open(weekly_dir / f"{week_str}.json", "w") as f:
        json.dump(report, f, indent=2)


def _build_meta_prompt(  # noqa: PLR0913 — prompt requires all context fields
    recent_scores, feature_importance, feature_config,
    available_fields, correlations, applied_history,
):
    """Build the LLM prompt for meta-learning analysis."""
    return META_LEARNING_PROMPT.format(
        accuracy_data=json.dumps(recent_scores, indent=2) if recent_scores else "No accuracy data yet",
        feature_importance=json.dumps(feature_importance, indent=2) if feature_importance else "No models trained yet",
        feature_config=json.dumps(feature_config, indent=2),
        available_fields=json.dumps(available_fields),
        correlations=json.dumps(correlations[:10], indent=2) if correlations else "No correlations yet",
        previous_suggestions=json.dumps(applied_history.get("applied", [])[-5:], indent=2),
    )


def _finalize_meta_learning(  # noqa: PLR0913 — aggregates all meta-learning outputs
    results, applied_count, applied_history, store,
    config, week_str, recent_scores, suggestions,
):
    """Save report, update history, and retrain if needed."""
    weekly_report = {
        "week": week_str,
        "generated_at": datetime.now().isoformat(),
        "suggestions": results,
        "applied_count": applied_count,
        "accuracy_context": recent_scores,
    }
    _save_weekly_report(config, week_str, weekly_report)

    for r in results:
        if r.get("applied"):
            applied_history["applied"].append({
                "date": datetime.now().isoformat(),
                "suggestion": r["suggestion"],
                "improvement": r["improvement"],
            })
    applied_history["total_applied"] = len(applied_history["applied"])
    store.save_applied_suggestions(applied_history)

    if applied_count > 0:
        from aria.engine.models.training import train_all_models
        print(f"Retraining models with {applied_count} config changes...")
        train_all_models(config=config, store=store, days=90)

    print(f"Meta-learning complete: {applied_count}/{len(suggestions)} suggestions applied")
    return weekly_report


def run_meta_learning(config: AppConfig = None, store: DataStore = None):
    """Run weekly meta-learning analysis and auto-apply guardrailed suggestions."""
    if config is None:
        config = AppConfig.from_env()
    if store is None:
        store = DataStore(config.paths)

    try:
        from sklearn.ensemble import GradientBoostingRegressor  # noqa: F401
    except ImportError:
        print("sklearn not installed, skipping meta-learning")
        return {"error": "sklearn not installed"}

    from aria.engine.models.training import count_days_of_data

    days = count_days_of_data(config.paths)
    if days < 14:
        print(f"Insufficient data for meta-learning ({days} days, need 14+)")
        return {"error": f"insufficient data ({days} days)"}

    (accuracy_history, recent_scores, feature_importance, feature_config,
     correlations, applied_history, available_fields) = _gather_meta_context(config, store)

    prompt = _build_meta_prompt(
        recent_scores, feature_importance, feature_config,
        available_fields, correlations, applied_history,
    )

    print("Querying LLM for meta-learning analysis...")
    ollama_config = OllamaConfig(url=config.ollama.url, model=config.ollama.model, timeout=120)
    response = ollama_chat(prompt, config=ollama_config)
    if not response:
        print("LLM returned empty response")
        return {"error": "empty LLM response"}

    suggestions = parse_suggestions(response)
    print(f"Parsed {len(suggestions)} suggestions from LLM")

    week_str = datetime.now().strftime("%Y-W%W")

    revert_check = check_revert_needed(accuracy_history, applied_history)
    if revert_check.get("revert_needed"):
        return _handle_revert(store, config, revert_check, recent_scores, week_str)

    snapshots = store.load_all_intraday_snapshots(90)
    if len(snapshots) < 14:
        snapshots = store.load_recent_snapshots(90)

    results, applied_count = _apply_suggestions(suggestions, snapshots, feature_config, store)

    return _finalize_meta_learning(
        results, applied_count, applied_history, store,
        config, week_str, recent_scores, suggestions,
    )
