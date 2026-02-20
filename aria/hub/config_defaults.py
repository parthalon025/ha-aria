"""Config defaults registry — single source of truth for all tunable parameters.

Each parameter is defined with its key, default value, type, constraints, and
UI metadata. On startup, seed_config_defaults() inserts any missing keys using
INSERT OR IGNORE, preserving user overrides.
"""

from typing import Any

CONFIG_DEFAULTS: list[dict[str, Any]] = [
    # ── Activity Monitor ──────────────────────────────────────────────
    {
        "key": "activity.daily_snapshot_cap",
        "default_value": "20",
        "value_type": "number",
        "label": "Daily Snapshot Cap",
        "description": "Maximum adaptive snapshots triggered per day.",
        "category": "Activity Monitor",
        "min_value": 5,
        "max_value": 100,
        "step": 1,
    },
    {
        "key": "activity.snapshot_cooldown_s",
        "default_value": "1800",
        "value_type": "number",
        "label": "Snapshot Cooldown (s)",
        "description": "Minimum seconds between adaptive snapshots.",
        "category": "Activity Monitor",
        "min_value": 300,
        "max_value": 7200,
        "step": 60,
    },
    {
        "key": "activity.flush_interval_s",
        "default_value": "900",
        "value_type": "number",
        "label": "Flush Interval (s)",
        "description": "How often buffered events are flushed to cache windows.",
        "category": "Activity Monitor",
        "min_value": 60,
        "max_value": 3600,
        "step": 60,
    },
    {
        "key": "activity.max_window_age_h",
        "default_value": "24",
        "value_type": "number",
        "label": "Max Window Age (h)",
        "description": "Rolling window retention in hours.",
        "category": "Activity Monitor",
        "min_value": 6,
        "max_value": 168,
        "step": 1,
    },
    # ── Event Store ────────────────────────────────────────────────────
    {
        "key": "events.retention_days",
        "default_value": "90",
        "value_type": "number",
        "label": "Event Retention (Days)",
        "description": "How many days of raw state_changed events to keep in the event store.",
        "category": "Event Store",
        "min_value": 7,
        "max_value": 365,
        "step": 1,
    },
    # ── Feature Engineering ───────────────────────────────────────────
    {
        "key": "features.decay_half_life_days",
        "default_value": "7",
        "value_type": "number",
        "label": "Decay Half-Life (days)",
        "description": "Exponential decay half-life for training sample recency weighting.",
        "category": "Feature Engineering",
        "min_value": 1,
        "max_value": 30,
        "step": 1,
    },
    {
        "key": "features.weekday_alignment_bonus",
        "default_value": "1.5",
        "value_type": "number",
        "label": "Weekday Alignment Bonus",
        "description": "Multiplier for training samples from the same day of week.",
        "category": "Feature Engineering",
        "min_value": 1.0,
        "max_value": 3.0,
        "step": 0.1,
    },
    # ── Shadow Engine ─────────────────────────────────────────────────
    {
        "key": "shadow.min_confidence",
        "default_value": "0.3",
        "value_type": "number",
        "label": "Min Confidence",
        "description": "Predictions below this confidence are not stored.",
        "category": "Shadow Engine",
        "min_value": 0.05,
        "max_value": 0.9,
        "step": 0.05,
    },
    {
        "key": "shadow.default_window_seconds",
        "default_value": "600",
        "value_type": "number",
        "label": "Default Window (s)",
        "description": "Evaluation window for predictions in seconds.",
        "category": "Shadow Engine",
        "min_value": 60,
        "max_value": 3600,
        "step": 30,
    },
    {
        "key": "shadow.resolution_interval_s",
        "default_value": "60",
        "value_type": "number",
        "label": "Resolution Interval (s)",
        "description": "How often expired prediction windows are resolved.",
        "category": "Shadow Engine",
        "min_value": 10,
        "max_value": 300,
        "step": 10,
    },
    {
        "key": "shadow.prediction_cooldown_s",
        "default_value": "30",
        "value_type": "number",
        "label": "Prediction Cooldown (s)",
        "description": "Minimum seconds between prediction attempts (debounce).",
        "category": "Shadow Engine",
        "min_value": 5,
        "max_value": 300,
        "step": 5,
    },
    {
        "key": "shadow.explore_strategy",
        "default_value": "thompson",
        "value_type": "select",
        "label": "Explore Strategy",
        "description": "Exploration strategy for shadow predictions: Thompson Sampling or epsilon-greedy.",
        "category": "Shadow Engine",
        "options": "thompson,epsilon",
    },
    {
        "key": "shadow.thompson_discount_factor",
        "default_value": "0.95",
        "value_type": "number",
        "label": "Thompson Discount Factor",
        "description": "f-dsw discount factor for Thompson Sampling posteriors (lower = faster adaptation).",
        "category": "Shadow Engine",
        "min_value": 0.8,
        "max_value": 1.0,
        "step": 0.01,
    },
    {
        "key": "shadow.thompson_window_size",
        "default_value": "100",
        "value_type": "number",
        "label": "Thompson Window Size",
        "description": "Maximum effective history length for Thompson Sampling buckets.",
        "category": "Shadow Engine",
        "min_value": 20,
        "max_value": 500,
        "step": 10,
    },
    # ── Data Quality ──────────────────────────────────────────────────
    {
        "key": "curation.auto_exclude_domains",
        "default_value": (
            "update,tts,stt,scene,button,number,select,input_boolean,input_number,"
            "input_select,input_text,input_datetime,counter,script,zone,sun,weather,"
            "conversation,event,automation,camera,image,remote"
        ),
        "value_type": "string",
        "label": "Auto-Exclude Domains",
        "description": "Comma-separated domains automatically excluded from curation.",
        "category": "Data Quality",
    },
    {
        "key": "curation.noise_event_threshold",
        "default_value": "1000",
        "value_type": "number",
        "label": "Noise Event Threshold",
        "description": "Daily event count above which an entity is considered noise (if low state variety).",
        "category": "Data Quality",
        "min_value": 100,
        "max_value": 10000,
        "step": 100,
    },
    {
        "key": "curation.stale_days_threshold",
        "default_value": "30",
        "value_type": "number",
        "label": "Stale Days Threshold",
        "description": "Entities with no state changes in this many days are auto-excluded.",
        "category": "Data Quality",
        "min_value": 7,
        "max_value": 90,
        "step": 1,
    },
    {
        "key": "curation.unavailable_grace_hours",
        "default_value": "0",
        "value_type": "number",
        "label": "Unavailable Grace Hours",
        "description": "Entities unavailable longer than this are auto-excluded. 0 = disabled.",
        "category": "Data Quality",
        "min_value": 0,
        "max_value": 168,
        "step": 1,
    },
    {
        "key": "curation.vehicle_patterns",
        "default_value": "tesla,luda,tessy,vehicle,car_",
        "value_type": "string",
        "label": "Vehicle Patterns",
        "description": "Comma-separated patterns to match vehicle-related entity names.",
        "category": "Data Quality",
    },
    # ── Anomaly Detection ──────────────────────────────────────────────
    {
        "key": "anomaly.use_autoencoder",
        "default_value": "true",
        "value_type": "boolean",
        "label": "Use Autoencoder",
        "description": "Enable hybrid autoencoder + IsolationForest anomaly detection.",
        "category": "Anomaly Detection",
    },
    {
        "key": "anomaly.ae_hidden_layers",
        "default_value": "24,12,24",
        "value_type": "string",
        "label": "Autoencoder Hidden Layers",
        "description": "Comma-separated hidden layer sizes for the autoencoder (e.g. 24,12,24).",
        "category": "Anomaly Detection",
    },
    {
        "key": "anomaly.ae_max_iter",
        "default_value": "200",
        "value_type": "number",
        "label": "Autoencoder Max Iterations",
        "description": "Maximum training iterations for the autoencoder.",
        "category": "Anomaly Detection",
        "min_value": 50,
        "max_value": 1000,
        "step": 50,
    },
    {
        "key": "anomaly.contamination",
        "default_value": "0.05",
        "value_type": "number",
        "label": "Contamination Factor",
        "description": "Expected proportion of anomalies in the dataset.",
        "category": "Anomaly Detection",
        "min_value": 0.01,
        "max_value": 0.20,
        "step": 0.01,
    },
    # ── Forecaster ─────────────────────────────────────────────────────
    {
        "key": "forecaster.backend",
        "default_value": "auto",
        "value_type": "select",
        "label": "Forecaster Backend",
        "description": "Which forecasting backend to use. 'auto' prefers NeuralProphet, falls back to Prophet.",
        "category": "Forecaster",
        "options": "auto,neuralprophet,prophet",
    },
    {
        "key": "forecaster.epochs",
        "default_value": "100",
        "value_type": "number",
        "label": "Training Epochs",
        "description": "Number of training epochs for NeuralProphet (ignored by Prophet).",
        "category": "Forecaster",
        "min_value": 10,
        "max_value": 500,
        "step": 10,
    },
    {
        "key": "forecaster.learning_rate",
        "default_value": "0.1",
        "value_type": "number",
        "label": "Learning Rate",
        "description": "Optimizer learning rate for NeuralProphet (ignored by Prophet).",
        "category": "Forecaster",
        "min_value": 0.001,
        "max_value": 1.0,
        "step": 0.01,
    },
    {
        "key": "forecaster.ar_order",
        "default_value": "7",
        "value_type": "number",
        "label": "AR Order",
        "description": "Number of autoregression lags for NeuralProphet (7 = one week of history).",
        "category": "Forecaster",
        "min_value": 1,
        "max_value": 30,
        "step": 1,
    },
    # ── Drift Detection ─────────────────────────────────────────────
    {
        "key": "drift.adwin_delta",
        "default_value": "0.002",
        "value_type": "number",
        "label": "ADWIN Delta",
        "description": "ADWIN confidence parameter. Lower values reduce false positives but slow detection.",
        "category": "Drift Detection",
        "min_value": 0.0001,
        "max_value": 0.1,
        "step": 0.001,
    },
    {
        "key": "drift.require_confirmation",
        "default_value": "false",
        "value_type": "boolean",
        "label": "Require Confirmation",
        "description": "When true, drift must be confirmed by multiple detectors before triggering retrain.",
        "category": "Drift Detection",
    },
    {
        "key": "drift.page_hinkley_enabled",
        "default_value": "true",
        "value_type": "boolean",
        "label": "Page-Hinkley Enabled",
        "description": "Whether to run Page-Hinkley drift detection alongside threshold checks.",
        "category": "Drift Detection",
    },
    # ── Reference Model ────────────────────────────────────────────────
    {
        "key": "reference_model.enabled",
        "default_value": "true",
        "value_type": "boolean",
        "label": "Reference Model Enabled",
        "description": "Enable clean reference model for distinguishing meta-learner errors from behavioral drift.",
        "category": "Reference Model",
    },
    {
        "key": "reference_model.comparison_window_days",
        "default_value": "7",
        "value_type": "number",
        "label": "Comparison Window (days)",
        "description": "Number of days of accuracy history to compare between primary and reference models.",
        "category": "Reference Model",
        "min_value": 3,
        "max_value": 30,
        "step": 1,
    },
    {
        "key": "reference_model.alert_threshold_pct",
        "default_value": "5.0",
        "value_type": "number",
        "label": "Alert Threshold (%)",
        "description": "Minimum accuracy divergence percentage to trigger a meta-learner error alert.",
        "category": "Reference Model",
        "min_value": 1.0,
        "max_value": 20.0,
        "step": 0.5,
    },
    # ── Feature Selection ──────────────────────────────────────────────
    {
        "key": "feature_selection.enabled",
        "default_value": "true",
        "value_type": "boolean",
        "label": "Feature Selection Enabled",
        "description": "Enable automatic feature selection before model training.",
        "category": "Feature Selection",
    },
    {
        "key": "feature_selection.method",
        "default_value": "mrmr",
        "value_type": "select",
        "label": "Selection Method",
        "description": "Feature selection algorithm: mRMR, importance-based, or none.",
        "category": "Feature Selection",
        "options": "mrmr,importance,none",
    },
    {
        "key": "feature_selection.max_features",
        "default_value": "30",
        "value_type": "number",
        "label": "Max Features",
        "description": "Maximum number of features to retain after selection.",
        "category": "Feature Selection",
        "min_value": 10,
        "max_value": 48,
        "step": 1,
    },
    {
        "key": "feature_selection.recompute_interval_days",
        "default_value": "7",
        "value_type": "number",
        "label": "Recompute Interval (days)",
        "description": "How often to recompute the selected feature set.",
        "category": "Feature Selection",
        "min_value": 1,
        "max_value": 30,
        "step": 1,
    },
    # ── Narration ─────────────────────────────────────────────────────
    {
        "key": "narration.use_shap",
        "default_value": "true",
        "value_type": "boolean",
        "label": "Use SHAP",
        "description": "Ground LLM narration in SHAP feature attributions for faithful explanations.",
        "category": "Narration",
    },
    {
        "key": "narration.top_features",
        "default_value": "5",
        "value_type": "number",
        "label": "Top Features",
        "description": "Number of top SHAP contributors to include in narration.",
        "category": "Narration",
        "min_value": 3,
        "max_value": 10,
        "step": 1,
    },
    {
        "key": "narration.grounding_mode",
        "default_value": "shap",
        "value_type": "select",
        "label": "Grounding Mode",
        "description": "Source for feature attribution: SHAP values, built-in importance, or none.",
        "category": "Narration",
        "options": "shap,importance,none",
    },
    # ── Incremental Training ──────────────────────────────────────────
    {
        "key": "incremental.enabled",
        "default_value": "true",
        "value_type": "boolean",
        "label": "Incremental Training Enabled",
        "description": "Enable eGBDT incremental LightGBM adaptation on drift detection.",
        "category": "Incremental Training",
    },
    {
        "key": "incremental.boost_rounds",
        "default_value": "20",
        "value_type": "number",
        "label": "Boost Rounds",
        "description": "Number of boosting rounds to add during incremental training.",
        "category": "Incremental Training",
        "min_value": 5,
        "max_value": 100,
        "step": 5,
    },
    {
        "key": "incremental.max_total_trees",
        "default_value": "500",
        "value_type": "number",
        "label": "Max Total Trees",
        "description": "Maximum tree count before forcing a full retrain.",
        "category": "Incremental Training",
        "min_value": 100,
        "max_value": 2000,
        "step": 100,
    },
    {
        "key": "incremental.data_window_days",
        "default_value": "14",
        "value_type": "number",
        "label": "Data Window (days)",
        "description": "Number of recent days of data used for incremental training.",
        "category": "Incremental Training",
        "min_value": 7,
        "max_value": 30,
        "step": 1,
    },
    # ── Presence Tracking ─────────────────────────────────────────────
    {
        "key": "presence.mqtt_host",
        "default_value": "",
        "value_type": "string",
        "label": "MQTT Host",
        "description": "Hostname or IP of the MQTT broker (Mosquitto on HA).",
        "category": "Presence Tracking",
    },
    {
        "key": "presence.mqtt_port",
        "default_value": "1883",
        "value_type": "number",
        "label": "MQTT Port",
        "description": "Port of the MQTT broker.",
        "category": "Presence Tracking",
        "min_value": 1,
        "max_value": 65535,
        "step": 1,
    },
    {
        "key": "presence.mqtt_user",
        "default_value": "",
        "value_type": "string",
        "label": "MQTT User",
        "description": "Username for MQTT broker authentication.",
        "category": "Presence Tracking",
    },
    {
        "key": "presence.mqtt_password",
        "default_value": "",
        "value_type": "string",
        "label": "MQTT Password",
        "description": "Password for MQTT broker authentication.",
        "category": "Presence Tracking",
    },
    {
        "key": "presence.camera_rooms",
        "default_value": "",
        "value_type": "string",
        "label": "Camera-to-Room Mapping",
        "description": (
            "Comma-separated camera:room pairs (e.g. front_yard:front_yard,bedroom:bedroom). Leave empty for defaults."
        ),
        "category": "Presence Tracking",
    },
    # ── Discovery Lifecycle ────────────────────────────────────────────
    {
        "key": "discovery.stale_ttl_hours",
        "default_value": "72",
        "value_type": "number",
        "label": "Entity Stale TTL (hours)",
        "description": (
            "Hours after an entity disappears from HA discovery before it is archived. "
            "While stale, entities remain usable. After archival, they are excluded from "
            "active consumers but preserved for reference. Set to 0 to archive immediately."
        ),
        "category": "Discovery",
        "min_value": 0,
        "max_value": 720,
        "step": 1,
    },
    # ── Correction Propagation ─────────────────────────────────────────
    {
        "key": "propagation.enabled",
        "default_value": "true",
        "value_type": "boolean",
        "label": "Propagation Enabled",
        "description": "Enable adaptive correction propagation (Slivkins zooming + replay).",
        "category": "Correction Propagation",
    },
    {
        "key": "propagation.base_radius_hours",
        "default_value": "1.0",
        "value_type": "number",
        "label": "Base Radius (hours)",
        "description": "Base temporal radius for correction propagation before adaptive narrowing.",
        "category": "Correction Propagation",
        "min_value": 0.25,
        "max_value": 4.0,
        "step": 0.25,
    },
    {
        "key": "propagation.min_observations_for_narrow",
        "default_value": "10",
        "value_type": "number",
        "label": "Min Observations to Narrow",
        "description": "Minimum observations in a context cell before radius begins narrowing.",
        "category": "Correction Propagation",
        "min_value": 5,
        "max_value": 50,
        "step": 5,
    },
    {
        "key": "propagation.kernel_bandwidth",
        "default_value": "0.5",
        "value_type": "number",
        "label": "Kernel Bandwidth",
        "description": "Gaussian kernel bandwidth for context similarity weighting.",
        "category": "Correction Propagation",
        "min_value": 0.1,
        "max_value": 2.0,
        "step": 0.1,
    },
    {
        "key": "propagation.replay_buffer_size",
        "default_value": "200",
        "value_type": "number",
        "label": "Replay Buffer Size",
        "description": "Maximum entries in the prioritized experience replay buffer.",
        "category": "Correction Propagation",
        "min_value": 50,
        "max_value": 1000,
        "step": 50,
    },
    {
        "key": "propagation.replay_alpha",
        "default_value": "0.6",
        "value_type": "number",
        "label": "Replay Alpha",
        "description": "Prioritization exponent for replay sampling (0=uniform, 1=fully prioritized).",
        "category": "Correction Propagation",
        "min_value": 0.0,
        "max_value": 1.0,
        "step": 0.1,
    },
    {
        "key": "propagation.max_propagations_per_score",
        "default_value": "5",
        "value_type": "number",
        "label": "Max Propagations per Score",
        "description": "Maximum number of nearby contexts to propagate each correction to.",
        "category": "Correction Propagation",
        "min_value": 1,
        "max_value": 20,
        "step": 1,
    },
    # ── ML Pipeline ──────────────────────────────────────────────────
    {
        "key": "ml.tier_override",
        "default_value": "auto",
        "value_type": "select",
        "label": "ML Tier Override",
        "description": "Override auto-detected ML tier. auto=use hardware detection.",
        "category": "ml",
        "options": "auto,1,2,3,4",
    },
    {
        "key": "ml.fallback_ttl_days",
        "default_value": "7",
        "value_type": "number",
        "label": "Fallback TTL (days)",
        "description": "Days before retrying a model that fell back to lower tier.",
        "category": "ml",
        "min_value": 1,
        "max_value": 30,
    },
    {
        "key": "ml.feature_prune_threshold",
        "default_value": "0.01",
        "value_type": "number",
        "label": "Feature Prune Threshold",
        "description": "Features below this importance are candidates for pruning.",
        "category": "ml",
        "min_value": 0.001,
        "max_value": 0.1,
        "step": 0.005,
    },
    {
        "key": "ml.feature_prune_cycles",
        "default_value": "3",
        "value_type": "number",
        "label": "Feature Prune Cycles",
        "description": "Consecutive low-importance cycles before auto-pruning (Tier 3+).",
        "category": "ml",
        "min_value": 1,
        "max_value": 10,
    },
    {
        "key": "ml.optuna_trials",
        "default_value": "20",
        "value_type": "number",
        "label": "Optuna Trials",
        "description": "Number of hyperparameter optimization trials per training cycle (Tier 3+).",
        "category": "ml",
        "min_value": 5,
        "max_value": 100,
    },
    {
        "key": "ml.cv_folds",
        "default_value": "auto",
        "value_type": "select",
        "label": "CV Folds",
        "description": "Cross-validation folds. auto=tier-based (1/3/5).",
        "category": "ml",
        "options": "auto,1,3,5,10",
    },
    {
        "key": "ml.online_blend_weight",
        "default_value": "0.3",
        "value_type": "number",
        "label": "Online Blend Weight",
        "description": (
            "Weight for online model predictions in ensemble blend (0=disabled, 1=online only). Tier 3+ only."
        ),
        "category": "ml",
        "min_value": 0.0,
        "max_value": 1.0,
        "step": 0.05,
    },
    {
        "key": "ml.online_min_samples",
        "default_value": "5",
        "value_type": "number",
        "label": "Online Min Samples",
        "description": "Minimum observations before online model starts predicting.",
        "category": "ml",
        "min_value": 1,
        "max_value": 50,
    },
    {
        "key": "ml.auto_tune_weights",
        "default_value": "true",
        "value_type": "boolean",
        "label": "Auto-Tune Weights",
        "description": "Automatically adjust ensemble weights based on rolling MAE (Tier 3+).",
        "category": "ml",
    },
    {
        "key": "ml.weight_tuner_window_days",
        "default_value": "7",
        "value_type": "number",
        "label": "Weight Tuner Window (days)",
        "description": "Rolling window for MAE-based weight computation.",
        "category": "ml",
        "min_value": 1,
        "max_value": 30,
    },
    # Phase 3: Pattern Recognition
    {
        "key": "pattern.min_tier",
        "default_value": "3",
        "value_type": "number",
        "label": "Pattern Recognition Min Tier",
        "description": "Minimum hardware tier required to activate pattern recognition (1-4).",
        "category": "pattern",
        "min_value": 1,
        "max_value": 4,
        "step": 1,
    },
    {
        "key": "pattern.sequence_window_size",
        "default_value": "6",
        "value_type": "number",
        "label": "Sequence Window Size",
        "description": "Number of snapshots in the trajectory classification sliding window",
        "category": "pattern",
        "min_value": 3,
        "max_value": 24,
        "step": 1,
    },
    {
        "key": "pattern.dtw_neighbors",
        "default_value": "3",
        "value_type": "number",
        "label": "DTW Neighbors",
        "description": "Number of neighbors for DTW sequence classifier (higher = smoother, slower)",
        "category": "pattern",
        "min_value": 1,
        "max_value": 10,
        "step": 1,
    },
    {
        "key": "pattern.anomaly_top_n",
        "default_value": "3",
        "value_type": "number",
        "label": "Anomaly Top Features",
        "description": "Number of top contributing features to report per anomaly",
        "category": "pattern",
        "min_value": 1,
        "max_value": 10,
        "step": 1,
    },
    {
        "key": "pattern.trajectory_change_threshold",
        "default_value": "0.20",
        "value_type": "number",
        "label": "Trajectory Change Threshold",
        "description": "Minimum percent change in target metric to classify as ramping/winding (0.0-1.0)",
        "category": "pattern",
        "min_value": 0.05,
        "max_value": 0.50,
        "step": 0.05,
    },
    # Phase 4: Transfer & Attention
    {
        "key": "transfer.min_similarity",
        "default_value": "0.6",
        "value_type": "number",
        "label": "Transfer Min Similarity",
        "description": "Minimum Jaccard structural similarity for generating transfer candidates (0.4-1.0)",
        "category": "transfer",
        "min_value": 0.4,
        "max_value": 1.0,
        "step": 0.05,
    },
    {
        "key": "transfer.promotion_days",
        "default_value": "7",
        "value_type": "number",
        "label": "Transfer Promotion Days",
        "description": "Minimum days of shadow testing before a transfer candidate can be promoted",
        "category": "transfer",
        "min_value": 3,
        "max_value": 30,
        "step": 1,
    },
    {
        "key": "transfer.promotion_hit_rate",
        "default_value": "0.6",
        "value_type": "number",
        "label": "Transfer Promotion Hit Rate",
        "description": "Minimum shadow hit rate for transfer promotion (0.0-1.0)",
        "category": "transfer",
        "min_value": 0.3,
        "max_value": 0.9,
        "step": 0.05,
    },
    {
        "key": "transfer.reject_hit_rate",
        "default_value": "0.3",
        "value_type": "number",
        "label": "Transfer Reject Hit Rate",
        "description": "Shadow hit rate below which transfer candidates are rejected (0.0-1.0)",
        "category": "transfer",
        "min_value": 0.1,
        "max_value": 0.5,
        "step": 0.05,
    },
    {
        "key": "attention.hidden_dim",
        "default_value": "32",
        "value_type": "number",
        "label": "Attention Hidden Dim",
        "description": "Hidden dimension for the attention autoencoder (Tier 4 only, higher = more expressive)",
        "category": "attention",
        "min_value": 8,
        "max_value": 128,
        "step": 8,
    },
    {
        "key": "attention.train_epochs",
        "default_value": "20",
        "value_type": "number",
        "label": "Attention Training Epochs",
        "description": "Training epochs for the attention autoencoder (Tier 4 only)",
        "category": "attention",
        "min_value": 5,
        "max_value": 100,
        "step": 5,
    },
    # ── Audit Logger ───────────────────────────────────────────────────
    {
        "key": "audit.enabled",
        "default_value": "true",
        "value_type": "boolean",
        "label": "Audit Logging Enabled",
        "description": "Master switch for audit event logging.",
        "category": "Audit",
    },
    {
        "key": "audit.retention_days",
        "default_value": "90",
        "value_type": "number",
        "label": "Audit Retention (days)",
        "description": "Number of days to retain audit events before pruning.",
        "category": "Audit",
        "min_value": 7,
        "max_value": 365,
        "step": 1,
    },
    {
        "key": "audit.log_api_requests",
        "default_value": "true",
        "value_type": "boolean",
        "label": "Log API Requests",
        "description": "Record all inbound API requests in the audit log.",
        "category": "Audit",
    },
    {
        "key": "audit.log_cache_writes",
        "default_value": "true",
        "value_type": "boolean",
        "label": "Log Cache Writes",
        "description": "Record cache write operations in the audit log.",
        "category": "Audit",
    },
    {
        "key": "audit.buffer_size",
        "default_value": "10000",
        "value_type": "number",
        "label": "Write Buffer Size",
        "description": "Maximum number of audit events to hold in memory before flushing.",
        "category": "Audit",
        "min_value": 1000,
        "max_value": 100000,
        "step": 1000,
    },
    {
        "key": "audit.flush_interval_ms",
        "default_value": "500",
        "value_type": "number",
        "label": "Flush Interval (ms)",
        "description": "Milliseconds between automatic buffer flushes to disk.",
        "category": "Audit",
        "min_value": 100,
        "max_value": 5000,
        "step": 100,
    },
    {
        "key": "audit.alert_on_errors",
        "default_value": "true",
        "value_type": "boolean",
        "label": "Alert on Errors",
        "description": "Send alerts when audit error rate exceeds the configured threshold.",
        "category": "Audit",
    },
    {
        "key": "audit.alert_threshold",
        "default_value": "10",
        "value_type": "number",
        "label": "Alert Threshold",
        "description": "Number of errors within the alert window that triggers an alert.",
        "category": "Audit",
        "min_value": 1,
        "max_value": 100,
        "step": 1,
    },
    {
        "key": "audit.alert_window_minutes",
        "default_value": "5",
        "value_type": "number",
        "label": "Alert Window (min)",
        "description": "Rolling window in minutes used to count errors for alerting.",
        "category": "Audit",
        "min_value": 1,
        "max_value": 60,
        "step": 1,
    },
    {
        "key": "audit.archive_on_prune",
        "default_value": "true",
        "value_type": "boolean",
        "label": "Archive Before Prune",
        "description": "Export events to JSONL archive before deleting them during pruning.",
        "category": "Audit",
    },
    # ── Presence Weights ─────────────────────────────────────────────
    {
        "key": "presence.weight.motion",
        "default_value": "0.9",
        "value_type": "number",
        "label": "Motion Sensor Trust",
        "description": ("Bayesian prior weight for motion-type signals in the occupancy fusion algorithm."),
        "description_layman": (
            "How much should ARIA trust motion sensors? Higher means motion alone can determine someone is in a room."
        ),
        "description_technical": (
            "Bayesian prior weight for motion signals in occupancy fusion."
            " Range 0.1-1.0. Default 0.9. At 0.1: motion barely registers,"
            " needs corroboration. At 1.0: single motion event = 100%"
            " confidence, may false-positive from pets."
        ),
        "category": "Presence Weights",
        "min_value": 0.1,
        "max_value": 1.0,
        "step": 0.05,
    },
    {
        "key": "presence.weight.door",
        "default_value": "0.6",
        "value_type": "number",
        "label": "Door Sensor Trust",
        "description": ("Bayesian prior weight for door-type signals in the occupancy fusion algorithm."),
        "description_layman": (
            "How much should ARIA trust door sensors? A door opening suggests someone entered, but isn't as definitive."
        ),
        "description_technical": (
            "Bayesian prior weight for door open/close signals."
            " Range 0.1-1.0. Default 0.6. Door events are directionally"
            " ambiguous (enter vs exit), hence moderate default."
        ),
        "category": "Presence Weights",
        "min_value": 0.1,
        "max_value": 1.0,
        "step": 0.05,
    },
    {
        "key": "presence.weight.media",
        "default_value": "0.4",
        "value_type": "number",
        "label": "Media Player Trust",
        "description": ("Bayesian prior weight for media player signals in the occupancy fusion algorithm."),
        "description_layman": (
            "How much should ARIA trust media players (TV, speakers)?"
            " A playing TV suggests someone is watching, but could be left on."
        ),
        "description_technical": (
            "Bayesian prior weight for media_player state signals."
            " Range 0.1-1.0. Default 0.4. Lower because media devices"
            " are often left playing in empty rooms."
        ),
        "category": "Presence Weights",
        "min_value": 0.1,
        "max_value": 1.0,
        "step": 0.05,
    },
    {
        "key": "presence.weight.power",
        "default_value": "0.3",
        "value_type": "number",
        "label": "Power Draw Trust",
        "description": ("Bayesian prior weight for power draw signals in the occupancy fusion algorithm."),
        "description_layman": (
            "How much should ARIA trust power consumption? High power"
            " draw hints someone is using appliances, but many run on their own."
        ),
        "description_technical": (
            "Bayesian prior weight for power consumption signals."
            " Range 0.1-1.0. Default 0.3. Lowest default weight —"
            " appliances cycle autonomously, fridges/HVACs confound."
        ),
        "category": "Presence Weights",
        "min_value": 0.1,
        "max_value": 1.0,
        "step": 0.05,
    },
    {
        "key": "presence.weight.device_tracker",
        "default_value": "0.5",
        "value_type": "number",
        "label": "Device Tracker Trust",
        "description": ("Bayesian prior weight for device tracker signals in the occupancy fusion algorithm."),
        "description_layman": (
            "How much should ARIA trust phone location? Phones at home"
            " mean someone is likely home, but could be left behind."
        ),
        "description_technical": (
            "Bayesian prior weight for device_tracker (phone presence)."
            " Range 0.1-1.0. Default 0.5. Binary signal (home/away),"
            " no decay. Moderate default accounts for phones left at home."
        ),
        "category": "Presence Weights",
        "min_value": 0.1,
        "max_value": 1.0,
        "step": 0.05,
    },
    {
        "key": "presence.weight.camera_person",
        "default_value": "0.95",
        "value_type": "number",
        "label": "Camera Person Detection Trust",
        "description": "Bayesian prior weight for camera person detection signals.",
        "description_layman": (
            "How much should ARIA trust camera person detection? Very high — seeing a person shape is strong evidence."
        ),
        "description_technical": (
            "Bayesian prior weight for Frigate person detection via MQTT."
            " Range 0.1-1.0. Default 0.95. Near-certain: person-class"
            " detection is highly reliable with modern models."
        ),
        "category": "Presence Weights",
        "min_value": 0.1,
        "max_value": 1.0,
        "step": 0.05,
    },
    {
        "key": "presence.weight.camera_face",
        "default_value": "1.0",
        "value_type": "number",
        "label": "Camera Face Recognition Trust",
        "description": "Bayesian prior weight for camera face recognition signals.",
        "description_layman": (
            "How much should ARIA trust face recognition? Maximum —"
            " a recognized face is the strongest possible evidence."
        ),
        "description_technical": (
            "Bayesian prior weight for face recognition signals."
            " Range 0.1-1.0. Default 1.0. Maximum confidence:"
            " identified face is definitive. No decay (binary)."
        ),
        "category": "Presence Weights",
        "min_value": 0.1,
        "max_value": 1.0,
        "step": 0.05,
    },
    {
        "key": "presence.weight.light_interaction",
        "default_value": "0.7",
        "value_type": "number",
        "label": "Light Interaction Trust",
        "description": "Bayesian prior weight for light on/off interaction signals.",
        "description_layman": (
            "How much should ARIA trust light switches? Someone turning"
            " a light on/off is good evidence they are in the room."
        ),
        "description_technical": (
            "Bayesian prior weight for light state change events."
            " Range 0.1-1.0. Default 0.7. Light toggling implies human"
            " action, though automations can also toggle lights."
        ),
        "category": "Presence Weights",
        "min_value": 0.1,
        "max_value": 1.0,
        "step": 0.05,
    },
    {
        "key": "presence.weight.dimmer_press",
        "default_value": "0.85",
        "value_type": "number",
        "label": "Dimmer Press Trust",
        "description": "Bayesian prior weight for physical dimmer press signals.",
        "description_layman": (
            "How much should ARIA trust dimmer button presses? Very strong — only a person can push a physical button."
        ),
        "description_technical": (
            "Bayesian prior weight for physical dimmer/button press events."
            " Range 0.1-1.0. Default 0.85. Higher than light_interaction"
            " because automations cannot trigger physical presses."
        ),
        "category": "Presence Weights",
        "min_value": 0.1,
        "max_value": 1.0,
        "step": 0.05,
    },
    # ── Presence Decay Times ──────────────────────────────────────────
    {
        "key": "presence.decay.motion",
        "default_value": "300",
        "value_type": "number",
        "label": "Motion Decay (seconds)",
        "description": "How long a motion signal persists before fading.",
        "description_layman": (
            "After motion is detected, how many seconds until ARIA forgets about it? Default 5 minutes."
        ),
        "description_technical": (
            "Exponential decay time constant for motion signals in seconds."
            " Default 300 (5 min). Short decay avoids stale motion keeping"
            " rooms 'occupied' after people leave."
        ),
        "category": "Presence Decay",
        "min_value": 30,
        "max_value": 3600,
        "step": 30,
    },
    {
        "key": "presence.decay.door",
        "default_value": "600",
        "value_type": "number",
        "label": "Door Decay (seconds)",
        "description": "How long a door signal persists before fading.",
        "description_layman": (
            "After a door opens, how many seconds until ARIA stops considering it? Default 10 minutes."
        ),
        "description_technical": (
            "Decay time constant for door open/close signals."
            " Default 600 (10 min). Longer than motion because a door"
            " event implies a transition that takes time to resolve."
        ),
        "category": "Presence Decay",
        "min_value": 30,
        "max_value": 3600,
        "step": 30,
    },
    {
        "key": "presence.decay.media",
        "default_value": "1800",
        "value_type": "number",
        "label": "Media Decay (seconds)",
        "description": "How long a media player signal persists before fading.",
        "description_layman": (
            "After media stops playing, how many seconds until ARIA stops"
            " counting it? Default 30 min — someone might pause and return."
        ),
        "description_technical": (
            "Decay time constant for media player signals."
            " Default 1800 (30 min). Long decay because media sessions are"
            " typically continuous and pausing doesn't mean leaving."
        ),
        "category": "Presence Decay",
        "min_value": 60,
        "max_value": 7200,
        "step": 60,
    },
    {
        "key": "presence.decay.power",
        "default_value": "3600",
        "value_type": "number",
        "label": "Power Draw Decay (seconds)",
        "description": "How long a power draw signal persists before fading.",
        "description_layman": (
            "After high power draw is detected, how long until ARIA forgets?"
            " Default 1 hour — appliances often run in long cycles."
        ),
        "description_technical": (
            "Decay time constant for power consumption signals."
            " Default 3600 (1 hr). Longest decay — power cycles are slow"
            " (washer, dryer, HVAC). Reducing below 600s may cause flapping."
        ),
        "category": "Presence Decay",
        "min_value": 60,
        "max_value": 7200,
        "step": 60,
    },
    {
        "key": "presence.decay.device_tracker",
        "default_value": "0",
        "value_type": "number",
        "label": "Device Tracker Decay (seconds)",
        "description": "Decay for device tracker signals. 0 means binary (no decay).",
        "description_layman": (
            "How quickly should phone presence fade? Default 0 (instant) — either the phone is home or it isn't."
        ),
        "description_technical": (
            "Decay time constant for device_tracker. Default 0 (binary,"
            " no decay). device_tracker is a persistent state (home/away),"
            " not a transient event. Set >0 if tracker is unreliable."
        ),
        "category": "Presence Decay",
        "min_value": 0,
        "max_value": 3600,
        "step": 30,
    },
    {
        "key": "presence.decay.camera_person",
        "default_value": "120",
        "value_type": "number",
        "label": "Camera Person Decay (seconds)",
        "description": ("How long a camera person detection persists before fading."),
        "description_layman": (
            "After a camera sees a person, how long does that count?"
            " Default 2 min — short because the camera keeps updating."
        ),
        "description_technical": (
            "Decay time constant for Frigate person detection. Default 120"
            " (2 min). Short because Frigate sends continuous events while"
            " a person is visible — decay only matters after they leave."
        ),
        "category": "Presence Decay",
        "min_value": 30,
        "max_value": 1800,
        "step": 30,
    },
    {
        "key": "presence.decay.camera_face",
        "default_value": "0",
        "value_type": "number",
        "label": "Camera Face Decay (seconds)",
        "description": "Decay for face recognition signals. 0 means binary (no decay).",
        "description_layman": (
            "How quickly should face recognition fade? Default 0"
            " — once recognized, presence is confirmed until disproven."
        ),
        "description_technical": (
            "Decay time for face recognition. Default 0 (binary). Face"
            " recognition is a discrete identification event — presence"
            " persists until contradicted by other signals."
        ),
        "category": "Presence Decay",
        "min_value": 0,
        "max_value": 1800,
        "step": 30,
    },
    {
        "key": "presence.decay.light_interaction",
        "default_value": "600",
        "value_type": "number",
        "label": "Light Interaction Decay (seconds)",
        "description": "How long a light interaction signal persists before fading.",
        "description_layman": (
            "After someone toggles a light, how long does that count as evidence? Default 10 minutes."
        ),
        "description_technical": (
            "Decay time for light toggle events. Default 600 (10 min)."
            " Matches door decay — a light interaction implies someone"
            " is actively using the room."
        ),
        "category": "Presence Decay",
        "min_value": 30,
        "max_value": 3600,
        "step": 30,
    },
    {
        "key": "presence.decay.dimmer_press",
        "default_value": "300",
        "value_type": "number",
        "label": "Dimmer Press Decay (seconds)",
        "description": "How long a dimmer press signal persists before fading.",
        "description_layman": (
            "After someone presses a dimmer button, how long does that"
            " count? Default 5 min — like motion, a point-in-time event."
        ),
        "description_technical": (
            "Decay time for physical button press events. Default 300"
            " (5 min). Same as motion — physical interaction is a point"
            " event, not a continuous state."
        ),
        "category": "Presence Decay",
        "min_value": 30,
        "max_value": 3600,
        "step": 30,
    },
    # ── Module Data Sources ────────────────────────────────────────────
    {
        "key": "presence.enabled_signals",
        "default_value": (
            "camera_person,camera_face,motion,light_interaction,dimmer_press,door,media_active,device_tracker"
        ),
        "value_type": "string",
        "label": "Enabled Signals",
        "description": "Comma-separated list of signal types that feed presence detection.",
        "category": "Presence",
    },
    {
        "key": "activity.enabled_domains",
        "default_value": "light,switch,binary_sensor,media_player,climate,cover",
        "value_type": "string",
        "label": "Enabled Domains",
        "description": "Comma-separated list of entity domains tracked for activity monitoring.",
        "category": "Activity Monitor",
    },
    {
        "key": "anomaly.enabled_entities",
        "default_value": "light,binary_sensor,climate,media_player,switch",
        "value_type": "string",
        "label": "Enabled Entity Domains",
        "description": "Comma-separated entity IDs for anomaly detection, or 'all' for all discovered entities.",
        "category": "ML Pipeline",
    },
    {
        "key": "shadow.enabled_capabilities",
        "default_value": "light,binary_sensor,climate,media_player",
        "value_type": "string",
        "label": "Enabled Capability Domains",
        "description": (
            "Comma-separated capability names for shadow prediction, or 'all' for all with can_predict=true."
        ),
        "category": "Shadow Mode",
    },
    {
        "key": "discovery.domain_filter",
        "default_value": "light,switch,binary_sensor,sensor,climate,cover,media_player,fan,lock",
        "value_type": "string",
        "label": "Domain Filter",
        "description": "Comma-separated entity domains to include in capability discovery.",
        "category": "Discovery",
    },
]


async def seed_config_defaults(cache) -> int:
    """Seed all config defaults into the database.

    Uses INSERT OR IGNORE so existing user overrides are preserved.

    Args:
        cache: CacheManager instance (must be initialized).

    Returns:
        Number of new parameters inserted.
    """
    inserted = 0
    for param in CONFIG_DEFAULTS:
        was_inserted = await cache.upsert_config_default(
            key=param["key"],
            default_value=param["default_value"],
            value_type=param["value_type"],
            label=param.get("label", ""),
            description=param.get("description", ""),
            category=param.get("category", ""),
            min_value=param.get("min_value"),
            max_value=param.get("max_value"),
            options=param.get("options"),
            step=param.get("step"),
            description_layman=param.get("description_layman"),
            description_technical=param.get("description_technical"),
        )
        if was_inserted:
            inserted += 1
    return inserted
