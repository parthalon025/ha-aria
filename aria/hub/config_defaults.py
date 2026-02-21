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
        "description_layman": (
            "How many times per day ARIA can take a snapshot of your"
            " home's activity. Higher values let ARIA learn faster but"
            " use more system resources."
        ),
        "description_technical": (
            "Hard cap on ActivityMonitor adaptive snapshots per UTC day."
            " Range 5-100, default 20. Counter resets at midnight UTC."
            " At max (100), expect ~6 MB/day of snapshot data. At min"
            " (5), slow-changing entities may be under-sampled."
        ),
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
        "description_layman": (
            "How long ARIA waits between snapshots of your home's"
            " activity. A longer cooldown means fewer snapshots but"
            " less load on your system."
        ),
        "description_technical": (
            "Minimum elapsed seconds between consecutive adaptive"
            " snapshots. Range 300-7200, default 1800 (30 min)."
            " Enforced per-entity. At 300s with cap=100, theoretical"
            " max throughput is ~288/day (cap limits to 100)."
        ),
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
        "description_layman": (
            "How often ARIA saves its collected activity data to"
            " storage. Shorter intervals mean more up-to-date data"
            " but slightly more disk writes."
        ),
        "description_technical": (
            "Interval in seconds between buffer flushes to rolling"
            " cache windows. Range 60-3600, default 900 (15 min)."
            " Events accumulate in memory between flushes. At 60s,"
            " near-real-time but higher I/O. At 3600s, up to 1 hour"
            " of data loss on crash."
        ),
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
        "description_layman": (
            "How many hours of recent activity ARIA keeps in its"
            " short-term memory. Older data is discarded to keep"
            " the system running efficiently."
        ),
        "description_technical": (
            "Rolling window retention for activity cache entries in"
            " hours. Range 6-168, default 24. Windows older than this"
            " are pruned on each flush cycle. At 6h, only very recent"
            " patterns are visible. At 168h (7 days), memory usage"
            " scales linearly with entity count."
        ),
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
        "description_layman": (
            "How many days of detailed device history ARIA keeps."
            " Longer retention lets ARIA spot seasonal patterns"
            " but uses more disk space."
        ),
        "description_technical": (
            "Retention period for raw state_changed events in the"
            " SQLite event store. Range 7-365, default 90. Events"
            " older than this are pruned during maintenance. At 365"
            " days with ~3000 entities, expect 2-5 GB of event data."
        ),
        "category": "Event Store",
        "min_value": 7,
        "max_value": 365,
        "step": 1,
    },
    # ── Event Segments ─────────────────────────────────────────────────
    {
        "key": "segments.generation_interval_m",
        "default_value": "15",
        "value_type": "number",
        "label": "Segment Generation Interval",
        "description": "How often (in minutes) to generate ML feature segments from events.",
        "description_layman": (
            "How often ARIA summarizes recent activity into a learning"
            " snapshot. Shorter intervals capture finer detail but use"
            " more CPU."
        ),
        "description_technical": (
            "SegmentBuilder runs on this interval to convert EventStore"
            " rows into feature segments for ML training. Range 5-60 min,"
            " default 15. At 5 min: ~288 segments/day, high resolution."
            " At 60 min: ~24 segments/day, less CPU, coarser patterns."
        ),
        "category": "Event Segments",
        "min_value": 5,
        "max_value": 60,
        "step": 5,
    },
    {
        "key": "segments.retention_hours",
        "default_value": "168",
        "value_type": "number",
        "label": "Segment Retention (hours)",
        "description": "How long to keep generated segments before pruning.",
        "description_layman": (
            "How many hours of learning snapshots to keep. Longer"
            " retention lets ARIA spot weekly patterns but uses more"
            " disk space."
        ),
        "description_technical": (
            "Segments older than this are pruned during maintenance."
            " Default 168 hours (7 days). At 168h with 15-min intervals:"
            " ~672 segments retained. At 720h (30 days): ~2880 segments."
            " Segments are small (~200 bytes each) so storage is minimal."
        ),
        "category": "Event Segments",
        "min_value": 24,
        "max_value": 720,
        "step": 24,
    },
    # ── Feature Engineering ───────────────────────────────────────────
    {
        "key": "features.decay_half_life_days",
        "default_value": "7",
        "value_type": "number",
        "label": "Decay Half-Life (days)",
        "description": "Exponential decay half-life for training sample recency weighting.",
        "description_layman": (
            "How quickly ARIA forgets old patterns when learning."
            " A shorter value means recent behavior matters more;"
            " a longer value means ARIA has a longer memory."
        ),
        "description_technical": (
            "Half-life in days for exponential recency weighting of"
            " training samples. Range 1-30, default 7. A sample from"
            " N days ago receives weight 0.5^(N/half_life). At 1 day,"
            " week-old data has weight ~0.008. At 30 days, month-old"
            " data retains ~50% influence."
        ),
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
        "description_layman": (
            "ARIA gives extra importance to patterns from the same"
            " day of the week. If your Monday routine differs from"
            " Saturday, raising this helps ARIA learn the difference."
        ),
        "description_technical": (
            "Multiplicative bonus applied to training samples that"
            " share the same weekday as the prediction target."
            " Range 1.0-3.0, default 1.5. At 1.0, no weekday"
            " preference. At 3.0, same-weekday samples have 3x"
            " weight, strongly biasing toward weekly periodicity."
        ),
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
        "description_layman": (
            "The minimum certainty ARIA needs before it records a"
            " prediction. Lower values let ARIA record more guesses;"
            " higher values mean only confident predictions are kept."
        ),
        "description_technical": (
            "Confidence floor for shadow prediction storage."
            " Range 0.05-0.9, default 0.3. Predictions with"
            " confidence < this are silently dropped. At 0.05,"
            " nearly all predictions stored (noisy). At 0.9,"
            " only high-confidence predictions survive, reducing"
            " training signal diversity."
        ),
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
        "description_layman": (
            "How long ARIA waits to see if a prediction came true."
            " A longer window gives predictions more time to be"
            " verified but delays scoring."
        ),
        "description_technical": (
            "Default evaluation window for shadow predictions in"
            " seconds. Range 60-3600, default 600 (10 min). After"
            " this window expires, the prediction is resolved as"
            " hit or miss. Short windows bias toward fast-changing"
            " entities; long windows suit slow transitions."
        ),
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
        "description_layman": (
            "How often ARIA checks whether its predictions were"
            " correct. More frequent checks mean faster feedback"
            " but use slightly more processing power."
        ),
        "description_technical": (
            "Interval in seconds between resolution sweeps for"
            " expired prediction windows. Range 10-300, default 60."
            " Each sweep queries all pending predictions and resolves"
            " expired ones. At 10s, near-real-time resolution but"
            " higher DB load. At 300s, up to 5 min resolution lag."
        ),
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
        "description_layman": (
            "How long ARIA waits after making one prediction before"
            " it can make another for the same device. Prevents"
            " ARIA from making too many rapid guesses."
        ),
        "description_technical": (
            "Per-entity debounce for prediction attempts in seconds."
            " Range 5-300, default 30. Prevents burst predictions"
            " during rapid state changes. At 5s, nearly every event"
            " triggers a prediction. At 300s, at most one prediction"
            " per 5 minutes per entity."
        ),
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
        "description_layman": (
            "Controls how ARIA balances trying new predictions versus"
            " sticking with what has worked before. Thompson is"
            " smarter but more complex; epsilon is simpler."
        ),
        "description_technical": (
            "Exploration strategy for shadow predictions. 'thompson'"
            " uses Bayesian Thompson Sampling with f-dsw discounting"
            " for adaptive exploration. 'epsilon' uses a fixed"
            " random exploration rate. Thompson adapts automatically;"
            " epsilon is deterministic and easier to debug."
        ),
        "category": "Shadow Engine",
        "options": "thompson,epsilon",
    },
    {
        "key": "shadow.thompson_discount_factor",
        "default_value": "0.95",
        "value_type": "number",
        "label": "Thompson Discount Factor",
        "description": "f-dsw discount factor for Thompson Sampling posteriors (lower = faster adaptation).",
        "description_layman": (
            "Controls how quickly ARIA adapts its prediction strategy"
            " to new patterns. Lower values make ARIA adapt faster"
            " but forget old patterns sooner."
        ),
        "description_technical": (
            "Forgetting-discounted sliding window factor for Thompson"
            " Sampling posterior updates. Range 0.8-1.0, default 0.95."
            " At 0.8, effective history ~5 observations. At 1.0, no"
            " discounting (infinite memory). Applied multiplicatively"
            " to alpha/beta parameters each round."
        ),
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
        "description_layman": (
            "How many past predictions ARIA considers when deciding"
            " its strategy. More history means more stable behavior;"
            " less history means faster adaptation to changes."
        ),
        "description_technical": (
            "Maximum effective sample count for Thompson Sampling"
            " buckets before oldest observations are aged out."
            " Range 20-500, default 100. Interacts with discount"
            " factor: effective window = min(window_size,"
            " 1/(1-discount)). At 20, very reactive. At 500,"
            " very stable but slow to shift."
        ),
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
        "description_layman": (
            "Types of Home Assistant devices that ARIA ignores"
            " automatically, like buttons, scenes, and weather."
            " These devices don't have useful patterns for learning."
        ),
        "description_technical": (
            "Comma-separated HA domains excluded from entity curation."
            " Default excludes ~20 non-behavioral domains (update,"
            " scene, input_*, etc.). Adding a domain here removes all"
            " its entities from training. Removing a domain re-exposes"
            " those entities at next discovery cycle."
        ),
        "category": "Data Quality",
    },
    {
        "key": "curation.noise_event_threshold",
        "default_value": "1000",
        "value_type": "number",
        "label": "Noise Event Threshold",
        "description": "Daily event count above which an entity is considered noise (if low state variety).",
        "description_layman": (
            "If a device sends more than this many updates per day"
            " without really changing, ARIA considers it noisy and"
            " ignores it. Helps filter out chatty sensors."
        ),
        "description_technical": (
            "Daily event count threshold for noise classification."
            " Range 100-10000, default 1000. Entities exceeding this"
            " with low state variety (entropy) are auto-excluded."
            " At 100, aggressive filtering may exclude legitimate"
            " high-frequency sensors. At 10000, only extreme cases"
            " are caught."
        ),
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
        "description_layman": (
            "If a device hasn't changed state in this many days,"
            " ARIA stops tracking it. This keeps ARIA focused on"
            " devices that are actually being used."
        ),
        "description_technical": (
            "Days of inactivity before an entity is auto-excluded"
            " from curation. Range 7-90, default 30. Checked during"
            " discovery refresh. At 7, seasonal devices (e.g. holiday"
            " lights) may be excluded prematurely. At 90, truly dead"
            " entities linger in the training set."
        ),
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
        "description_layman": (
            "If a device has been showing as 'unavailable' for this"
            " many hours, ARIA stops tracking it. Set to 0 to"
            " disable this check entirely."
        ),
        "description_technical": (
            "Hours of continuous 'unavailable' state before auto-"
            "exclusion. Range 0-168, default 0 (disabled). At 0,"
            " unavailable entities are never auto-excluded. At 1,"
            " any brief network hiccup triggers exclusion. At 168"
            " (7 days), only long-term outages are caught."
        ),
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
        "description_layman": (
            "Words that identify vehicle-related devices in your"
            " Home Assistant. ARIA uses these to find car-related"
            " entities like Tesla or other vehicle integrations."
        ),
        "description_technical": (
            "Comma-separated substring patterns matched against"
            " entity_id for vehicle classification. Default includes"
            " common vehicle integration names. Matched case-"
            "insensitively. Used by curation to tag vehicle entities"
            " for specialized handling in presence detection."
        ),
        "category": "Data Quality",
    },
    # ── Anomaly Detection ──────────────────────────────────────────────
    {
        "key": "anomaly.use_autoencoder",
        "default_value": "true",
        "value_type": "boolean",
        "label": "Use Autoencoder",
        "description": "Enable hybrid autoencoder + IsolationForest anomaly detection.",
        "description_layman": (
            "Whether ARIA uses an advanced neural network alongside"
            " its basic anomaly detector. When enabled, ARIA is"
            " better at spotting unusual patterns in your home."
        ),
        "description_technical": (
            "Enables hybrid autoencoder + IsolationForest anomaly"
            " detection pipeline. When true, reconstruction error"
            " from the autoencoder is combined with IsolationForest"
            " scores. When false, only IsolationForest is used."
            " Autoencoder adds ~2x training time but catches"
            " non-linear anomalies that IsolationForest misses."
        ),
        "category": "Anomaly Detection",
    },
    {
        "key": "anomaly.ae_hidden_layers",
        "default_value": "24,12,24",
        "value_type": "string",
        "label": "Autoencoder Hidden Layers",
        "description": "Comma-separated hidden layer sizes for the autoencoder (e.g. 24,12,24).",
        "description_layman": (
            "Controls the size of the neural network ARIA uses to"
            " detect anomalies. The default works well for most"
            " homes. Only change if you have unusual needs."
        ),
        "description_technical": (
            "Comma-separated hidden layer sizes for the MLPRegressor"
            " autoencoder. Default '24,12,24' (bottleneck architecture)."
            " The middle layer (12) is the compression dimension."
            " Wider layers increase capacity but risk overfitting."
            " Must be symmetric for proper reconstruction."
        ),
        "category": "Anomaly Detection",
    },
    {
        "key": "anomaly.ae_max_iter",
        "default_value": "200",
        "value_type": "number",
        "label": "Autoencoder Max Iterations",
        "description": "Maximum training iterations for the autoencoder.",
        "description_layman": (
            "How long ARIA trains its anomaly detection model."
            " More iterations can improve accuracy but take longer"
            " to complete during each training cycle."
        ),
        "description_technical": (
            "Maximum training iterations for the MLPRegressor"
            " autoencoder. Range 50-1000, default 200. Training"
            " uses early stopping, so actual iterations may be"
            " fewer. At 50, likely underfitting. At 1000, diminishing"
            " returns and ~5x longer training time on RPi hardware."
        ),
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
        "description_layman": (
            "What percentage of your home's activity ARIA should"
            " consider unusual. A higher value means ARIA flags"
            " more things as anomalies; lower means fewer alerts."
        ),
        "description_technical": (
            "Expected anomaly proportion passed to IsolationForest"
            " contamination parameter. Range 0.01-0.20, default 0.05."
            " Directly sets the decision boundary. At 0.01, only"
            " extreme outliers flagged. At 0.20, 1 in 5 samples"
            " classified anomalous, likely many false positives."
        ),
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
        "description_layman": (
            "Which forecasting engine ARIA uses to predict future"
            " device states. 'auto' picks the best available option."
            " Most users should leave this on auto."
        ),
        "description_technical": (
            "Forecasting backend selection. Options: auto (prefers"
            " NeuralProphet, falls back to Prophet), neuralprophet"
            " (requires torch), prophet (Facebook Prophet). Auto"
            " checks import availability at startup. NeuralProphet"
            " supports AR and lagged regressors; Prophet is more"
            " stable on low-memory hardware."
        ),
        "category": "Forecaster",
        "options": "auto,neuralprophet,prophet",
    },
    {
        "key": "forecaster.epochs",
        "default_value": "100",
        "value_type": "number",
        "label": "Training Epochs",
        "description": "Number of training epochs for NeuralProphet (ignored by Prophet).",
        "description_layman": (
            "How many times ARIA's forecaster reviews the training"
            " data. More passes can improve accuracy but take longer."
            " Only applies when using the NeuralProphet backend."
        ),
        "description_technical": (
            "Training epochs for NeuralProphet backend. Range 10-500,"
            " default 100. Ignored when backend=prophet. Higher"
            " values reduce training loss but risk overfitting on"
            " small datasets. On RPi, each epoch adds ~0.5s."
            " Early stopping may terminate before reaching this limit."
        ),
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
        "description_layman": (
            "How big of adjustments ARIA's forecaster makes while"
            " learning. Smaller values learn slowly but carefully;"
            " larger values learn fast but may overshoot."
        ),
        "description_technical": (
            "Adam optimizer learning rate for NeuralProphet."
            " Range 0.001-1.0, default 0.1. Ignored when"
            " backend=prophet. At 0.001, very slow convergence"
            " (may need 500+ epochs). At 1.0, likely diverges."
            " NeuralProphet's default scheduler may adjust this"
            " during training."
        ),
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
        "description_layman": (
            "How many days of past data ARIA's forecaster looks back"
            " at when making predictions. The default of 7 means it"
            " looks at the past week of patterns."
        ),
        "description_technical": (
            "Autoregression order (lag count) for NeuralProphet."
            " Range 1-30, default 7. Each lag adds a feature column"
            " for that day's historical value. At 1, only yesterday's"
            " value used. At 30, a full month of lags — increases"
            " model complexity and memory. Ignored by Prophet backend."
        ),
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
        "description_layman": (
            "How sensitive ARIA is to detecting changes in your"
            " home's patterns. Lower values mean ARIA is more"
            " cautious and only reacts to clear changes."
        ),
        "description_technical": (
            "ADWIN (Adaptive Windowing) confidence parameter delta."
            " Range 0.0001-0.1, default 0.002. Controls the"
            " statistical significance threshold for detecting"
            " distribution change. At 0.0001, very conservative"
            " (few false alarms, slow detection). At 0.1, very"
            " sensitive (fast detection, many false positives)."
        ),
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
        "description_layman": (
            "Whether ARIA requires multiple signals to agree before"
            " deciding your home's patterns have changed. Enabling"
            " this reduces false alarms but may delay adaptation."
        ),
        "description_technical": (
            "When true, drift detection requires confirmation from"
            " both ADWIN and Page-Hinkley (or threshold) detectors"
            " before triggering model retrain. When false, any"
            " single detector can trigger retrain independently."
            " Recommended true for stable environments."
        ),
        "category": "Drift Detection",
    },
    {
        "key": "drift.page_hinkley_enabled",
        "default_value": "true",
        "value_type": "boolean",
        "label": "Page-Hinkley Enabled",
        "description": "Whether to run Page-Hinkley drift detection alongside threshold checks.",
        "description_layman": (
            "Enables an additional method for detecting when your"
            " home's behavior patterns have shifted. Provides a"
            " second opinion alongside the primary drift detector."
        ),
        "description_technical": (
            "Enables Page-Hinkley change detection test alongside"
            " ADWIN threshold checks. Page-Hinkley detects gradual"
            " drift (mean shift) that ADWIN may miss. When both are"
            " enabled and require_confirmation=true, both must agree"
            " before triggering retrain."
        ),
        "category": "Drift Detection",
    },
    # ── Reference Model ────────────────────────────────────────────────
    {
        "key": "reference_model.enabled",
        "default_value": "true",
        "value_type": "boolean",
        "label": "Reference Model Enabled",
        "description": "Enable clean reference model for distinguishing meta-learner errors from behavioral drift.",
        "description_layman": (
            "Keeps a separate 'clean' model to help ARIA tell the"
            " difference between its own mistakes and actual changes"
            " in your household routines."
        ),
        "description_technical": (
            "Enables a clean reference model trained without online"
            " updates. Used to distinguish meta-learner errors"
            " (primary diverges from reference = learner bug) from"
            " genuine behavioral drift (both diverge = real change)."
            " Adds ~50% training overhead."
        ),
        "category": "Reference Model",
    },
    {
        "key": "reference_model.comparison_window_days",
        "default_value": "7",
        "value_type": "number",
        "label": "Comparison Window (days)",
        "description": "Number of days of accuracy history to compare between primary and reference models.",
        "description_layman": (
            "How many days of prediction history ARIA compares"
            " between its main model and the reference model."
            " A longer window gives a more reliable comparison."
        ),
        "description_technical": (
            "Rolling window in days for comparing accuracy between"
            " primary and reference models. Range 3-30, default 7."
            " At 3, sensitive to short-term noise. At 30, slow to"
            " detect divergence but fewer false alerts. Accuracy"
            " is computed as rolling mean absolute error."
        ),
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
        "description_layman": (
            "How much worse ARIA's main model must perform compared"
            " to the reference model before ARIA flags a potential"
            " problem with its learning process."
        ),
        "description_technical": (
            "Minimum accuracy divergence (percentage points) between"
            " primary and reference models to trigger an alert."
            " Range 1.0-20.0, default 5.0. At 1.0, frequent false"
            " alerts from normal variance. At 20.0, only severe"
            " degradation triggers alerts."
        ),
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
        "description_layman": (
            "Whether ARIA automatically picks the most useful data"
            " points for its predictions. Turning this off uses all"
            " available data, which may slow things down."
        ),
        "description_technical": (
            "Enables automatic feature selection before model"
            " training. When true, features are ranked and pruned"
            " using the configured method. When false, all available"
            " features are passed to the model. Disabling may improve"
            " accuracy with small feature sets but degrades"
            " performance with 30+ features."
        ),
        "category": "Feature Selection",
    },
    {
        "key": "feature_selection.method",
        "default_value": "mrmr",
        "value_type": "select",
        "label": "Selection Method",
        "description": "Feature selection algorithm: mRMR, importance-based, or none.",
        "description_layman": (
            "Which method ARIA uses to choose the best data points."
            " 'mrmr' is the smartest option; 'importance' is simpler;"
            " 'none' turns feature selection off."
        ),
        "description_technical": (
            "Feature selection algorithm. 'mrmr' (minimum Redundancy"
            " Maximum Relevance) selects diverse, informative"
            " features. 'importance' uses LightGBM feature"
            " importance scores. 'none' disables selection. mRMR"
            " is O(n*k) where k=max_features; importance is O(1)"
            " post-training."
        ),
        "category": "Feature Selection",
        "options": "mrmr,importance,none",
    },
    {
        "key": "feature_selection.max_features",
        "default_value": "30",
        "value_type": "number",
        "label": "Max Features",
        "description": "Maximum number of features to retain after selection.",
        "description_layman": (
            "The maximum number of data points ARIA keeps after"
            " narrowing down what matters most. More features can"
            " capture more detail but slow down predictions."
        ),
        "description_technical": (
            "Maximum features retained after selection. Range 10-48,"
            " default 30. Features beyond this count are dropped"
            " regardless of score. At 10, model is fast but may"
            " miss relevant signals. At 48, nearly all features"
            " retained — selection has minimal effect."
        ),
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
        "description_layman": (
            "How often ARIA re-evaluates which data points are most"
            " useful. More frequent recomputation keeps ARIA current"
            " but adds processing time."
        ),
        "description_technical": (
            "Days between feature selection recomputation. Range 1-30,"
            " default 7. Feature rankings are cached and reused"
            " between recomputes. At 1, daily recomputation tracks"
            " fast-changing relevance. At 30, feature set is very"
            " stable but may lag behind behavioral shifts."
        ),
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
        "description_layman": (
            "Whether ARIA uses advanced analysis to explain why it"
            " made a prediction. When enabled, explanations are"
            " based on actual data, not just guesses."
        ),
        "description_technical": (
            "Enables SHAP (SHapley Additive exPlanations) feature"
            " attribution for grounding LLM narration. When true,"
            " SHAP values are computed per prediction and injected"
            " into the narration prompt. Adds ~200ms per explanation."
            " When false, narration uses heuristic feature importance."
        ),
        "category": "Narration",
    },
    {
        "key": "narration.top_features",
        "default_value": "5",
        "value_type": "number",
        "label": "Top Features",
        "description": "Number of top SHAP contributors to include in narration.",
        "description_layman": (
            "How many reasons ARIA includes when explaining a"
            " prediction. More reasons give a fuller picture but"
            " may be harder to read at a glance."
        ),
        "description_technical": (
            "Number of top SHAP contributors included in narration"
            " context. Range 3-10, default 5. Only the top-N"
            " features by absolute SHAP value are passed to the"
            " LLM. At 3, brief but may omit relevant factors. At"
            " 10, comprehensive but risks LLM prompt dilution."
        ),
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
        "description_layman": (
            "How ARIA figures out which factors drove a prediction."
            " 'shap' is the most accurate; 'importance' is faster;"
            " 'none' skips explanations entirely."
        ),
        "description_technical": (
            "Feature attribution source for narration grounding."
            " 'shap' computes per-prediction SHAP values (most"
            " faithful, slowest). 'importance' uses global LightGBM"
            " feature importance (fast, less granular). 'none'"
            " disables grounding — LLM narrates without attribution"
            " data, risking confabulation."
        ),
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
        "description_layman": (
            "Whether ARIA can quickly update its model when it"
            " detects your home's patterns have changed, instead"
            " of doing a full retraining from scratch."
        ),
        "description_technical": (
            "Enables incremental LightGBM (eGBDT) adaptation"
            " triggered by drift detection. When true, new boosting"
            " rounds are appended to the existing model. When false,"
            " drift always triggers a full retrain. Incremental"
            " updates are ~10x faster but accumulate trees over time."
        ),
        "category": "Incremental Training",
    },
    {
        "key": "incremental.boost_rounds",
        "default_value": "20",
        "value_type": "number",
        "label": "Boost Rounds",
        "description": "Number of boosting rounds to add during incremental training.",
        "description_layman": (
            "How much additional learning ARIA does during a quick"
            " update. More rounds mean a more thorough update but"
            " take longer to complete."
        ),
        "description_technical": (
            "Number of new LightGBM boosting rounds added per"
            " incremental training cycle. Range 5-100, default 20."
            " Each round adds one decision tree. At 5, minimal"
            " adaptation per cycle. At 100, significant model"
            " change per cycle — may overfit to recent drift window."
        ),
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
        "description_layman": (
            "The maximum complexity ARIA's model can reach before"
            " it resets and learns from scratch. This prevents the"
            " model from becoming too large and slow."
        ),
        "description_technical": (
            "Maximum accumulated tree count in the LightGBM model"
            " before forcing a full retrain. Range 100-2000,"
            " default 500. Once total trees exceed this, the next"
            " drift event triggers full retraining instead of"
            " incremental. At 100, frequent full retrains. At 2000,"
            " model may grow very large (~50 MB)."
        ),
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
        "description_layman": (
            "How many days of recent data ARIA uses when doing a"
            " quick model update. A wider window captures more"
            " context but includes older patterns."
        ),
        "description_technical": (
            "Rolling window in days for incremental training data."
            " Range 7-30, default 14. Only events within this"
            " window are used for incremental boosting rounds."
            " At 7, adapts to very recent data only. At 30,"
            " includes older patterns that may dilute the drift"
            " signal being adapted to."
        ),
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
        "description_layman": (
            "The network address of your MQTT message broker,"
            " usually running on your Home Assistant device."
            " Required for camera-based presence detection."
        ),
        "description_technical": (
            "Hostname or IP for the MQTT broker (typically Mosquitto"
            " addon on HAOS). Empty string disables MQTT connection."
            " Supports hostname or IPv4. Used by the presence module"
            " to subscribe to Frigate person/face detection events."
        ),
        "category": "Presence Tracking",
    },
    {
        "key": "presence.mqtt_port",
        "default_value": "1883",
        "value_type": "number",
        "label": "MQTT Port",
        "description": "Port of the MQTT broker.",
        "description_layman": (
            "The network port your MQTT broker listens on."
            " The default of 1883 is standard for most setups."
            " Only change this if your broker uses a custom port."
        ),
        "description_technical": (
            "TCP port for MQTT broker connection. Range 1-65535,"
            " default 1883 (standard MQTT). Use 8883 for TLS."
            " The presence module connects on startup and reconnects"
            " automatically on disconnect."
        ),
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
        "description_layman": (
            "The username for connecting to your MQTT broker."
            " Leave empty if your broker does not require"
            " authentication."
        ),
        "description_technical": (
            "MQTT broker authentication username. Empty string"
            " skips authentication. When set, must be paired with"
            " mqtt_password. Passed to aiomqtt client on connect."
            " For HAOS Mosquitto addon, use an HA local user."
        ),
        "category": "Presence Tracking",
    },
    {
        "key": "presence.mqtt_password",
        "default_value": "",
        "value_type": "string",
        "label": "MQTT Password",
        "description": "Password for MQTT broker authentication.",
        "description_layman": (
            "The password for connecting to your MQTT broker."
            " Leave empty if your broker does not require"
            " authentication."
        ),
        "description_technical": (
            "MQTT broker authentication password. Empty string"
            " skips authentication. Stored in config DB — not"
            " encrypted at rest. For security-sensitive setups,"
            " prefer environment variable injection over this field."
        ),
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
        "description_layman": (
            "Maps your cameras to rooms so ARIA knows which room"
            " a person was detected in. Format: camera_name:room_name"
            " separated by commas. Leave empty to use camera names"
            " as room names."
        ),
        "description_technical": (
            "Comma-separated camera:room mapping pairs for Frigate"
            " MQTT events. Empty string defaults to using the Frigate"
            " camera name as the room name. Each pair maps a Frigate"
            " camera_id to an ARIA room_id. Unknown cameras fall"
            " back to camera name as room."
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
        "description_layman": (
            "How long ARIA remembers a device after it disappears"
            " from Home Assistant. During this grace period, the"
            " device can come back without losing its history."
            " Set to 0 to archive devices immediately."
        ),
        "description_technical": (
            "Stale TTL in hours for discovery lifecycle. Range 0-720,"
            " default 72. Entities missing from HA discovery are"
            " marked stale but remain in active consumers. After"
            " TTL expires, they transition to archived state. At 0,"
            " immediate archival (no grace period). Rediscovered"
            " entities auto-promote from stale/archived."
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
        "description_layman": (
            "Whether ARIA spreads your corrections to similar"
            " situations. When you fix a prediction, ARIA can apply"
            " that fix to other nearby time periods automatically."
        ),
        "description_technical": (
            "Enables adaptive correction propagation using Slivkins"
            " contextual zooming with prioritized experience replay."
            " When true, user corrections are propagated to similar"
            " temporal contexts. When false, corrections only affect"
            " the exact timestamp corrected."
        ),
        "category": "Correction Propagation",
    },
    {
        "key": "propagation.base_radius_hours",
        "default_value": "1.0",
        "value_type": "number",
        "label": "Base Radius (hours)",
        "description": "Base temporal radius for correction propagation before adaptive narrowing.",
        "description_layman": (
            "How far in time ARIA spreads your corrections. A"
            " correction at 8 PM with a 1-hour radius also affects"
            " predictions between 7 PM and 9 PM."
        ),
        "description_technical": (
            "Base temporal radius in hours for Slivkins zooming"
            " correction propagation. Range 0.25-4.0, default 1.0."
            " Radius narrows adaptively as observations accumulate."
            " At 0.25, very localized corrections. At 4.0, broad"
            " propagation that may overwrite unrelated contexts."
        ),
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
        "description_layman": (
            "How many corrections ARIA needs in a time period before"
            " it starts being more precise about where to apply them."
            " More observations mean more targeted corrections."
        ),
        "description_technical": (
            "Minimum observation count in a Slivkins context cell"
            " before adaptive radius narrowing begins. Range 5-50,"
            " default 10. Below this threshold, the full base_radius"
            " is used. At 5, narrowing starts early (may be noisy)."
            " At 50, requires many corrections before precision."
        ),
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
        "description_layman": (
            "Controls how broadly ARIA considers situations to be"
            " 'similar' when spreading corrections. Higher values"
            " mean corrections spread to more loosely related times."
        ),
        "description_technical": (
            "Gaussian kernel bandwidth (sigma) for computing context"
            " similarity weights during propagation. Range 0.1-2.0,"
            " default 0.5. At 0.1, very narrow — only near-identical"
            " contexts receive weight. At 2.0, wide kernel — distant"
            " contexts still influence, risking over-smoothing."
        ),
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
        "description_layman": (
            "How many past corrections ARIA remembers for reuse."
            " A larger buffer helps ARIA learn from more history"
            " but uses more memory."
        ),
        "description_technical": (
            "Maximum entries in the prioritized experience replay"
            " buffer. Range 50-1000, default 200. Oldest entries are"
            " evicted when full. Entries are sampled proportionally"
            " to priority (TD error). At 50, limited replay"
            " diversity. At 1000, ~4 KB memory per entry."
        ),
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
        "description_layman": (
            "How much ARIA prioritizes its biggest mistakes when"
            " replaying past corrections. Higher values focus more"
            " on the corrections that mattered most."
        ),
        "description_technical": (
            "Prioritization exponent (alpha) for replay buffer"
            " sampling. Range 0.0-1.0, default 0.6. P(i) ~"
            " priority_i^alpha. At 0.0, uniform random sampling"
            " (no prioritization). At 1.0, fully prioritized —"
            " high-error experiences dominate, risking overfitting"
            " to outliers."
        ),
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
        "description_layman": (
            "How many similar time periods ARIA updates when you"
            " make one correction. Higher values spread each fix"
            " further but may affect unrelated predictions."
        ),
        "description_technical": (
            "Maximum contexts receiving propagation from a single"
            " correction event. Range 1-20, default 5. Contexts"
            " are ranked by kernel similarity; top-N are updated."
            " At 1, no propagation beyond the correction itself."
            " At 20, wide propagation — useful for highly periodic"
            " behaviors."
        ),
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
        "description_layman": (
            "Override which complexity level of machine learning ARIA"
            " uses. 'auto' detects your hardware and picks the best"
            " tier. Higher tiers use more advanced but heavier models."
        ),
        "description_technical": (
            "Override for auto-detected ML hardware tier. Options:"
            " auto (detect), 1 (basic stats), 2 (LightGBM), 3"
            " (LightGBM + Optuna + online), 4 (attention + transfer)."
            " Auto detection checks available RAM and CPU cores."
            " Overriding to a tier above hardware capacity causes"
            " OOM or timeout failures."
        ),
        "category": "ml",
        "options": "auto,1,2,3,4",
    },
    {
        "key": "ml.fallback_ttl_days",
        "default_value": "7",
        "value_type": "number",
        "label": "Fallback TTL (days)",
        "description": "Days before retrying a model that fell back to lower tier.",
        "description_layman": (
            "When a model fails and ARIA falls back to a simpler"
            " version, this controls how many days before ARIA tries"
            " the advanced version again."
        ),
        "description_technical": (
            "Days before retrying a model that fell back to a lower"
            " ML tier. Range 1-30, default 7. After a tier-3 model"
            " OOMs and falls back to tier-2, the tier-3 attempt is"
            " blocked for this many days. At 1, daily retry (may"
            " cause repeated OOMs). At 30, long wait before retry."
        ),
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
        "description_layman": (
            "The minimum usefulness a data point must have to stay"
            " in ARIA's model. Data points less useful than this"
            " threshold may be removed to keep the model lean."
        ),
        "description_technical": (
            "LightGBM feature importance threshold below which"
            " features become pruning candidates. Range 0.001-0.1,"
            " default 0.01. Features must fall below this for"
            " prune_cycles consecutive cycles to be pruned. At"
            " 0.001, almost nothing pruned. At 0.1, aggressive"
            " pruning — only top features survive."
        ),
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
        "description_layman": (
            "How many training cycles a data point must be rated"
            " unimportant before ARIA removes it. More cycles mean"
            " ARIA is more cautious about removing data."
        ),
        "description_technical": (
            "Consecutive training cycles a feature must remain below"
            " prune_threshold before auto-pruning. Range 1-10,"
            " default 3. Tier 3+ only. At 1, one bad cycle triggers"
            " prune (may remove transiently important features). At"
            " 10, very conservative pruning."
        ),
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
        "description_layman": (
            "How many different model configurations ARIA tries when"
            " optimizing its predictions. More trials can find better"
            " settings but take longer to complete."
        ),
        "description_technical": (
            "Optuna hyperparameter optimization trial count per"
            " training cycle. Range 5-100, default 20. Tier 3+ only."
            " Each trial trains a full LightGBM model with different"
            " hyperparameters. At 5, minimal tuning. At 100, thorough"
            " search but ~100x training time of a single fit."
        ),
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
        "description_layman": (
            "How thoroughly ARIA tests its model during training."
            " Higher values give more reliable accuracy estimates"
            " but take longer. 'auto' picks based on your hardware."
        ),
        "description_technical": (
            "Cross-validation fold count. Options: auto (tier-based:"
            " T1=1, T2=3, T3+=5), 1, 3, 5, 10. Higher folds give"
            " lower-variance accuracy estimates but multiply training"
            " time. At 1, no cross-validation (holdout only). At 10,"
            " 10x training time with diminishing returns."
        ),
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
        "description_layman": (
            "How much influence ARIA's fast-adapting model has versus"
            " its stable batch model. Higher values favor recent"
            " patterns; lower values favor long-term trends."
        ),
        "description_technical": (
            "Blend weight for online model in ensemble predictions."
            " Range 0.0-1.0, default 0.3. Tier 3+ only. Final"
            " prediction = (1-w)*batch + w*online. At 0.0, online"
            " model disabled. At 1.0, only online model used"
            " (ignores batch). Online model adapts faster but has"
            " higher variance."
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
        "description_layman": (
            "How many data points ARIA's fast-adapting model needs"
            " to see before it starts making predictions. Fewer"
            " means faster startup but less reliable initial guesses."
        ),
        "description_technical": (
            "Minimum observation count before the online model"
            " contributes to ensemble predictions. Range 1-50,"
            " default 5. Below this count, blend_weight is forced"
            " to 0 (batch-only). At 1, online model activates"
            " immediately (noisy). At 50, long cold-start delay."
        ),
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
        "description_layman": (
            "Whether ARIA automatically adjusts how much it trusts"
            " each of its prediction models based on recent accuracy."
            " Better-performing models get more influence."
        ),
        "description_technical": (
            "Enables automatic ensemble weight adjustment using"
            " rolling Mean Absolute Error (MAE). Tier 3+ only."
            " When true, model weights are recomputed each cycle"
            " inversely proportional to recent MAE. When false,"
            " static online_blend_weight is used throughout."
        ),
        "category": "ml",
    },
    {
        "key": "ml.weight_tuner_window_days",
        "default_value": "7",
        "value_type": "number",
        "label": "Weight Tuner Window (days)",
        "description": "Rolling window for MAE-based weight computation.",
        "description_layman": (
            "How many days of recent accuracy data ARIA uses when"
            " deciding which models to trust more. A longer window"
            " gives more stable weights."
        ),
        "description_technical": (
            "Rolling window in days for MAE-based ensemble weight"
            " computation. Range 1-30, default 7. Only prediction"
            " errors within this window are considered. At 1, weights"
            " fluctuate daily. At 30, very stable but slow to reflect"
            " recent model quality changes."
        ),
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
        "description_layman": (
            "The minimum hardware capability your system needs for"
            " ARIA to run pattern recognition. Lower tiers work on"
            " weaker hardware; higher tiers need more powerful systems."
        ),
        "description_technical": (
            "Minimum ML tier for pattern recognition activation."
            " Range 1-4, default 3. Below this tier, pattern"
            " recognition is disabled entirely. Tier 3 requires"
            " ~4 GB RAM and 4+ CPU cores. Setting to 1 enables"
            " pattern recognition on all hardware but may cause"
            " OOM on low-end devices."
        ),
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
        "description_layman": (
            "How many recent snapshots ARIA looks at together to"
            " identify trends like 'ramping up' or 'winding down.'"
            " A larger window detects longer patterns."
        ),
        "description_technical": (
            "Sliding window size in snapshots for trajectory"
            " classification. Range 3-24, default 6. Trajectories"
            " (stable, ramping, winding, erratic) are classified"
            " over this window. At 3, very short-term trends. At"
            " 24, captures full-day patterns but lags short changes."
        ),
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
        "description_layman": (
            "How many similar past patterns ARIA compares against"
            " when classifying the current trend. More comparisons"
            " give smoother results but take longer."
        ),
        "description_technical": (
            "k-NN neighbors for DTW (Dynamic Time Warping) sequence"
            " classifier. Range 1-10, default 3. Higher k smooths"
            " classification but increases compute: DTW is O(n*m)"
            " per comparison. At 1, nearest-neighbor only (noisy)."
            " At 10, very stable but slower classification."
        ),
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
        "description_layman": (
            "How many reasons ARIA lists when it detects something"
            " unusual. More reasons help you understand what"
            " triggered the anomaly alert."
        ),
        "description_technical": (
            "Top-N feature contributors reported per anomaly"
            " detection. Range 1-10, default 3. Features are"
            " ranked by their contribution to the anomaly score"
            " (reconstruction error or isolation path length)."
            " At 1, only the primary driver shown. At 10, full"
            " breakdown but may include noise."
        ),
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
        "description_layman": (
            "How much a metric must change within the observation"
            " window for ARIA to call it a trend (going up or down)"
            " versus stable. Higher values mean only big changes"
            " count as trends."
        ),
        "description_technical": (
            "Minimum relative change (0.0-1.0) in target metric"
            " across the sequence window to classify trajectory as"
            " ramping or winding. Range 0.05-0.50, default 0.20."
            " Below threshold, trajectory is 'stable'. At 0.05,"
            " even 5% change = trend (noisy). At 0.50, only large"
            " swings are classified as directional."
        ),
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
        "description_layman": (
            "How similar two devices must be before ARIA tries to"
            " share learning between them. Higher values require"
            " devices to be more alike before sharing knowledge."
        ),
        "description_technical": (
            "Minimum Jaccard similarity between entity feature sets"
            " for transfer learning candidacy. Range 0.4-1.0,"
            " default 0.6. Computed from structural features (domain,"
            " attributes, state patterns). At 0.4, loose matching —"
            " may transfer between unrelated entities. At 1.0,"
            " requires identical structure (practically no transfers)."
        ),
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
        "description_layman": (
            "How many days ARIA tests a transferred model in the"
            " background before using it for real predictions."
            " Longer testing periods reduce the risk of bad transfers."
        ),
        "description_technical": (
            "Minimum shadow testing duration in days before a"
            " transfer candidate can be promoted to active. Range"
            " 3-30, default 7. During shadow testing, the transfer"
            " model predicts in parallel but results are not used."
            " At 3, fast promotion but less validation. At 30,"
            " thorough but slow adoption."
        ),
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
        "description_layman": (
            "What accuracy a transferred model must achieve during"
            " testing before ARIA starts using it. Higher values"
            " require better performance before promotion."
        ),
        "description_technical": (
            "Minimum shadow prediction hit rate for promotion. Range"
            " 0.3-0.9, default 0.6. Hit rate = correct predictions /"
            " total predictions during shadow period. At 0.3, low"
            " bar (promotes mediocre transfers). At 0.9, strict —"
            " only highly accurate transfers promoted."
        ),
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
        "description_layman": (
            "If a transferred model's accuracy falls below this"
            " level during testing, ARIA rejects it entirely."
            " This prevents bad transfers from ever being used."
        ),
        "description_technical": (
            "Shadow hit rate threshold below which transfer"
            " candidates are permanently rejected. Range 0.1-0.5,"
            " default 0.3. Must be less than promotion_hit_rate."
            " Between reject and promote thresholds, candidates"
            " continue shadow testing. At 0.1, very lenient. At"
            " 0.5, rejects anything below coin-flip accuracy."
        ),
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
        "description_layman": (
            "Controls the capacity of ARIA's most advanced anomaly"
            " detection model. Higher values can detect subtler"
            " patterns but need more computing power. Tier 4 only."
        ),
        "description_technical": (
            "Hidden dimension for the attention-based autoencoder."
            " Range 8-128, default 32. Tier 4 only. Determines"
            " the bottleneck representation size. At 8, aggressive"
            " compression (may miss subtle patterns). At 128, high"
            " capacity but ~16x more parameters and risk of"
            " overfitting on small datasets."
        ),
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
        "description_layman": (
            "How many training passes ARIA's advanced anomaly model"
            " goes through. More epochs improve accuracy but take"
            " longer. Only used on high-end hardware (Tier 4)."
        ),
        "description_technical": (
            "Training epochs for the attention autoencoder. Range"
            " 5-100, default 20. Tier 4 only. Each epoch processes"
            " the full dataset once. At 5, likely underfitting. At"
            " 100, diminishing returns; ~5x longer training. Early"
            " stopping may terminate before reaching this limit."
        ),
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
        "description_layman": (
            "Whether ARIA keeps a detailed log of everything it"
            " does. Useful for troubleshooting, but can be turned"
            " off to save disk space."
        ),
        "description_technical": (
            "Master switch for the audit logging subsystem. When"
            " false, no audit events are recorded regardless of"
            " other audit settings. When true, events are buffered"
            " and flushed per buffer_size and flush_interval_ms."
            " Disabling mid-operation does not flush the buffer."
        ),
        "category": "Audit",
    },
    {
        "key": "audit.retention_days",
        "default_value": "90",
        "value_type": "number",
        "label": "Audit Retention (days)",
        "description": "Number of days to retain audit events before pruning.",
        "description_layman": (
            "How many days of audit history ARIA keeps before"
            " cleaning up old records. Longer retention uses more"
            " disk space but keeps a more complete history."
        ),
        "description_technical": (
            "Retention period for audit events before pruning. Range"
            " 7-365, default 90. Events older than this are deleted"
            " (or archived if archive_on_prune=true) during the"
            " maintenance cycle. At 365 days with heavy API traffic,"
            " audit table may reach 1+ GB."
        ),
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
        "description_layman": (
            "Whether ARIA records every request made to its API."
            " Helpful for seeing who or what is accessing ARIA's"
            " data, but generates many log entries."
        ),
        "description_technical": (
            "Records all inbound HTTP API requests in the audit log."
            " When true, each request generates an audit event with"
            " method, path, status code, and duration. High-traffic"
            " APIs (e.g. /api/cache polled every 5s) can generate"
            " ~17K events/day. Disable to reduce audit volume."
        ),
        "category": "Audit",
    },
    {
        "key": "audit.log_cache_writes",
        "default_value": "true",
        "value_type": "boolean",
        "label": "Log Cache Writes",
        "description": "Record cache write operations in the audit log.",
        "description_layman": (
            "Whether ARIA logs every time it saves data to its"
            " internal database. Useful for debugging but creates"
            " a lot of log entries."
        ),
        "description_technical": (
            "Records cache write operations (set_cache, upsert)"
            " in the audit log. When true, each write generates"
            " an audit event with key, category, and value size."
            " Can produce high volume during snapshot ingestion."
            " Disable if audit DB growth is a concern."
        ),
        "category": "Audit",
    },
    {
        "key": "audit.buffer_size",
        "default_value": "10000",
        "value_type": "number",
        "label": "Write Buffer Size",
        "description": "Maximum number of audit events to hold in memory before flushing.",
        "description_layman": (
            "How many audit entries ARIA holds in memory before"
            " writing them to disk. A larger buffer means fewer"
            " disk writes but more memory use."
        ),
        "description_technical": (
            "In-memory audit event buffer capacity before force"
            " flush. Range 1000-100000, default 10000. Buffer is"
            " also flushed on flush_interval_ms timer. At 1000,"
            " frequent small writes. At 100000, ~4 MB memory but"
            " risk of losing events on crash."
        ),
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
        "description_layman": (
            "How often ARIA writes its buffered audit data to disk,"
            " in milliseconds. More frequent flushes reduce data"
            " loss risk but increase disk activity."
        ),
        "description_technical": (
            "Timer interval in milliseconds for automatic buffer"
            " flush. Range 100-5000, default 500. Flush occurs on"
            " this timer OR when buffer_size is reached, whichever"
            " comes first. At 100, near-real-time persistence. At"
            " 5000, up to 5 seconds of audit data at risk on crash."
        ),
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
        "description_layman": (
            "Whether ARIA sends you an alert when it detects too"
            " many errors in a short time. Helps you catch problems"
            " early before they become serious."
        ),
        "description_technical": (
            "Enables error rate alerting in the audit subsystem."
            " When true, errors are counted in a rolling window"
            " (alert_window_minutes) and an alert fires when"
            " alert_threshold is exceeded. Alerts are sent via"
            " the configured notification channel (Telegram)."
        ),
        "category": "Audit",
    },
    {
        "key": "audit.alert_threshold",
        "default_value": "10",
        "value_type": "number",
        "label": "Alert Threshold",
        "description": "Number of errors within the alert window that triggers an alert.",
        "description_layman": (
            "How many errors must occur in a short time before ARIA"
            " sends an alert. A higher number means only repeated"
            " problems trigger alerts."
        ),
        "description_technical": (
            "Error count threshold within alert_window_minutes to"
            " trigger an alert. Range 1-100, default 10. Uses a"
            " sliding window counter. At 1, any single error"
            " triggers alert (very noisy). At 100, only sustained"
            " error bursts trigger notification."
        ),
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
        "description_layman": (
            "The time window in minutes that ARIA looks at when"
            " counting errors. A longer window catches slow-building"
            " problems; a shorter window catches sudden spikes."
        ),
        "description_technical": (
            "Rolling window in minutes for error counting. Range"
            " 1-60, default 5. Errors older than this window are"
            " not counted toward the threshold. At 1, only very"
            " recent bursts trigger alerts. At 60, errors spread"
            " over an hour can accumulate to trigger."
        ),
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
        "description_layman": (
            "Whether ARIA saves old audit records to a file before"
            " deleting them. This lets you keep a permanent history"
            " without the database growing forever."
        ),
        "description_technical": (
            "Exports audit events to JSONL archive files before"
            " pruning from the database. When true, events are"
            " written to timestamped .jsonl files in the audit"
            " directory. When false, pruned events are permanently"
            " deleted. Archive files are not auto-cleaned."
        ),
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
        "description_layman": (
            "Which types of sensors ARIA uses to determine if"
            " someone is in a room. You can disable specific signal"
            " types by removing them from this list."
        ),
        "description_technical": (
            "Comma-separated signal types for presence detection"
            " fusion. Each type must match a SENSOR_CONFIG key."
            " Removing a signal type disables that input channel"
            " entirely. Order does not matter. Unknown signal types"
            " are silently ignored."
        ),
        "category": "Presence",
    },
    {
        "key": "activity.enabled_domains",
        "default_value": "light,switch,binary_sensor,media_player,climate,cover",
        "value_type": "string",
        "label": "Enabled Domains",
        "description": "Comma-separated list of entity domains tracked for activity monitoring.",
        "description_layman": (
            "Which types of devices ARIA monitors for activity"
            " patterns. Only devices in these categories are tracked."
            " Add or remove categories to customize what ARIA watches."
        ),
        "description_technical": (
            "Comma-separated HA entity domains for activity"
            " monitoring. Default includes behavioral domains"
            " (light, switch, binary_sensor, etc.). Domain filtering"
            " is applied on WebSocket state_changed events. Adding"
            " high-frequency domains (sensor) significantly"
            " increases event volume."
        ),
        "category": "Activity Monitor",
    },
    {
        "key": "anomaly.enabled_entities",
        "default_value": "light,binary_sensor,climate,media_player,switch",
        "value_type": "string",
        "label": "Enabled Entity Domains",
        "description": "Comma-separated entity IDs for anomaly detection, or 'all' for all discovered entities.",
        "description_layman": (
            "Which types of devices ARIA checks for unusual behavior."
            " Use 'all' to monitor everything, or list specific"
            " categories to focus anomaly detection."
        ),
        "description_technical": (
            "Comma-separated entity domains (or 'all') for anomaly"
            " detection input. Default targets behavioral domains."
            " When 'all', uses all discovered entities — may be"
            " computationally expensive with 3000+ entities."
            " Filtered at feature extraction time, not subscription."
        ),
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
        "description_layman": (
            "Which types of devices ARIA makes shadow predictions"
            " for. Use 'all' to predict everything predictable, or"
            " list specific categories to limit predictions."
        ),
        "description_technical": (
            "Comma-separated capability names for shadow prediction"
            " scope. 'all' enables prediction for all capabilities"
            " with can_predict=true. Default targets common"
            " behavioral domains. Capabilities not in this list"
            " are excluded from shadow prediction cycles."
        ),
        "category": "Shadow Mode",
    },
    {
        "key": "discovery.domain_filter",
        "default_value": "light,switch,binary_sensor,sensor,climate,cover,media_player,fan,lock",
        "value_type": "string",
        "label": "Domain Filter",
        "description": "Comma-separated entity domains to include in capability discovery.",
        "description_layman": (
            "Which types of devices ARIA discovers and tracks from"
            " Home Assistant. Only devices in these categories will"
            " be found during discovery scans."
        ),
        "description_technical": (
            "Comma-separated HA domains included in capability"
            " discovery. Entities outside these domains are ignored"
            " during discovery refresh. Default covers common"
            " controllable/observable domains. Adding 'sensor'"
            " dramatically increases discovered entity count"
            " (often 1000+ entities)."
        ),
        "category": "Discovery",
    },
    # ── Phase 3: Pattern Engine ────────────────────────────────────────
    {
        "key": "patterns.analysis_interval",
        "default_value": "7200",
        "value_type": "number",
        "label": "Pattern Analysis Interval (s)",
        "description": "How often ARIA looks for new patterns.",
        "description_layman": (
            "How often ARIA analyzes your home activity to find"
            " repeating patterns. Lower values mean faster learning"
            " but more CPU usage."
        ),
        "description_technical": (
            "Periodic scheduling interval in seconds for the pattern"
            " detection engine. Range 1800-86400, default 7200 (2h)."
            " Each run queries EventStore for the last 7 days."
        ),
        "category": "Pattern Engine",
        "min_value": 1800,
        "max_value": 86400,
        "step": 600,
    },
    {
        "key": "patterns.max_areas",
        "default_value": "20",
        "value_type": "number",
        "label": "Max Areas to Analyze",
        "description": "Maximum rooms to analyze at once for patterns.",
        "description_layman": (
            "The maximum number of rooms ARIA analyzes for patterns."
            " Higher values cover more of your home but take longer."
        ),
        "description_technical": (
            "Top-N most active areas selected by event count for"
            " pattern detection. Range 5-50, default 20. Areas below"
            " this cutoff are skipped to stay within memory budget."
        ),
        "category": "Pattern Engine",
        "min_value": 5,
        "max_value": 50,
        "step": 1,
    },
    {
        "key": "patterns.memory_budget_mb",
        "default_value": "512",
        "value_type": "number",
        "label": "Memory Budget (MB)",
        "description": "Memory limit for pattern analysis.",
        "description_layman": (
            "Maximum memory ARIA can use while analyzing patterns. Higher values allow deeper analysis of larger homes."
        ),
        "description_technical": (
            "Hard ceiling in MB for pattern detection working set."
            " Range 128-2048, default 512. If exceeded, analysis"
            " switches to sampling mode for remaining areas."
        ),
        "category": "Pattern Engine",
        "min_value": 128,
        "max_value": 2048,
        "step": 64,
    },
    {
        "key": "patterns.min_events",
        "default_value": "500",
        "value_type": "number",
        "label": "Minimum Events for Analysis",
        "description": "Minimum events before pattern analysis starts.",
        "description_layman": (
            "ARIA waits until it has seen this many events before"
            " trying to find patterns. Too few events produce"
            " unreliable results."
        ),
        "description_technical": (
            "Minimum total EventStore events before pattern engine"
            " runs. Range 100-5000, default 500. Below this count,"
            " engine returns empty results."
        ),
        "category": "Pattern Engine",
        "min_value": 100,
        "max_value": 5000,
        "step": 100,
    },
    {
        "key": "patterns.min_days",
        "default_value": "7",
        "value_type": "number",
        "label": "Minimum Days for Analysis",
        "description": "Minimum days of data before pattern analysis.",
        "description_layman": (
            "ARIA waits this many days after installation before"
            " looking for patterns. This ensures enough data for"
            " reliable detection."
        ),
        "description_technical": (
            "Minimum distinct days in EventStore before pattern detection activates. Range 3-30, default 7."
        ),
        "category": "Pattern Engine",
        "min_value": 3,
        "max_value": 30,
        "step": 1,
    },
    # ── Phase 3: Gap Analyzer ──────────────────────────────────────────
    {
        "key": "gap.analysis_interval",
        "default_value": "14400",
        "value_type": "number",
        "label": "Gap Analysis Interval (s)",
        "description": "How often ARIA checks for things you do manually.",
        "description_layman": (
            "How often ARIA looks for actions you repeat manually"
            " that could be automated. Less frequent than pattern"
            " analysis since gaps change slowly."
        ),
        "description_technical": (
            "Periodic scheduling interval for the gap analyzer."
            " Range 3600-86400, default 14400 (4h). Queries manual-only"
            " events (context_parent_id IS NULL)."
        ),
        "category": "Gap Analyzer",
        "min_value": 3600,
        "max_value": 86400,
        "step": 1800,
    },
    {
        "key": "gap.min_occurrences",
        "default_value": "15",
        "value_type": "number",
        "label": "Min Occurrences",
        "description": "Times you must do something before ARIA suggests automating.",
        "description_layman": (
            "You need to repeat an action this many times before"
            " ARIA suggests turning it into an automation. Higher"
            " values mean fewer but more reliable suggestions."
        ),
        "description_technical": (
            "Minimum frequency count for a manual action sequence before it appears as a gap. Range 5-100, default 15."
        ),
        "category": "Gap Analyzer",
        "min_value": 5,
        "max_value": 100,
        "step": 1,
    },
    {
        "key": "gap.min_consistency",
        "default_value": "0.6",
        "value_type": "number",
        "label": "Min Consistency",
        "description": "How consistent the action must be.",
        "description_layman": (
            "How regularly you need to repeat an action. 0.6 means"
            " you do it at least 60% of eligible days. Lower values"
            " catch more patterns but may suggest less reliable ones."
        ),
        "description_technical": (
            "Minimum consistency ratio (occurrences / eligible days) for gap detection. Range 0.3-1.0, default 0.6."
        ),
        "category": "Gap Analyzer",
        "min_value": 0.3,
        "max_value": 1.0,
        "step": 0.05,
    },
    {
        "key": "gap.min_days",
        "default_value": "14",
        "value_type": "number",
        "label": "Min Days for Gap Analysis",
        "description": "Minimum days of data before gap analysis.",
        "description_layman": (
            "ARIA needs at least this many days of data before it starts suggesting automations for manual actions."
        ),
        "description_technical": (
            "Minimum distinct days in EventStore before gap analyzer activates. Range 7-60, default 14."
        ),
        "category": "Gap Analyzer",
        "min_value": 7,
        "max_value": 60,
        "step": 1,
    },
    {
        "key": "gap.window_minutes",
        "default_value": "10",
        "value_type": "number",
        "label": "Gap Pairing Window (min)",
        "description": "Time window for pairing manual actions.",
        "description_layman": (
            "If you do two things within this many minutes, ARIA considers them related."
            " For example, turning on the kitchen light then the coffee maker within 10 minutes."
        ),
        "description_technical": (
            "Maximum time gap between consecutive manual events to be considered part of the same"
            " behavioral sequence. Range 1-60 minutes, default 10."
        ),
        "category": "Gap Analyzer",
        "min_value": 1,
        "max_value": 60,
        "step": 1,
    },
    {
        "key": "gap.max_chain_length",
        "default_value": "5",
        "value_type": "number",
        "label": "Max Chain Length",
        "description": "Maximum entities in a detected manual action chain.",
        "description_layman": (
            "The longest sequence of manual actions ARIA will track."
            " '3' means ARIA looks for patterns like 'light → coffee → blinds' but not longer."
        ),
        "description_technical": (
            "Maximum entities in a single detected manual sequence."
            " Longer chains are pruned to this length. Range 2-10, default 5."
        ),
        "category": "Gap Analyzer",
        "min_value": 2,
        "max_value": 10,
        "step": 1,
    },
    # ── Phase 3: Automation Generator ──────────────────────────────────
    {
        "key": "automation.max_suggestions_per_cycle",
        "default_value": "10",
        "value_type": "number",
        "label": "Max Suggestions per Cycle",
        "description": "Maximum new suggestions per analysis cycle.",
        "description_layman": (
            "How many new automation suggestions ARIA creates each"
            " time it analyzes your home. Keeps the list manageable."
        ),
        "description_technical": (
            "Top-N cap on suggestions produced per generator cycle."
            " Range 1-50, default 10. Remaining detections are queued"
            " for the next cycle."
        ),
        "category": "Automation Generator",
        "min_value": 1,
        "max_value": 50,
        "step": 1,
    },
    {
        "key": "automation.min_combined_score",
        "default_value": "0.6",
        "value_type": "number",
        "label": "Min Confidence Score",
        "description": "Minimum confidence before ARIA suggests anything.",
        "description_layman": (
            "How confident ARIA must be before suggesting an"
            " automation. Higher values mean fewer but more reliable"
            " suggestions."
        ),
        "description_technical": (
            "Combined score floor: pattern_confidence * 0.5 +"
            " gap_consistency * 0.3 + recency_weight * 0.2."
            " Range 0.3-0.9, default 0.6."
        ),
        "category": "Automation Generator",
        "min_value": 0.3,
        "max_value": 0.9,
        "step": 0.05,
    },
    {
        "key": "automation.min_observations",
        "default_value": "10",
        "value_type": "number",
        "label": "Min Observations",
        "description": "Minimum observations before suggesting.",
        "description_layman": ("ARIA must observe a pattern this many times before suggesting an automation."),
        "description_technical": (
            "Minimum observation_count on a DetectionResult before"
            " it enters the generation pipeline. Range 3-50, default 10."
        ),
        "category": "Automation Generator",
        "min_value": 3,
        "max_value": 50,
        "step": 1,
    },
    {
        "key": "automation.rejection_penalty",
        "default_value": "0.8",
        "value_type": "number",
        "label": "Rejection Penalty",
        "description": "How much confidence drops per rejection.",
        "description_layman": (
            "Each time you reject a suggestion, ARIA reduces its"
            " confidence in that pattern by this much. At 0.8, each"
            " rejection cuts confidence by 20%."
        ),
        "description_technical": (
            "Multiplicative penalty per user rejection on the same"
            " source pattern. Range 0.5-0.95, default 0.8."
            " After max_rejections, pattern is suppressed entirely."
        ),
        "category": "Automation Generator",
        "min_value": 0.5,
        "max_value": 0.95,
        "step": 0.05,
    },
    {
        "key": "automation.max_rejections",
        "default_value": "3",
        "value_type": "number",
        "label": "Max Rejections",
        "description": "Stop suggesting after this many rejections.",
        "description_layman": (
            "After you reject the same suggestion this many times, ARIA stops suggesting it entirely."
        ),
        "description_technical": (
            "Hard cap on rejection count per source pattern."
            " Range 1-10, default 3. Once reached, pattern is"
            " permanently suppressed from generation."
        ),
        "category": "Automation Generator",
        "min_value": 1,
        "max_value": 10,
        "step": 1,
    },
    # ── Phase 3: Shadow Comparison ─────────────────────────────────────
    {
        "key": "shadow.sync_interval",
        "default_value": "1800",
        "value_type": "number",
        "label": "Shadow Sync Interval (s)",
        "description": "How often ARIA checks your existing automations.",
        "description_layman": (
            "How often ARIA checks your existing Home Assistant automations to avoid suggesting duplicates."
        ),
        "description_technical": (
            "Periodic interval for GET /api/config/automation/config."
            " Range 600-86400, default 1800 (30min). Uses incremental"
            " hash-based sync to minimize processing."
        ),
        "category": "Shadow Comparison",
        "min_value": 600,
        "max_value": 86400,
        "step": 300,
    },
    {
        "key": "shadow.duplicate_threshold",
        "default_value": "0.8",
        "value_type": "number",
        "label": "Duplicate Threshold",
        "description": "How similar before ARIA considers it a duplicate.",
        "description_layman": (
            "How closely an ARIA suggestion must match an existing"
            " automation to be considered a duplicate. Higher values"
            " mean stricter matching."
        ),
        "description_technical": (
            "Jaccard similarity threshold for trigger+target entity"
            " overlap. Range 0.5-1.0, default 0.8. At 1.0, only"
            " exact matches are suppressed."
        ),
        "category": "Shadow Comparison",
        "min_value": 0.5,
        "max_value": 1.0,
        "step": 0.05,
    },
    # ── Phase 3: LLM Refinement ────────────────────────────────────────
    {
        "key": "llm.automation_model",
        "default_value": "qwen2.5-coder:14b",
        "value_type": "string",
        "label": "Automation LLM Model",
        "description": "AI model that polishes automation names/descriptions.",
        "description_layman": (
            "The AI model ARIA uses to write friendly names and descriptions for suggested automations."
        ),
        "description_technical": (
            "Ollama model name for automation alias/description"
            " refinement. Submitted through ollama-queue if"
            " llm.queue_enabled. Timeout controlled by"
            " llm.automation_timeout."
        ),
        "category": "LLM Refinement",
    },
    {
        "key": "llm.automation_timeout",
        "default_value": "30",
        "value_type": "number",
        "label": "LLM Timeout (s)",
        "description": "Seconds to wait for AI polish before using template.",
        "description_layman": (
            "How long ARIA waits for the AI to polish automation"
            " names. If it takes too long, ARIA uses the template"
            " version instead."
        ),
        "description_technical": (
            "aiohttp timeout for Ollama inference call. Range 10-120,"
            " default 30. On timeout, template output ships as-is."
        ),
        "category": "LLM Refinement",
        "min_value": 10,
        "max_value": 120,
        "step": 5,
    },
    # ── Phase 3: Data Filtering ────────────────────────────────────────
    {
        "key": "filter.ignored_states",
        "default_value": "unavailable,unknown",
        "value_type": "string",
        "label": "Ignored States",
        "description": "State values excluded from analysis.",
        "description_layman": (
            "Device states that ARIA ignores completely. Usually"
            " 'unavailable' and 'unknown' since they represent"
            " communication errors, not real activity."
        ),
        "description_technical": (
            "Comma-separated state values filtered out during"
            " normalization. Both old_state and new_state checked."
            " Events matching either direction are removed."
        ),
        "category": "Data Filtering",
    },
    {
        "key": "filter.min_availability_pct",
        "default_value": "80",
        "value_type": "number",
        "label": "Min Availability (%)",
        "description": "Devices below this % are ignored as unreliable.",
        "description_layman": (
            "Devices that are unavailable more than this percentage"
            " of the time are excluded from analysis. Keeps"
            " suggestions reliable."
        ),
        "description_technical": (
            "Entity health floor. Entities with availability below"
            " this threshold get health_grade='unreliable' and are"
            " excluded from detection. Range 50-99, default 80."
        ),
        "category": "Data Filtering",
        "min_value": 50,
        "max_value": 99,
        "step": 1,
    },
    {
        "key": "filter.exclude_entities",
        "default_value": "",
        "value_type": "string",
        "label": "Exclude Entities",
        "description": "Specific devices to exclude from analysis.",
        "description_layman": (
            "Specific devices you never want ARIA to analyze or"
            " suggest automations for. Enter entity IDs separated"
            " by commas."
        ),
        "description_technical": (
            "Comma-separated entity_ids excluded from normalizer. Exact match. Empty default means no exclusions."
        ),
        "category": "Data Filtering",
    },
    {
        "key": "filter.exclude_areas",
        "default_value": "",
        "value_type": "string",
        "label": "Exclude Areas",
        "description": "Rooms to exclude from suggestions.",
        "description_layman": (
            "Rooms you never want ARIA to analyze or create automations for. Enter area names separated by commas."
        ),
        "description_technical": (
            "Comma-separated HA area_ids excluded from normalizer. Events in these areas are filtered before detection."
        ),
        "category": "Data Filtering",
    },
    {
        "key": "filter.exclude_domains",
        "default_value": (
            "update,button,number,input_number,input_boolean,"
            "input_select,input_text,persistent_notification,"
            "scene,script,automation"
        ),
        "value_type": "string",
        "label": "Exclude Domains",
        "description": "Device types ARIA ignores.",
        "description_layman": (
            "Types of Home Assistant entities ARIA ignores. Usually"
            " includes helper entities and system entities that"
            " don't represent real physical actions."
        ),
        "description_technical": (
            "Comma-separated HA domains excluded from normalizer."
            " If filter.include_domains is set, this list is ignored"
            " (whitelist takes precedence)."
        ),
        "category": "Data Filtering",
    },
    {
        "key": "filter.include_domains",
        "default_value": "",
        "value_type": "string",
        "label": "Include Domains (whitelist)",
        "description": "If set, ONLY these types are analyzed.",
        "description_layman": (
            "If you only want ARIA to look at specific device types,"
            " list them here. Leave empty to use the exclude list"
            " instead."
        ),
        "description_technical": (
            "Comma-separated whitelist. If non-empty, overrides"
            " filter.exclude_domains — only these domains pass"
            " normalization. Empty default means blacklist mode."
        ),
        "category": "Data Filtering",
    },
    {
        "key": "filter.exclude_entity_patterns",
        "default_value": "*_battery,*_signal_strength,*_linkquality,*_firmware",
        "value_type": "string",
        "label": "Exclude Entity Patterns",
        "description": "Name patterns to exclude from analysis.",
        "description_layman": (
            "Entity name patterns ARIA ignores. Uses * as wildcard."
            " Default excludes battery, signal, and firmware entities"
            " which are noisy and not automatable."
        ),
        "description_technical": (
            "Comma-separated fnmatch glob patterns applied to"
            " entity_id. Matched entities are excluded during"
            " normalization."
        ),
        "category": "Data Filtering",
    },
    {
        "key": "filter.flaky_weight",
        "default_value": "0.5",
        "value_type": "number",
        "label": "Flaky Entity Weight",
        "description": "Confidence multiplier for unreliable devices.",
        "description_layman": (
            "How much to reduce confidence for devices that are"
            " sometimes unavailable. At 0.5, flaky devices contribute"
            " half the normal confidence."
        ),
        "description_technical": (
            "Multiplicative weight applied to confidence scores"
            " for entities with health_grade='flaky'. Range 0.1-1.0,"
            " default 0.5. Unreliable entities are excluded entirely."
        ),
        "category": "Data Filtering",
        "min_value": 0.1,
        "max_value": 1.0,
        "step": 0.1,
    },
    # ── Phase 3: Calendar ──────────────────────────────────────────────
    {
        "key": "calendar.enabled",
        "default_value": "true",
        "value_type": "boolean",
        "label": "Calendar Integration",
        "description": "Use your calendar to improve automation accuracy.",
        "description_layman": (
            "When enabled, ARIA uses your calendar to understand"
            " holidays, vacations, and work-from-home days so it"
            " can make better automation suggestions."
        ),
        "description_technical": (
            "Enables calendar-aware day classification. When disabled,"
            " all weekdays are 'workday' and weekends are 'weekend'."
            " No holiday/vacation/WFH detection."
        ),
        "category": "Calendar",
    },
    {
        "key": "calendar.holiday_keywords",
        "default_value": "holiday,vacation,PTO,trip,out of office,off",
        "value_type": "string",
        "label": "Holiday Keywords",
        "description": "Words that mean you're not working.",
        "description_layman": (
            "Calendar event titles containing these words are treated as holidays or days off. Separate with commas."
        ),
        "description_technical": (
            "Case-insensitive substring match against calendar event"
            " summaries. Matched days get day_type='holiday' or"
            " 'vacation' depending on duration."
        ),
        "category": "Calendar",
    },
    {
        "key": "calendar.wfh_keywords",
        "default_value": "WFH,remote,work from home",
        "value_type": "string",
        "label": "WFH Keywords",
        "description": "Words that mean you're working from home.",
        "description_layman": (
            "Calendar event titles containing these words are treated"
            " as work-from-home days, which may have different"
            " automation patterns."
        ),
        "description_technical": (
            "Case-insensitive substring match. Days classified as"
            " 'wfh' get separate pattern analysis if >5 days in"
            " window, else merge with workday."
        ),
        "category": "Calendar",
    },
    {
        "key": "calendar.source",
        "default_value": "google",
        "value_type": "select",
        "label": "Calendar Source",
        "description": "Which calendar to check for events.",
        "description_layman": (
            "Where ARIA reads your calendar from. Google uses the gog"
            " CLI tool, HA uses a Home Assistant calendar entity."
        ),
        "description_technical": (
            "Calendar data source. 'google' uses gog CLI subprocess,"
            " 'ha' uses HA calendar entity via REST API. 'none'"
            " disables calendar integration."
        ),
        "category": "Calendar",
        "options": "google,ha,none",
    },
    {
        "key": "calendar.entity_id",
        "default_value": "",
        "value_type": "string",
        "label": "HA Calendar Entity",
        "description": "HA calendar entity ID (if using HA calendar).",
        "description_layman": (
            "The Home Assistant calendar entity to check. Only needed if calendar source is set to 'ha'."
        ),
        "description_technical": (
            "HA entity_id of a calendar entity, e.g."
            " 'calendar.holidays'. Used when calendar.source='ha'."
            " Queried via GET /api/calendars/{entity_id}."
        ),
        "category": "Calendar",
    },
    # ── Phase 3: Normalizer ────────────────────────────────────────────
    {
        "key": "normalizer.environmental_correlation_threshold",
        "default_value": "0.7",
        "value_type": "number",
        "label": "Environmental Correlation Threshold",
        "description": "How strongly time must correlate with light/sun to prefer sensor trigger.",
        "description_layman": (
            "If your actions strongly correlate with sunrise/sunset"
            " or light levels, ARIA will suggest using a sensor"
            " trigger instead of a time trigger."
        ),
        "description_technical": (
            "Pearson r threshold for environmental correlation."
            " Above this, prefer sun/illuminance trigger over time."
            " Range 0.3-0.95, default 0.7."
        ),
        "category": "Normalizer",
        "min_value": 0.3,
        "max_value": 0.95,
        "step": 0.05,
    },
    {
        "key": "normalizer.adaptive_window_max_sigma",
        "default_value": "90",
        "value_type": "number",
        "label": "Max Timing Variance (min)",
        "description": "If timing varies more than this many minutes, skip time condition.",
        "description_layman": (
            "If you do something at wildly different times each day,"
            " ARIA won't add a time restriction to the automation."
            " This sets the threshold for 'wildly different'."
        ),
        "description_technical": (
            "Standard deviation threshold in minutes for adaptive"
            " time windows. If σ > this value, skip_time_condition"
            " is True and no time condition is generated."
            " Range 30-180, default 90."
        ),
        "category": "Normalizer",
        "min_value": 30,
        "max_value": 180,
        "step": 10,
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
