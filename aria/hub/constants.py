"""Shared constants for hub modules.

Cache keys are defined here to prevent implicit coupling between modules
that read and write the same cache categories.
"""

# Cache category keys — used by set_cache / get_cache
CACHE_ACTIVITY_LOG = "activity_log"
CACHE_ACTIVITY_SUMMARY = "activity_summary"
CACHE_INTELLIGENCE = "intelligence"
CACHE_CAPABILITIES = "capabilities"
CACHE_ENTITIES = "entities"
CACHE_DEVICES = "devices"
CACHE_AREAS = "areas"
CACHE_DISCOVERY_METADATA = "discovery_metadata"

# Shadow engine table name constants
CACHE_PREDICTIONS = "predictions"
CACHE_PIPELINE_STATE = "pipeline_state"

# Phase 2: Config store + data curation table name constants
CACHE_CONFIG = "config"
CACHE_ENTITY_CURATION = "entity_curation"
CACHE_CONFIG_HISTORY = "config_history"
