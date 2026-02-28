# Lesson: Config Keys Consumed at Runtime But Never Registered Return None on Fresh Install

**Date:** 2026-02-28
**System:** ARIA (ha-aria)
**Tier:** lesson
**Category:** configuration
**Keywords:** config, get_config_value, register_config, CONFIG_DEFAULTS, fresh install, None, cold-start, module disable
**Files:** aria/hub/routes_module_config.py:19-24, aria/shared/constants.py

---

## Observation (What Happened)

`routes_module_config.py` reads 5 config keys (`presence.enabled_signals`, `activity.enabled_domains`, etc.) via `get_config_value()` without those keys being registered in `CONFIG_DEFAULTS`. On a fresh install, `get_config_value()` returns `None` for all 5 keys, so all modules get empty source lists — they are silently disabled.

Lesson #54 covers the inverse (keys registered but never read — "dead knobs"). This is the mirror problem: keys consumed but never registered.

## Analysis (Root Cause — 5 Whys)

**Why #1:** The keys were added to the reader (`routes_module_config.py`) without being added to the config defaults registry.

**Why #2:** Config registration and config reading are in separate files; there is no compile-time or startup check that verifies all consumed keys have defaults.

**Why #3:** On an existing install, the keys may have been written to the DB via prior Settings UI interaction, masking the bug — it only surfaces on fresh installs where no DB record exists.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Register all 5 missing keys in `CONFIG_DEFAULTS` with sensible defaults | proposed | Justin | issue #181 |
| 2 | Add a startup assertion (or test) that verifies every key consumed by `get_config_value()` exists in `CONFIG_DEFAULTS` | proposed | Justin | issue #181 |
| 3 | Document the symmetric rule: "Every `get_config_value()` must have a registered default in CONFIG_DEFAULTS" (mirror of lesson #54) | proposed | Justin | — |

## Key Takeaway

Every `get_config_value(key)` must have a matching default in `CONFIG_DEFAULTS` — unregistered keys return `None` on fresh installs, silently disabling entire subsystems with no startup warning.
