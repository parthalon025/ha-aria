# Lesson: Boolean Config from Environment Variable Stays True Because String "false" Is Truthy

**Date:** 2026-02-28
**System:** community (TheR1D/shell_gpt)
**Tier:** lesson
**Category:** error-handling
**Keywords:** config, environment-variable, boolean, string, truthy, DISABLE_STREAM, parsing, type-coercion, ini-file
**Source:** https://github.com/TheR1D/shell_gpt/issues/705

---

## Observation (What Happened)

Setting `DISABLE_STREAM=true` in the sgptrc config file had no effect — streaming remained enabled. Investigation showed the config value was read as a string and compared incorrectly, so the string `"false"` evaluated as truthy, leaving the flag permanently in its default state.

## Analysis (Root Cause — 5 Whys)

**Why #1:** `DISABLE_STREAM=true` in the config file did not disable streaming.
**Why #2:** The config reader returned the value as a string `"true"` or `"false"` rather than a Python bool.
**Why #3:** The code checked `if config_value:` or `if config_value == True:` — both of which evaluate `"false"` (non-empty string) as truthy.
**Why #4:** No explicit string-to-bool coercion was applied when reading the config value.
**Why #5:** Boolean config values from text files (INI, .env, dotrc) are always strings — treating them as booleans without explicit parsing is a recurring mistake.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Parse boolean config values explicitly: `value.lower() in ("true", "1", "yes")` | proposed | community | issue #705 |
| 2 | Use `configparser.getboolean()` for INI-style config files — it handles "true"/"false"/"yes"/"no" correctly | proposed | community | Python docs |
| 3 | Add a type validation layer at config load time that converts string values to declared types | proposed | community | defensive |
| 4 | Regression test: set each boolean config to both `true` and `false` and assert the resulting behavior | proposed | community | regression |

## Key Takeaway

Any boolean read from an environment variable, INI file, or dotrc file is a string — `"false"` is truthy in Python; always coerce with explicit string comparison (`value.lower() == "true"`) or `configparser.getboolean()` before treating it as a bool.
