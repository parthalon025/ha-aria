# Lesson: LLM-Supplied Values Interpolated Unescaped Into Persistent Artifacts

**Date:** 2026-02-28
**System:** ARIA (ha-aria)
**Tier:** lesson
**Category:** security
**Keywords:** LLM, prompt injection, YAML, sanitization, entity ID, interpolation, automation, security
**Files:** aria/engine/llm/automation_suggestions.py:107-109

---

## Observation (What Happened)

In `automation_suggestions.py`, the `trigger_entity` field from the LLM's JSON response is string-interpolated directly into the YAML output file:

```python
entity = s.get("trigger_entity", "unknown")
s["yaml"] = f"alias: 'ARIA Suggestion: {entity}'\n" + s["yaml"]
```

If the LLM returns a `trigger_entity` containing a YAML-breaking character (`'`, `\n`, `:`) or injection payload, it is written verbatim into the `.yaml` file. Users import these suggestions directly into Home Assistant.

## Analysis (Root Cause — 5 Whys)

**Why #1:** The LLM response is treated as trusted input and interpolated without sanitization.

**Why #2:** LLMs can return any string content — even from a known prompt, output is not guaranteed to be free of special characters.

**Why #3:** YAML is injection-sensitive: a single quote or newline in a scalar can break the entire document or inject extra fields. The developer assumed entity IDs would be safe HA slugs.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Strip YAML-unsafe characters from LLM-supplied entity values before interpolation: `.replace("'", "").replace("\n", "").replace(":", "")` | proposed | Justin | issue #222 |
| 2 | Use a YAML library (`yaml.dump()`) to serialize the alias field rather than string f-strings | proposed | Justin | issue #222 |
| 3 | Apply the same treatment to all other LLM-supplied fields that are written to YAML artifacts | proposed | Justin | — |

## Key Takeaway

Never interpolate LLM-supplied string values directly into YAML, SQL, or shell strings — treat all LLM output as untrusted user input and sanitize or use the appropriate serializer.
