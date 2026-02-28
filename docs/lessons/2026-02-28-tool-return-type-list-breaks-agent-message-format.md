# Lesson: Tool Functions Returning Lists Fail in Agent Message Pipelines

**Date:** 2026-02-28
**System:** community (ag2ai/ag2)
**Tier:** lesson
**Category:** integration
**Keywords:** tool call, return type, list, serialization, agent message format, TypeError, GroupChat, Gemini, tool result, content format
**Source:** https://github.com/ag2ai/ag2/issues/2347

---

## Observation (What Happened)

In AG2 group chats using Gemini, tool functions that returned a plain Python `list` raised `TypeError: Wrong content format: every element should be dict if the content is a list`. The same tool returning a tuple worked. The framework's internal message formatter assumed list returns were lists of content-block dicts, not plain data values.

## Analysis (Root Cause — 5 Whys)

**Why #1:** The group chat's tool result serializer inspected the tool's return value and branched on `isinstance(result, list)`.
**Why #2:** When the result was a list, it assumed each element was a content-block dict — this was the Gemini multi-modal content format.
**Why #3:** Plain data lists (e.g., `[5, 3, 10]`) have no `"type"` key, failing the `AssertionError: Missing 'type' key in content's dict` check.
**Why #4:** The format contract between tool return values and the message serializer was implicit — no type annotation enforcement or wrapping layer at the tool boundary.
**Why #5:** Tuple was an accidental workaround because `isinstance(result, tuple)` was not caught by the list branch.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Wrap plain list/tuple tool returns in a content envelope before passing to the message formatter | proposed | community | https://github.com/ag2ai/ag2/issues/2347 |
| 2 | Add a runtime check: if `isinstance(result, list)` and elements are not dicts, treat as scalar serialized via `str()` or `json.dumps()` | proposed | community | issue |
| 3 | Document that tool functions must return str, dict, or list-of-dicts; enforce with a validation decorator | proposed | community | issue |

## Key Takeaway

Agent frameworks that share tool return values across a message serialization pipeline must normalize return types at the tool boundary — assuming list means list-of-content-blocks silently breaks any tool returning plain data collections.
