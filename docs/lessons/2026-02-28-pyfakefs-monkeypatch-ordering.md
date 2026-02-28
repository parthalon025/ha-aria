# Lesson: Layering monkeypatch Over pyfakefs Leaves builtins.open Poisoned After Teardown

**Date:** 2026-02-28
**System:** community (pytest-dev/pyfakefs)
**Tier:** lesson
**Category:** testing
**Keywords:** pytest, pyfakefs, monkeypatch, builtins.open, teardown, fixture ordering, patch leak
**Source:** https://github.com/pytest-dev/pyfakefs/issues/1200

---

## Observation (What Happened)
A test used both `fs` (pyfakefs) and `monkeypatch` and manually called `monkeypatch.setattr("builtins.open", fake_open)`. After the test completed, `builtins.open` in subsequent tests was poisoned with the pyfakefs `FakeFileOpen` object. The following test (`test_open`) could see the corrupted `open` even though no fixture was active.

## Analysis (Root Cause — 5 Whys)
**Why #1:** pyfakefs replaces `builtins.open` with its fake implementation when the `fs` fixture activates.
**Why #2:** `monkeypatch.setattr("builtins.open", fake_open)` captures the *current* value of `builtins.open` at patch time — which is already the pyfakefs fake.
**Why #3:** On monkeypatch teardown, it restores `builtins.open` to the captured value (the pyfakefs fake), not the original real `open`.
**Why #4:** pyfakefs then tears down and expects to restore `open` from its own saved real value — but monkeypatch teardown already ran and the pyfakefs saved value is now the current value again.
**Why #5:** The double-patch creates a teardown ordering problem where the last restorer wins, and neither restorer has the original real `open`.

## Corrective Actions
| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Never manually patch `builtins.open` with `monkeypatch` when `fs` (pyfakefs) is active — pyfakefs already patches it | proposed | community | issue #1200 |
| 2 | If a custom `FakeFileOpen` subclass is needed, pass it during pyfakefs fixture setup via the `fake_filesystem` argument or subclass `FakeFilesystem`, not via `monkeypatch` | proposed | community | issue #1200 |
| 3 | When combining pyfakefs and monkeypatch in one test, audit every `monkeypatch.setattr` call to ensure it is not overwriting something pyfakefs already patched | proposed | community | issue #1200 |

## Key Takeaway
Layering `monkeypatch.setattr("builtins.open", ...)` on top of an active `fs` fixture creates a double-patch that survives teardown — subsequent tests see the pyfakefs fake `open` because monkeypatch restores to the wrong value; never double-patch what a fixture already patches.
