# Lesson: pyfakefs fs Fixture Conflicts With tmp_path When Not All Tests Request Both

**Date:** 2026-02-28
**System:** community (pytest-dev/pyfakefs)
**Tier:** lesson
**Category:** testing
**Keywords:** pytest, pyfakefs, fake filesystem, tmp_path, fixture ordering, FileNotFoundError, fake fs scope
**Source:** https://github.com/pytest-dev/pyfakefs/issues/1143

---

## Observation (What Happened)
A fixture chain used both `fs` (pyfakefs) and `tmp_path`. When one test requested the chain fixture and another requested only the inner part (without explicitly requesting `tmp_path`), the second test raised `FileNotFoundError` in the fake filesystem because `tmp_path`'s real directory path was not mirrored into the fake FS. The test that explicitly included `tmp_path` worked; the one that didn't explicitly include it failed at setup.

## Analysis (Root Cause — 5 Whys)
**Why #1:** pyfakefs intercepts all filesystem calls and replaces the real FS with a fake one, making paths that existed in the real FS invisible.
**Why #2:** `tmp_path` creates a real directory (`/tmp/pytest-of-<user>/pytest-N/...`). When `fs` is active, that path doesn't exist in the fake FS.
**Why #3:** pyfakefs auto-passthrough for `tmp_path` only works when pytest fixture ordering guarantees `tmp_path` is resolved before `fs` activates.
**Why #4:** When a test does not declare `tmp_path` directly, pytest's fixture resolution order changes, and pyfakefs may activate before `tmp_path` registers its path.
**Why #5:** The fixture dependency is implicit — the chain fixture uses `tmp_path` internally but the consuming test doesn't declare it, so pyfakefs cannot detect the dependency.

## Corrective Actions
| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Any test using a chain fixture that internally uses both `fs` and `tmp_path` must explicitly declare both `fs` and `tmp_path` as parameters, even if the test body doesn't use them directly | proposed | community | issue #1143 |
| 2 | Use `fs.add_real_directory(tmp_path)` inside the chain fixture to explicitly mirror the `tmp_path` directory into the fake FS before any use | proposed | community | issue #1143 |
| 3 | Avoid combining `fs` with `tmp_path` in a shared parent fixture — instead keep fake-FS tests and real-FS tests separate | proposed | community | issue #1143 |

## Key Takeaway
When combining `pyfakefs`'s `fs` fixture with `tmp_path` in a chain fixture, every consuming test must explicitly declare `tmp_path` — implicit fixture dependencies bypass pyfakefs's auto-passthrough and produce `FileNotFoundError` in the fake filesystem at setup.
