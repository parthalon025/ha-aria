# Lesson: VCR before_record_response Returning None Silently Skips Cassette Write

**Date:** 2026-02-28
**System:** community (kevin1024/vcrpy)
**Tier:** lesson
**Category:** testing
**Keywords:** vcrpy, cassette, before_record_response, None, silent skip, HTTP mocking, record mode
**Source:** https://github.com/kevin1024/vcrpy/issues/863

---

## Observation (What Happened)
A test used `@vcr.use_cassette(before_record_response=lambda r: print(r["headers"]))` to inspect response headers before recording. The cassette file was never written. No error was raised; the test passed on first run (against the real server) but failed on re-run (attempting replay from a nonexistent cassette).

## Analysis (Root Cause — 5 Whys)
**Why #1:** The `before_record_response` callback returned `None` (Python's implicit return from a function with only a `print` call, no explicit `return`).
**Why #2:** VCR.py interprets a `None` return from `before_record_response` as a signal to drop the response — it skips recording that interaction entirely.
**Why #3:** The lambda was intended to be a side-effect hook (logging) but was written as a transformation hook. VCR.py has a single callable interface that must return the response dict.
**Why #4:** No warning is emitted when `before_record_response` returns `None` — silence is the only signal.
**Why #5:** The first test run succeeds because VCR records nothing but the real HTTP call goes through; the cassette is empty or missing on replay.

## Corrective Actions
| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | `before_record_response` callbacks MUST return the response dict (potentially modified) — never use them as pure side-effect hooks without returning the response | proposed | community | issue #863 |
| 2 | For side-effect logging, use: `before_record_response=lambda r: (print(r["headers"]), r)[1]` or a proper named function that explicitly returns `r` | proposed | community | issue #863 |
| 3 | After first cassette recording, verify the cassette file exists and has the expected number of interactions before relying on replay | proposed | community | issue #863 |

## Key Takeaway
VCR.py's `before_record_response` is a transformation hook, not an observer — returning `None` (the implicit Python return from a `print`-only lambda) silently drops the cassette interaction, producing a file with zero recorded responses and a silent failure on replay.
