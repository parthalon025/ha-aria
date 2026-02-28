# Lesson: VCR new_episodes Mode Records Wrong Request Count — Replayed Requests Not Counted in cassette.data

**Date:** 2026-02-28
**System:** community (kevin1024/vcrpy)
**Tier:** lesson
**Category:** testing
**Keywords:** vcrpy, cassette, new_episodes, request count, play_counts, allow_playback_repeats, HTTP recording, cassette.responses
**Source:** https://github.com/kevin1024/vcrpy/issues/753

---

## Observation (What Happened)
Using `record_mode="new_episodes"` with `allow_playback_repeats=False`, a test made 7 HTTP requests (1 old + 6 new). `len(cassette.data)` showed 4, not 7. Assertions on `cassette.responses` or request counts were wrong. The root cause was that repeated new-episode requests matched and reused an already-appended cassette entry without incrementing `play_counts`.

## Analysis (Root Cause — 5 Whys)
**Why #1:** In `new_episodes` mode, `cassette.append()` adds a new interaction but does not set `play_counts` for the new entry.
**Why #2:** When a subsequent request matches a newly-appended entry, VCR uses that entry for replay — but because `play_counts` is not set, it does not record the replay as a "used" interaction.
**Why #3:** VCR's cassette logic treats `play_counts=0` as "never replayed" and routes the request back to the real network in some code paths, creating a divergence between actual requests made and cassette entries.
**Why #4:** The `append` method is shared between initial load (where setting `play_counts` would be wrong) and new-episodes recording (where it is needed), creating a design ambiguity.
**Why #5:** `new_episodes` mode is the least-tested record mode and the interaction between `append` and `play_counts` was not validated for repeated same-request patterns.

## Corrective Actions
| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | When asserting on cassette request counts in `new_episodes` mode, do not rely on `len(cassette.data)` — use `cassette.all_played` or `cassette.responses` directly | proposed | community | issue #753 |
| 2 | Avoid `allow_playback_repeats=False` with `new_episodes` mode — the combination produces incorrect `play_counts` tracking | proposed | community | issue #753 |
| 3 | For deterministic request-count assertions, use `record_mode="none"` (replay only) after recording — switch to `none` for CI, use `new_episodes` only for local cassette refresh | proposed | community | issue #753 |

## Key Takeaway
VCR.py's `new_episodes` mode has a known bug where newly-appended cassette entries do not increment `play_counts`, causing `len(cassette.data)` to undercount actual requests made — do not assert on cassette size when using `new_episodes` with repeated identical requests.
