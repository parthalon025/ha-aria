# Phase 3+4: Issue Triage, GitHub Roadmap & Fix

**Date:** 2026-02-19
**Status:** Approved
**Parent:** `2026-02-19-lean-audit-roadmap.md`
**Scope:** Merges roadmap Phases 3 (Issue Triage & GitHub Roadmap) and 4 (Fix & Optimize) into a single execution phase.

## Purpose

Triage all 56 open GitHub issues against the Phase 1 leaner architecture (14 → 10 modules), set up GitHub project infrastructure, then fix all surviving issues in priority order.

## Context

- **Phase 1 (Done):** Archived 4 modules (online_learner, organic_discovery, transfer_engine, activity_labeler), merged data_quality → discovery, renamed pattern_recognition → trajectory_classifier
- **Phase 2 (In Progress):** Known-answer test harness — design/plan written, implementation parallel with this phase
- **Current state:** 56 open issues, 3 closed, no milestones, no project board

## Pass 1: Issue Triage

### Classification Rules

| Category | Rule | Action | Execution |
|----------|------|--------|-----------|
| **close-archived** | Exclusively about archived module | Close + `archived` label + comment | Auto |
| **close-dissolved** | Root cause eliminated by Phase 1 restructuring | Close + `archived` label + comment | Auto |
| **close-fixed** | Already resolved in current codebase (Phase 1 or otherwise) — verified by checking actual code state | Close + comment linking evidence | Auto |
| **annotate-partial** | References archived modules AND surviving concerns | Comment noting archived portions; keep open | Review |
| **re-label** | Missing priority/category labels | Add labels based on content analysis | Auto |
| **keep-as-is** | Correctly labeled, still relevant | No action | — |

### Issues Referencing Archived Modules

Identified via body/title search:

| Issue | Archived Module | Likely Category |
|-------|----------------|-----------------|
| #10 | transfer_engine | close-archived (requests Phase 4 modules) |
| #25 | data_quality | annotate-partial (cold-start concern survives in discovery) |
| #27 | transfer_engine | annotate-partial (cache bypass concern survives) |
| #31 | online_learner | review (backpressure concern may be resolved) |
| #33 | activity_labeler | annotate-partial (Ollama monitoring survives) |
| #48 | data_quality | annotate-partial (snapshot quality concern survives) |
| #52 | organic_discovery | annotate-partial (blocking I/O in other modules survives) |
| #53 | data_quality | annotate-partial (N+1 queries in shadow_engine survives) |
| #55 | data_quality | annotate-partial (sequential init concern survives) |
| #57 | online_learner | review (scan_hardware redundancy may be resolved) |
| #61 | online_learner | close-dissolved or annotate-partial |
| #62 | online_learner, transfer_engine | annotate-partial (cross-layer coupling survives) |

### Ambiguous Issues for User Review

Issues requiring judgment call (not purely mechanical):
- Mixed archived/surviving concerns where the surviving portion may also be resolved
- Issues where Phase 1 restructuring partially addressed the root cause
- Priority re-evaluation based on leaner architecture

## Pass 2: GitHub Infrastructure

### New Labels

| Label | Color | Purpose |
|-------|-------|---------|
| `archived` | `#C5DEF5` | Closed because module was archived in Phase 1 |
| `phase:4` | `#D93F0B` | Assigned to Fix & Optimize work |
| `phase:5` | `#1D76DB` | Assigned to UI Decision Tool work |

### Milestones

| Milestone | Description |
|-----------|-------------|
| Phase 4: Fix & Optimize | All surviving issues after triage — security > reliability > performance > architecture |
| Phase 5: UI Decision Tool | OODA-based dashboard redesign issues |

### Project Board

**Format:** Kanban
**Columns:** Backlog → Ready → In Progress → Review → Done
**Scope:** All ha-aria issues

## Pass 3: Fix All Surviving Issues

### Priority Order (from roadmap)

1. **Security** — API auth (#64), CORS (#44), credential exposure (#43), WebSocket auth (#65)
2. **Reliability** — silent failures (#21, #45, #63), unbounded collections (#56), race conditions (#22), retry logic (#47), cold-start (#23, #25, #50)
3. **Performance** — blocking I/O (#51, #52, #18), N+1 queries (#53), startup parallelization (#55), cache overhead (#54), session reuse (#58)
4. **Architecture** — cross-layer coupling (#62), config propagation (#24), module registration (#60), dual dispatch (#32)

### Fix Approach

- Each fix gets a known-answer test where applicable (Phase 2 infrastructure built as needed)
- One issue per commit with `closes #N` in message
- Quality gates (`pytest --timeout=120 -x -q`) between every batch of 3-5 fixes
- Parallel sub-agents for independent fixes within same priority tier

### Batch Structure (Planned)

| Batch | Priority | Issues | Dependencies |
|-------|----------|--------|-------------|
| 1 | Security | #43, #44, #64, #65 | Independent |
| 2 | Reliability (critical) | #19, #20, #21 | Independent |
| 3 | Reliability (high) | #22, #23, #25, #27 | #25 may interact with #23 |
| 4 | Reliability (medium) | #45, #46, #47, #56 | Independent |
| 5 | Performance (blocking) | #51, #52, #18 | Independent |
| 6 | Performance (queries) | #53, #54, #55, #58 | #53 and #54 touch same hot path |
| 7 | Architecture | #24, #60, #62 | #60 may simplify #62 |
| 8 | Remaining | All unlisted surviving issues | TBD after triage |

## Success Criteria

- [ ] Open issue count reduced by ~50% from triage closures
- [ ] All surviving issues labeled with priority + category
- [ ] GitHub milestones created for Phases 4 and 5
- [ ] Kanban project board operational with all issues
- [ ] All surviving issues fixed (security > reliability > performance > architecture)
- [ ] Known-answer tests added for fixes where applicable
- [ ] No regressions in existing test suite
- [ ] Roadmap updated to reflect merged Phase 3+4

## Risks

- **Scope:** ~40+ fixes is ambitious. Priority ordering ensures highest-value work lands first if we stop partway.
- **Phase 2 parallel:** Known-answer test infrastructure built incrementally alongside fixes. Some fixes may lack golden-snapshot validation until Phase 2 completes.
- **Cross-issue interactions:** Fixes in the same module may conflict. Batch structure groups related fixes and runs quality gates between batches.
