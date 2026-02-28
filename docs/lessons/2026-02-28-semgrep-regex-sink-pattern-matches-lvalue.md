# Lesson: Semgrep `pattern-regex` Sink Matches Left-Hand Side Variable Reassignment — False Positive in Taint Mode

**Date:** 2026-02-28
**System:** community (semgrep/semgrep)
**Tier:** lesson
**Category:** testing
**Keywords:** semgrep, taint, false positive, pattern-regex, sink, lvalue, variable reassignment, taint analysis, rule authoring
**Source:** https://github.com/semgrep/semgrep/issues/10984

---

## Observation (What Happened)
A taint-mode Semgrep rule used `pattern-regex: '.*sink(\d*).*#?'` as a sink pattern. It falsely flagged `tainted = obj.sink(81)` — a line where `tainted` (the tainted variable from a prior assignment) was being reassigned to the sink's return value, not passed as a sink argument. The tainted data was not flowing into the sink; the rule reported it as if it did.

## Analysis (Root Cause — 5 Wheys)
**Why #1:** `pattern-regex` in Semgrep matches text over AST node representations — it matched the entire line `tainted = obj.sink(81)` because `tainted` appeared on the left-hand side and the pattern matched the line text.

**Why #2:** Taint analysis tracks whether a tainted variable reaches a sink's *argument position* — but `pattern-regex` as a sink definition does not specify argument position, allowing the pattern to match on the variable name appearing anywhere in the matched text.

**Why #3:** The developer intended to capture `obj.sink(arg)` calls but the regex `.*sink.*` also matched lines where a tainted variable on the LHS coincidentally appeared before the sink call.

**Why #4:** Semgrep's semantic analysis for taint mode uses the pattern to identify potential sink *expressions*, not sink *argument bindings* — a regex without structural anchoring can over-match.

**Why #5:** The documentation does not make explicit that `pattern-regex` for sinks should be restricted to argument focus using `focus-metavariable`.

## Corrective Actions
| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Use structural AST patterns for sinks: `pattern: $FUNC(..., $ARG, ...)` with `focus-metavariable: $ARG` and `metavariable-regex` to constrain `$FUNC` | proposed | maintainer | issue #10984 comment |
| 2 | Avoid bare `pattern-regex` for sink definitions in taint mode — always pair with `focus-metavariable` to explicitly mark which argument carries taint | proposed | community | issue #10984 |
| 3 | Test each taint rule with a case where a tainted variable is reassigned on the LHS of a sink call to verify it does not trigger | proposed | community | issue #10984 |

## Key Takeaway
`pattern-regex` sinks in Semgrep taint mode match line text, not argument position — a tainted variable name appearing anywhere on the matched line (including as LHS of assignment) produces a false positive; use `pattern:` + `focus-metavariable:` instead.
