# Lesson: Semgrep Parser Fallback on Unsupported Syntax Corrupts Pattern Matching in the Same File

**Date:** 2026-02-28
**System:** community (semgrep/semgrep)
**Tier:** lesson
**Category:** testing
**Keywords:** semgrep, parser, fallback, match-case, Python, AST, pattern matching broken, false positive, autofix corruption
**Source:** https://github.com/semgrep/semgrep/issues/10944

---

## Observation (What Happened)
A Semgrep rule that correctly matched `tuple[$TYPE]` patterns in most Python files produced incorrect partial matches and corrupted autofixes in files containing Python 3.10+ `match`/`case` statements. A valid annotation `tuple[int, str]` was matched as `tuple[int, str` (missing closing bracket), and the autofix produced syntactically invalid code.

## Analysis (Root Cause — 5 Whys)
**Why #1:** Semgrep uses two parsers per language (menhir + tree-sitter). The primary menhir parser fails on Python `match`/`case` syntax (not yet implemented in that parser).

**Why #2:** When menhir fails, Semgrep falls back to the tree-sitter parser. Tree-sitter produces a subtly different AST representation for the same code.

**Why #3:** The `...` wildcard in the rule pattern `tuple[$_, ...]` has dual meanings — ellipsis in Semgrep patterns AND Python `Ellipsis` literal — causing the tree-sitter AST traversal to match at wrong node boundaries.

**Why #4:** The fallback is silent — no warning is emitted to indicate which parser was used for which file, making the discrepancy invisible.

**Why #5:** Rules are typically developed and tested on files without `match`/`case` syntax, so the parser fallback path is never exercised during rule authoring.

## Corrective Actions
| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Always test Semgrep rules against files containing modern language features (match-case, structural pattern matching, walrus operator) even if the rule target is unrelated | proposed | community | issue #10944 |
| 2 | When autofixes are corrupted, check whether the affected file uses syntax not fully supported by the primary parser — disable the rule on those files via `.semgrepignore` | proposed | community | issue #10944 |
| 3 | Use `semgrep --dump-ast` to inspect which parser was invoked and validate the AST matches expected structure before deploying a rule | proposed | community | issue #10944 |

## Key Takeaway
Semgrep silently falls back to a different parser when the primary parser fails on unsupported syntax — this can produce incorrect pattern matches and corrupted autofixes in files that use new language features even if the rule target is unrelated code.
