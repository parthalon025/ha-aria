# Lesson: GitHub Actions `${{ }}` Expressions in `run:` Blocks Enable Script Injection

**Date:** 2026-02-28
**System:** community (pypa/gh-action-pypi-publish)
**Tier:** lesson
**Category:** security
**Keywords:** github-actions, injection, expression, run-block, env-block, GHSA-vxmw-7h4f-hqxh, workflow-security
**Source:** https://github.com/pypa/gh-action-pypi-publish/issues/380

---

## Observation (What Happened)

A security advisory (GHSA-vxmw-7h4f-hqxh) was filed against gh-action-pypi-publish because `${{ env.VAR }}` expressions used directly inside `run:` shell blocks bypass normal shell quoting rules — an attacker who controls the variable value (e.g., via a PR branch name or issue title) can inject arbitrary shell commands. The fix was to pass all dynamic values through the `env:` block rather than interpolating them directly in the shell script.

## Analysis (Root Cause — 5 Whys)

**Why #1:** Workflow author wrote `run: echo ${{ env.MY_VAR }}` because it mirrors normal shell variable syntax.
**Why #2:** GitHub Actions processes `${{ }}` expressions *before* the shell sees the string — the substituted value is inserted as raw unquoted text, not as a shell variable.
**Why #3:** Shell quoting (single/double quotes) wrapping the expression only protects the outer shell layer; the `${{ }}` substitution has already happened at the Actions expression layer.
**Why #4:** Values sourced from untrusted external inputs (PR title, branch name, issue body via `github.event.*`) land directly in the shell command string — a value containing `; rm -rf /` executes.
**Why #5:** The distinction between Actions expression evaluation and shell evaluation is not surfaced in tooling; the YAML looks syntactically correct and works fine for benign values.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Pass all dynamic values into `run:` via the `env:` block — reference them as `$MY_VAR` (shell variable), not `${{ env.MY_VAR }}` (expression) | proposed | community | https://github.com/pypa/gh-action-pypi-publish/issues/380 |
| 2 | Use `${{ inputs.value }}` only in `with:` or `env:` assignment positions, never inline in `run:` | proposed | community | https://github.com/advisories/GHSA-vxmw-7h4f-hqxh |
| 3 | Audit all `run:` blocks for `${{ github.event.* }}` or `${{ env.* }}` inline interpolation | proposed | community | — |

## Key Takeaway

`${{ }}` in a `run:` block is string interpolation before shell parsing — not a shell variable — so it is injection-vulnerable; all dynamic values must flow through the `env:` block and be referenced as normal shell variables.
