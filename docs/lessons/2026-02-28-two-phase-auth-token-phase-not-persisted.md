# Lesson: Multi-Phase Auth Flow Broken When Token Phase State Is Not Persisted

**Date:** 2026-02-28
**System:** community (langgenius/dify)
**Tier:** lesson
**Category:** security
**Keywords:** two-phase auth, token, phase state, email verification, security, auth bypass, state machine, token reuse, change-email
**Source:** https://github.com/langgenius/dify/issues/32710

---

## Observation (What Happened)

Dify's change-email flow required two verification steps: (1) verify old email ownership, (2) verify new email ownership. The token issued in step 1 was not tagged with its phase state. A user (or attacker) could capture the step-1 token and directly call the reset endpoint (`change-email/reset`) with a new email address, bypassing the new-email verification step entirely. The reset endpoint accepted the token without checking that it had advanced to the final verified-new-email phase.

## Analysis (Root Cause — 5 Whys)

**Why #1:** The reset endpoint did not validate that the presented token was in the "new email verified" phase.
**Why #2:** Token phase state was not persisted server-side — once issued, a token was valid for any phase transition.
**Why #3:** The two-phase state machine existed in documentation but was not encoded as a state stored alongside the token.
**Why #4:** The endpoint only validated the token's signature and associated email, not its lifecycle stage.
**Why #5:** Security review focused on token authenticity (is it a real token?) rather than token authorization (is this token allowed to perform this action at this stage?).

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Persist token phase metadata server-side (e.g., `{token_id: ..., phase: "old_email_verified", new_email: null}`) | proposed | community | https://github.com/langgenius/dify/issues/32710 |
| 2 | Reset endpoint must reject tokens not in the `"new_email_verified"` phase and reject `new_email` mismatches | proposed | community | issue |
| 3 | Add unit tests for each invalid phase transition: step-1 token → reset (should fail), step-2 token → step-2 again (should fail) | proposed | community | issue |

## Key Takeaway

Multi-phase authentication flows must store phase state server-side with the token — a token that is cryptographically valid is not the same as a token that is authorized for the current phase; failing to store and check phase state converts a two-step verification into a one-step bypass.
