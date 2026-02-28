# Lesson: JWT Signature Verification Accepts Adjacent Base64 Characters — Single-Char Tamper Bypasses Check

**Date:** 2026-02-28
**System:** community (pyjwt)
**Tier:** lesson
**Category:** security
**Keywords:** jwt, signature, base64, padding, tampering, verification bypass, HMAC, HS256, canonical decoding
**Source:** https://github.com/jpadilla/pyjwt/issues/1069

---

## Observation (What Happened)

A tampered JWT — where only the last character of the signature segment was incremented by one (e.g., `I` → `J`, `J`, `K`) — decoded successfully up to three consecutive times before finally failing. The root cause is that Base64 padding allows several distinct character sequences to decode to the same byte sequence; characters in the "padding zone" of the last group are effectively ignored by the decoder, so adjacent Base64 characters produce the same canonical bytes and pass HMAC comparison.

## Analysis (Root Cause — 5 Whys)

**Why #1:** Base64 decoding of the signature segment is non-injective at the boundary — multiple encodings map to the same decoded bytes when padding bits are present.

**Why #2:** The HMAC comparison is done on the decoded bytes, not the raw Base64 string, so the canonical comparison succeeds for all inputs that decode to the same byte sequence.

**Why #3:** The JWT spec (RFC 7515) uses Base64url without padding (`=`), which still has this property for the last few characters depending on payload length.

**Why #4:** JWT libraries delegate HMAC comparison to the underlying crypto primitive, which operates on bytes — the Base64 ambiguity is invisible at that layer.

**Why #5:** Standard test suites verify that a completely different signature fails, but do not test the narrow band of "adjacent Base64" mutations that produce the same decoded value.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | When implementing signature verification, compare the raw Base64url string, not decoded bytes, after constant-time HMAC validation — or re-encode the validated bytes and compare against the original signature string | proposed | community | issue #1069 |
| 2 | Add test cases that mutate the last 1-4 characters of a valid JWT signature and assert `InvalidSignatureError` is raised for all of them | proposed | community | issue #1069 |
| 3 | Use `hmac.compare_digest(sig_bytes, expected_bytes)` as the authoritative check — but pair it with a length check first, as equal-length comparison is a prerequisite for timing safety | proposed | community | issue #1069 |

## Key Takeaway

Base64 padding makes adjacent characters in the final group decode to the same bytes — verify JWT signatures against the decoded bytes AND assert the raw signature string matches the re-encoded expected value to catch padding-zone tampering.
