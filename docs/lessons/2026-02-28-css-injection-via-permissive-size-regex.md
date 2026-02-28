# Lesson: CSS Injection via Permissive Regex — Unsanitized URL Params into CSS Template Literals

**Date:** 2026-02-28
**System:** community (MauriceNino/dashdot)
**Tier:** lesson
**Category:** error-handling
**Keywords:** CSS injection, URL query parameter, styled-components, regex validation, sanitization, XSS, iframe, embed, CVSS
**Source:** https://github.com/MauriceNino/dashdot/issues/1378

---

## Observation (What Happened)

The dashdot dashboard embed mode accepted URL query parameters (`textOffset`, `textSize`, `gap`, `innerRadius`) that were validated with the regex `/^\d+\D+$/` — which only requires the string to start with digits and end with non-digits. This allows a payload like `1px;display:none` to pass validation and be interpolated directly into styled-components CSS template literals, enabling arbitrary CSS injection. Since the widget was designed for `<iframe>` embedding, a crafted URL could fully hide content, load external resources (leaking viewer IP), or deface the embedded widget. CVSS 3.1 score: 4.3–6.3 Medium.

## Analysis (Root Cause — 5 Whys)

**Why #1:** Attackers can craft a URL that hides dashboard content or loads external resources.

**Why #2:** URL query params are interpolated directly into CSS template literals (`margin-top: ${offset}; font-size: ${size};`) after a validation step.

**Why #3:** The validation regex `/^\d+\D+$/` only enforces "starts with digits, ends with non-digits" — it does not reject semicolons, colons, or other CSS syntax characters.

**Why #4:** The developer assumed that requiring a unit suffix (e.g., `px`) was sufficient sanitization, not anticipating that CSS syntax characters after the unit would be passed through.

**Why #5:** No allowlist of valid CSS units or strict character whitelist was applied; the regex was written to accept `12px` but unintentionally also accepts `12px;color:red`.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Replace permissive regex with a strict allowlist: `/^\d+(\.\d+)?(px|em|rem|vh|vw|%)$/` — this only accepts a number followed by a known safe unit, rejecting any additional characters. | resolved | MauriceNino/dashdot v6.3.0 | https://github.com/MauriceNino/dashdot/issues/1378 |
| 2 | Never interpolate user-controlled strings directly into CSS template literals (styled-components, emotion, CSS-in-JS). Always sanitize to a numeric value + known unit before interpolation. | proposed | community | https://github.com/MauriceNino/dashdot/issues/1378 |
| 3 | URL parameters that control visual presentation (size, spacing, color) in embeddable widgets are a CSS injection surface — treat them with the same paranoia as SQL parameters. | proposed | community | https://github.com/MauriceNino/dashdot/issues/1378 |
| 4 | Audit: search codebase for `styled.*(url|query|param|search)` and `css\`.*\${.*param` patterns — these are injection sinks. | proposed | community | https://github.com/MauriceNino/dashdot/issues/1378 |

## Key Takeaway

A regex that validates format but not content (e.g., `/^\d+\D+$/`) allows CSS injection when the matched string is interpolated into a CSS template literal — use a strict unit allowlist regex and never interpolate user-controlled strings directly into CSS properties.
