# Lesson: Missing `key` on Conditionally Swapped Elements Causes Unexpected DOM Mutations

**Date:** 2026-02-28
**System:** community (preactjs/preact)
**Tier:** lesson
**Category:** frontend
**Keywords:** preact, vdom, key, reconciliation, button, type, form, submit, DOM mutation, attribute patch, conditional render
**Source:** https://github.com/preactjs/preact/issues/4922

---

## Observation (What Happened)

When a `<button type="button">` is conditionally re-rendered as a `<button type="submit">` in the same position in the virtual DOM tree (same parent, same child index), Preact (and React) optimizes by patching the existing DOM element's `type` attribute rather than replacing it. If a click event fires synchronously before the new element is fully rendered, the browser's form submit logic fires against the now-`type="submit"` button, unexpectedly submitting the form. This silently breaks multi-state forms.

## Analysis (Root Cause — 5 Whys)

**Why #1:** Clicking a button that renders as `type="button"` triggers an unexpected form submission.

**Why #2:** After the click, the conditional re-render swaps the button to `type="submit"` by patching only the `type` attribute on the existing DOM element (VDOM diffing).

**Why #3:** The browser's form submit detection runs after the click event, at which point `type="submit"` is already in the DOM.

**Why #4:** VDOM reconciliation patches attributes in-place when an element at the same tree position has the same tag name — it does not create a new DOM element unless a `key` change is detected.

**Why #5:** The developer assumed that rendering semantically different elements (a cancel button vs. a submit button) would create new DOM nodes; this is not how VDOM diff-and-patch works.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Assign different `key` values to elements that represent semantically distinct UI states at the same tree position: `<button key="edit" type="button">` / `<button key="submit" type="submit">`. This forces VDOM to unmount+remount rather than patch. | proposed | community | https://github.com/preactjs/preact/issues/4922 |
| 2 | Any time the `type` attribute of a `<button>` or `<input>` changes conditionally, treat it as requiring a `key` change — `type` changes on form elements have side effects the VDOM cannot guard against. | proposed | community | https://github.com/preactjs/preact/issues/4922 |
| 3 | More broadly: when two conditional branches render the same HTML tag but with fundamentally different semantics or event behavior, use distinct `key` values — do not rely on attribute patching to produce correct browser behavior. | proposed | community | https://github.com/preactjs/preact/issues/4922 |

## Key Takeaway

VDOM diffing patches `type` and other attributes in-place on the same DOM element — conditionally swapping `<button type="button">` for `<button type="submit">` without a `key` change causes the browser to observe an unexpected `type="submit"` during the same event loop tick, triggering accidental form submission; always use distinct `key` values when element semantics change at the same tree position.
