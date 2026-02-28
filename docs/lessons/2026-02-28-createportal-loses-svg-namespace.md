# Lesson: createPortal Loses SVG Namespace — Elements Render as HTMLUnknownElement

**Date:** 2026-02-28
**System:** community (preactjs/preact)
**Tier:** lesson
**Category:** frontend
**Keywords:** preact, portal, createPortal, SVG, namespace, namespaceURI, xhtml, MathML, rendering, invisible element
**Source:** https://github.com/preactjs/preact/issues/4992

---

## Observation (What Happened)

When SVG elements are rendered via `createPortal` into an SVG container in Preact, the portal loses the SVG namespace context. Elements are created as `http://www.w3.org/1999/xhtml` (HTMLUnknownElement) instead of `http://www.w3.org/2000/svg` (SVGElement), causing them to be completely invisible in the browser. The same code works correctly in React.

Confirmed on Preact 10.28.x. Root cause: Preact's portal implementation constructs a `_temp` container for diffing without inheriting the `namespaceURI` of the portal target, so all children created within the portal are assigned the wrong namespace.

## Analysis (Root Cause — 5 Whys)

**Why #1:** SVG elements rendered through a Preact portal are invisible.

**Why #2:** DevTools shows `namespaceURI: "http://www.w3.org/1999/xhtml"` on elements that should be `http://www.w3.org/2000/svg`.

**Why #3:** Preact's `createPortal` constructs an internal `_temp` diffing container without copying the `namespaceURI` from the portal's target element.

**Why #4:** Element creation in Preact uses the parent's namespace to decide whether to call `createElementNS`. When the temp container lacks a namespace, all children default to XHTML.

**Why #5:** React passes the portal container's namespace through its fiber reconciler; Preact's simpler portal implementation omits this propagation step.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Avoid `createPortal` for SVG content in Preact — render SVG elements directly in the component tree where the SVG context is established. | proposed | community | https://github.com/preactjs/preact/issues/4992 |
| 2 | If portaling into an SVG container is unavoidable, use `dangerouslySetInnerHTML` with a raw SVG string as a workaround until the upstream fix ships. | proposed | community | https://github.com/preactjs/preact/issues/4992 |
| 3 | The same namespace-loss issue applies to MathML portals — any non-HTML namespace portal target will silently produce wrong-namespace elements. | proposed | community | https://github.com/preactjs/preact/issues/4920 |
| 4 | When debugging invisible SVG elements, always check `element.namespaceURI` in DevTools console — a value of `http://www.w3.org/1999/xhtml` on an SVG element is the definitive signal this bug is present. | proposed | community | https://github.com/preactjs/preact/issues/4992 |

## Key Takeaway

`createPortal` in Preact does not inherit the SVG (or MathML) namespace from the target container — SVG elements portaled into an SVG parent are silently created as `HTMLUnknownElement` and rendered invisible; do not use portals for SVG content until the upstream fix ships.
