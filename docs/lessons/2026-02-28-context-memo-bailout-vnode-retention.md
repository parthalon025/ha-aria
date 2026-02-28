# Lesson: Context + memo Bailout Retains Consumer Subtree VNodes in Memory

**Date:** 2026-02-28
**System:** community (preactjs/preact)
**Tier:** lesson
**Category:** performance
**Keywords:** preact, vdom, context, memo, memory leak, vnode, reconciliation, bailout, heap, detached nodes
**Source:** https://github.com/preactjs/preact/issues/4914

---

## Observation (What Happened)

When a `Context.Provider` updates its value at high frequency and a consumer subtree is wrapped in `memo` (causing occasional render bailouts), the previously rendered consumer subtree is not released from memory. Heap snapshots show accumulating detached VNodes/objects over time, growing linearly.

The root cause: when `memo` bails out, it does not update `props.children`. Through context propagation, the framework still updates the immediate children — but `props.children` continues to reference the stale (pre-bailout) VNode tree. Those stale vnodes are never cleared and therefore never GC'd.

## Analysis (Root Cause — 5 Whys)

**Why #1:** Heap grows after running a high-frequency context provider with a memoized consumer subtree.

**Why #2:** When `memo` bails out, `props.children` on the memoized component retains a reference to the old rendered tree.

**Why #3:** Preact mutates VNodes directly rather than using its own internal copy, so the user-supplied `props.children` becomes a long-lived retention root.

**Why #4:** Context propagation does a second-pass commit on the consumer subtree, but bailed-out boundaries skip updating `props.children`, leaving the old tree reachable.

**Why #5:** This is a structural issue with Preact's two-pass commit + direct vnode mutation model — it only manifests when memo bailouts and context propagation interact simultaneously.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Avoid toggling large memoized subtrees inside high-frequency context providers. Use a stable intermediate component that reads context and passes derived props, so memo sees prop changes cleanly. | proposed | community | https://github.com/preactjs/preact/issues/4914 |
| 2 | If a subtree must be conditionally mounted/unmounted inside a fast-updating context, use an explicit `key` to force full unmount/remount, preventing stale vnode accumulation. | proposed | community | https://github.com/preactjs/preact/issues/4914 |
| 3 | Profile heap in production builds (not dev/prefresh) — prefresh holds extra references that mask real leaks. | proposed | community | https://github.com/preactjs/preact/issues/5001 |

## Key Takeaway

Combining a high-frequency Context.Provider with a `memo`-wrapped consumer subtree that conditionally mounts/unmounts is a vnode retention trap — the memo bailout prevents `props.children` from being cleared, causing detached nodes to accumulate; use a stable intermediate component or an explicit `key` to break the retention chain.
