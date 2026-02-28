# Fix-All Conventions — 11 Patterns for ARIA Issues

All sub-agents executing batches of the fix-all-issues plan MUST read this document before touching any code. These conventions define the exact fix patterns for the recurring bug types found across open issues. Follow them literally.

---

## Convention A: Silent Return → Logged Return

**Pattern name:** `silent-return`
**Applies to:** Any function that returns `None`, `[]`, or `{}` when a resource is unavailable

**Wrong:**
```python
def predict(self, data):
    if self._model is None:
        return None
```

**Correct:**
```python
def predict(self, data):
    if self._model is None:
        model_path = getattr(self, '_model_path', '<not configured>')
        logger.warning(
            "%s.predict() called but model not loaded — path: %s",
            self.__class__.__name__, model_path
        )
        return None  # or typed empty default — see rules below
```

**Rules:**
- Log at WARNING (never DEBUG, never INFO for missing-resource paths)
- Include class name AND resource path/identifier in the message
- Return a TYPED empty default matching the function's return type:
  - `-> float` → `return 0.0`
  - `-> list` → `return []`
  - `-> dict` → `return {}`
  - `-> Optional[X]` where None is documented → `return None` WITH the warning log
- NEVER swallow the condition silently — if it can fail, it will fail silently

---

## Convention B: Missing Guard (Python — snap/dict access)

**Pattern name:** `null-guard-python`
**Applies to:** Any bare `dict["key"]` or `obj["key"]["subkey"]` on data from external sources (snapshots, HA API, cache)

**Wrong:**
```python
result = snapshot["presence"]["room"]
data = resp["items"][0]
```

**Correct:**
```python
presence = snapshot.get("presence", {})
result = presence.get("room")
if result is None:
    logger.warning(
        "snapshot missing presence.room — snapshot id: %s",
        snapshot.get("id", "unknown")
    )
    return {}
```

For `json.load()`:
```python
try:
    data = json.load(f)
except json.JSONDecodeError as e:
    corrupt_path = path.with_suffix(path.suffix + ".corrupt")
    path.rename(corrupt_path)
    logger.warning("Corrupt JSON at %s — renamed to %s: %s", path, corrupt_path, e)
    return {}  # typed empty default
```

**Rules:**
- Use `.get()` with typed default — never bare `[]` on external data
- Log at WARNING with enough context to identify which call site
- Wrap all `json.load()` in try/except JSONDecodeError; rename corrupt files with `.corrupt` suffix

---

## Convention C: Missing Guard (JavaScript/JSX — array access)

**Pattern name:** `null-guard-js`
**Applies to:** Any `.map()`, `.filter()`, `.length`, or `.forEach()` on data received from API or props

**Wrong:**
```js
data.map(item => renderItem(item))
if (occupants.length > 0) ...
matrix[0].length
```

**Correct:**
```js
Array.isArray(data) ? data.map(item => renderItem(item)) : []
if (Array.isArray(occupants) && occupants.length > 0) ...
Array.isArray(matrix) && matrix.length > 0 && matrix[0].length > 0
```

For uPlot specifically:
```js
// Before passing to uPlot:
if (!Array.isArray(series) || series.length === 0 || !series[0] || series[0].length === 0) {
    return <div class="chart-empty">No data</div>
}
```

**Rules:**
- Use `Array.isArray(x)` before ANY `.map()`/`.filter()`/`.length` on props or API data
- Do NOT use `x?.length > 0` as a substitute — optional chaining does not validate array type
- uPlot requires `data[0].length > 0` — existence check alone is insufficient (Lesson #101)

---

## Convention D: Missing shutdown() Method

**Pattern name:** `missing-shutdown`
**Applies to:** Any module class with `self._task`, `self._session`, or background threads

**Template:**
```python
async def shutdown(self) -> None:
    """Cancel in-flight tasks and release resources."""
    if hasattr(self, '_task') and self._task is not None:
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    if hasattr(self, '_session') and self._session is not None:
        await self._session.close()
        self._session = None

    logger.debug("%s shutdown complete", self.__class__.__name__)
```

For `asyncio.to_thread` tasks:
```python
# In shutdown():
if self._thread_task is not None:
    self._thread_task.cancel()
    try:
        await self._thread_task
    except (asyncio.CancelledError, Exception):
        pass
    self._thread_task = None
```

**Rules:**
- Every module with background tasks MUST have `async def shutdown(self) -> None`
- Must cancel tasks AND close sessions — do both
- Log at DEBUG on completion (shutdown is normal lifecycle, not an error)
- Register with hub: confirm module registration includes shutdown callback

---

## Convention E: Wrong Response Shape (JavaScript — error fallbacks)

**Pattern name:** `response-shape-js`
**Applies to:** Any HTTP error fallback or 404 default in store.js / api.js

**Principle (Lesson #98):** HTTP error fallback objects must match the success response shape exactly.

**Wrong:**
```js
// 404 fallback returns wrong shape
case 404: return {}

// Mutation error swallowed
} catch(e) {
    console.error(e)  // user sees nothing
}
```

**Correct:**
```js
// In api.js — define typed empty-state constants:
export const EMPTY_CAPABILITIES = { capabilities: {}, entities: {}, devices: {}, areas: {} }
export const EMPTY_INTELLIGENCE = { predictions: [], anomalies: [], correlations: [], drift: null }
export const EMPTY_EVENTS = { events: [], total: 0, limit: 100 }
export const EMPTY_SETTINGS = { discovery: {}, thresholds: {}, modules: {} }

// safeFetch: surface non-404 errors
async function safeFetch(url, opts) {
    const resp = await fetch(url, opts)
    if (!resp.ok) {
        if (resp.status === 404) return null  // caller uses empty-state constant
        throw new Error(`HTTP ${resp.status} ${resp.statusText} from ${url}`)
    }
    return resp.json()
}

// Mutation catch blocks: surface to user
} catch(e) {
    setError(e.message || 'Save failed — check console')  // user-visible
    console.error(e)
}
```

**Rules:**
- Define an `EMPTY_*` constant per major data shape — used by BOTH success path and error path
- `safeFetch` must throw (not return null) for non-404 errors
- Mutation catch blocks MUST update visible error state — `console.error` alone is a silent failure

---

## Convention F: Integration Seam Contract

**Pattern name:** `seam-contract`
**Applies to:** Any value that crosses a layer boundary (Python→JSON, module→hub, engine→hub, frontend→backend)

**datetime → JSON boundary:**
```python
# Wrong — raw datetime crashes json.dumps
hub.publish("event", {"timestamp": datetime.now()})

# Correct — always isoformat at the boundary
hub.publish("event", {"timestamp": datetime.now(tz=timezone.utc).isoformat()})
```

**Feature column → ML boundary:**
```python
# Wrong — two places define the feature list independently
# training builds: ["feat_a", "feat_b"]
# inference builds: ["feat_a", "feat_b", "feat_c"]  <- SKEW

# Correct — single source of truth
from aria.engine.features.feature_config import FEATURE_COLUMNS
# use FEATURE_COLUMNS in BOTH training and inference
```

**Hub method → module boundary:**
```python
# Wrong — calling a method that doesn't exist
value = self.hub.get_config_value("unifi.host")  # AttributeError

# Correct — verify the actual hub API first, use what exists
value = self.hub.get_config("unifi", "host")  # check hub/core.py for real method name
```

**API auth → CORS boundary (test requirement):**
- Fixes to auth headers and CORS must be verified with `ARIA_API_KEY=test-key` set
- Localhost + no-auth masks both issues — test with auth enabled

**Rules:**
- datetime objects → call `.isoformat()` before any `json.dumps()` / `publish()` / `set_cache()`
- Feature lists → extract to shared constant in `aria/engine/features/feature_config.py`
- Hub method calls → verify method exists in `aria/hub/core.py` before calling
- Auth/CORS → integration smoke test required (unit tests alone are insufficient)

---

## Convention G: asyncio.get_event_loop() Replacement

**Pattern name:** `event-loop-api`
**Applies to:** Any `asyncio.get_event_loop()` call in async context or code called from async

**Wrong:**
```python
loop = asyncio.get_event_loop()
loop.run_until_complete(coro())
```

**Correct (inside a coroutine):**
```python
loop = asyncio.get_running_loop()
```

**Correct (entry point / not in coroutine):**
```python
asyncio.run(coro())
```

**Rules:**
- NEVER use `asyncio.get_event_loop()` — deprecated and raises RuntimeError in Python 3.12+
- In `async def`: use `asyncio.get_running_loop()`
- In sync context launching async: use `asyncio.run()`
- In tests: use `pytest-asyncio` or `new_event_loop() + run_until_complete()` pattern

---

## Convention H: Typed String Enums (Literal)

**Pattern name:** `typed-literal`
**Applies to:** Any `str` annotation where only a fixed set of values is valid

**Wrong:**
```python
level: str  # only "warning", "critical", "info" are valid
```

**Correct:**
```python
from typing import Literal
level: Literal["warning", "critical", "info"]
```

**Rules:**
- Use `Literal` for ALL string fields with a fixed valid set
- If the set has >5 values, use an `Enum` instead
- Always include the Literal type in the class `__init__` or `TypedDict` definition

---

## Convention I: asyncio.to_thread() for Blocking I/O in Async Context

**Pattern name:** `async-blocking-io`
**Applies to:** `open()`, `json.load()`, `pickle.load()`, `sqlite3` calls inside `async def` methods

**Wrong:**
```python
async def _load_models(self):
    with open(path) as f:
        data = json.load(f)
```

**Correct:**
```python
async def _load_models(self):
    def _read():
        with open(path) as f:
            return json.load(f)
    data = await asyncio.to_thread(_read)
```

**Rules:**
- ANY blocking I/O inside `async def` must use `asyncio.to_thread()`
- This includes: file reads >1KB, sqlite3 queries, pickle.load, numpy file ops
- Small config reads (<1KB, called infrequently) may be exempt with a comment explaining why
- Always log a WARNING if `asyncio.to_thread` raises — do not swallow

---

## Convention J: AbortController for Preact fetch Cleanup

**Pattern name:** `fetch-abort`
**Applies to:** Any `fetch()` call inside `useEffect` that may outlive component mount

**Wrong:**
```js
useEffect(() => {
  fetch(url).then(r => r.json()).then(setData)
}, [])
```

**Correct:**
```js
useEffect(() => {
  const controller = new AbortController()
  fetch(url, { signal: controller.signal })
    .then(r => r.json())
    .then(setData)
    .catch(e => { if (e.name !== 'AbortError') console.error(e) })
  return () => controller.abort()
}, [])
```

**Rules:**
- Every `fetch` in `useEffect` must have a cleanup function that calls `controller.abort()`
- Catch `AbortError` separately — it is not a real error, do not surface to user
- For polling intervals: clear the interval AND abort any in-flight fetch in cleanup

---

## Convention K: Issue Auto-Close in Commits

**Pattern name:** `issue-close`
**Applies to:** ALL fix commits

Every commit that fixes a confirmed GitHub issue MUST include `"closes #NNN"` in the commit message body. Format:

```
fix(domain): brief description of fix

closes #NNN
```

Or for multiple issues in one commit:

```
fix(domain): brief description

closes #NNN, closes #MMM
```

**Rules:**
- This causes GitHub to auto-close the issue when the PR merges — no exceptions
- If a commit partially addresses an issue but doesn't fully fix it, use `refs #NNN` instead
- The `closes #NNN` MUST be in the commit body (not the subject line)

---

## Convention L: sqlite3 Must Use contextlib.closing() (Lesson #34)

**Pattern name:** `sqlite3-closing`
**Applies to:** Any `sqlite3.connect()` call in the codebase (NOT `aiosqlite`)

**Wrong:**
```python
with sqlite3.connect(path) as conn:
    conn.execute(...)
# conn is NOT closed — 'with sqlite3' manages the transaction, not the connection
```

**Correct:**
```python
from contextlib import closing
with closing(sqlite3.connect(path)) as conn:
    conn.execute(...)
# conn.close() is guaranteed on exit
```

**Rules:**
- EVERY `sqlite3.connect()` must be wrapped with `closing()`
- `aiosqlite` uses `async with aiosqlite.connect() as db:` which IS correct — do not change
- If you see `sqlite3` without `closing()`, fix it as part of any touch to that file
