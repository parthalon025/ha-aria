# Enable LLM Naming for Organic Discovery

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Route organic discovery naming through ollama-queue (not direct Ollama), add missing `update_settings()` method, enable LLM naming, and trigger a re-discovery run.

**Architecture:** The LLM naming code already exists in `naming.py`. Three changes: (1) route `_call_ollama()` through ollama-queue at port 7683 instead of direct Ollama at 11434, (2) add `update_settings()` to OrganicDiscoveryModule, (3) switch `naming_backend` setting from `"heuristic"` to `"ollama"` via API.

**Tech Stack:** Python/aiohttp, ollama-queue API, deepseek-r1:8b model

---

### Task 1: Route `_call_ollama()` through ollama-queue

**Files:**
- Modify: `aria/modules/organic_discovery/naming.py:158-169`
- Test: `tests/hub/test_organic_naming.py`

**Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_call_ollama_uses_queue_endpoint():
    """_call_ollama should use ollama-queue (port 7683), not direct Ollama."""
    with patch("aiohttp.ClientSession") as mock_session:
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"response": "test_name"})
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)
        mock_post = AsyncMock(return_value=mock_resp)
        mock_session.return_value.__aenter__ = AsyncMock(return_value=MagicMock(post=mock_post))
        mock_session.return_value.__aexit__ = AsyncMock(return_value=False)

        from aria.modules.organic_discovery.naming import _call_ollama
        await _call_ollama("test prompt")

        call_args = mock_post.call_args
        assert "7683" in call_args[0][0]  # URL contains ollama-queue port
```

**Step 2: Run test, verify fail**

**Step 3: Update `_call_ollama()` to use ollama-queue**

Change port 11434 → 7683, and endpoint from localhost to 127.0.0.1 for consistency with activity_labeler.

**Step 4: Run test, verify pass. Run full naming suite.**

**Step 5: Commit**

```bash
git add aria/modules/organic_discovery/naming.py tests/hub/test_organic_naming.py
git commit -m "fix: route organic discovery LLM naming through ollama-queue"
```

---

### Task 2: Add `update_settings()` method to OrganicDiscoveryModule

**Files:**
- Modify: `aria/modules/organic_discovery/module.py`
- Test: `tests/hub/test_organic_discovery_module.py`

**Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_update_settings_changes_naming_backend(mock_hub):
    module = OrganicDiscoveryModule(mock_hub)
    assert module.settings["naming_backend"] == "heuristic"

    await module.update_settings({"naming_backend": "ollama"})
    assert module.settings["naming_backend"] == "ollama"

    # Verify persisted to cache
    mock_hub.set_cache.assert_called_with(
        "discovery_settings", module.settings, {"source": "settings_update"}
    )

@pytest.mark.asyncio
async def test_update_settings_rejects_invalid_backend(mock_hub):
    module = OrganicDiscoveryModule(mock_hub)
    with pytest.raises(ValueError):
        await module.update_settings({"naming_backend": "invalid"})

@pytest.mark.asyncio
async def test_update_settings_preserves_other_settings(mock_hub):
    module = OrganicDiscoveryModule(mock_hub)
    await module.update_settings({"naming_backend": "ollama"})
    assert module.settings["promote_threshold"] == 50  # unchanged
```

**Step 2: Run test, verify fail**

**Step 3: Implement `update_settings()`**

```python
async def update_settings(self, updates: dict) -> None:
    """Update discovery settings with validation."""
    VALID_BACKENDS = {"heuristic", "ollama"}
    VALID_MODES = {"suggest_and_wait", "auto_promote", "manual_only"}

    if "naming_backend" in updates and updates["naming_backend"] not in VALID_BACKENDS:
        raise ValueError(f"Invalid naming_backend: {updates['naming_backend']}. Must be one of {VALID_BACKENDS}")
    if "autonomy_mode" in updates and updates["autonomy_mode"] not in VALID_MODES:
        raise ValueError(f"Invalid autonomy_mode: {updates['autonomy_mode']}. Must be one of {VALID_MODES}")

    self.settings.update(updates)
    await self.hub.set_cache("discovery_settings", self.settings, {"source": "settings_update"})
    self.logger.info(f"Settings updated: {updates}")
```

**Step 4: Run test, verify pass**

**Step 5: Commit**

```bash
git add aria/modules/organic_discovery/module.py tests/hub/test_organic_discovery_module.py
git commit -m "feat: add update_settings() method to organic discovery module"
```

---

### Task 3: Enable LLM naming and trigger re-discovery

**Step 1: Enable via API**

```bash
curl -X PUT http://127.0.0.1:8001/api/settings/discovery \
  -H "Content-Type: application/json" \
  -d '{"naming_backend": "ollama"}'
```

**Step 2: Trigger re-discovery**

```bash
aria discover-organic
```

**Step 3: Verify names improved**

```bash
curl -s http://127.0.0.1:8001/api/cache/capabilities | python3 -c "..."
```

---

## Task Dependencies

```
Task 1 (ollama-queue routing) ──┐
                                ├──► Task 3 (enable + trigger)
Task 2 (update_settings)  ─────┘
```

**Wave 1:** Tasks 1, 2 (parallel)
**Wave 2:** Task 3 (sequential, needs service restart)
