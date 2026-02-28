# Lesson: Attaching PyTorch Lightning Callbacks Makes `torch.save(model)` Fail — State Dict Required

**Date:** 2026-02-28
**System:** community (unit8co/darts)
**Tier:** lesson
**Category:** integration
**Keywords:** PyTorch, Lightning, callback, torch.save, pickle, parametrized, weight_norm, serialization, state_dict
**Source:** https://github.com/unit8co/darts/issues/2638

---

## Observation (What Happened)

A Darts `TCNModel` with `weight_norm=True` could be saved with `model.save()` without any callbacks. However, adding a custom `pytorch_lightning.callbacks.Callback` to `pl_trainer_kwargs` caused `torch.save(self, f_out)` to fail with `RuntimeError: Serialization of parametrized modules is only supported through state_dict()`. The callback attachment caused PyTorch to serialize the Lightning module directly rather than just the model, exposing a serialization restriction on parametrized layers.

## Analysis (Root Cause — 5 Whys)

**Why #1:** `RuntimeError: Serialization of parametrized modules is only supported through state_dict()` on model save.
**Why #2:** `torch.save(model)` serializes the entire object graph — with a callback attached, the Lightning trainer and module were included in the pickle, which hit a restriction on parametrized layers.
**Why #3:** `weight_norm=True` uses `torch.nn.utils.parametrize`, which PyTorch restricts to `state_dict()` only serialization.
**Why #4:** The save path used `torch.save(self, ...)` (whole-object pickle) rather than `torch.save(self.state_dict(), ...)` for the underlying Lightning module.
**Why #5:** The two serialization code paths (with/without callbacks) exercised different internals because callback presence changed what was included in the pickle graph.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Any PyTorch model using `torch.nn.utils.parametrize` (weight normalization, spectral norm) must serialize via `state_dict()`, not `torch.save(model)` | proposed | community | issue #2638 |
| 2 | Test model `save()` → `load()` round-trip with every combination of trainer kwargs that might be used in production (including callbacks) | proposed | community | issue #2638 |
| 3 | Framework wrappers that expose trainer kwargs must document which kwargs affect serializability | proposed | community | issue #2638 |

## Key Takeaway

`torch.save(model)` (whole-object pickle) breaks when the object graph includes parametrized layers — always use `state_dict()` for serializing PyTorch models with weight normalization or custom callbacks, and test save/load with the same configuration used in training.
