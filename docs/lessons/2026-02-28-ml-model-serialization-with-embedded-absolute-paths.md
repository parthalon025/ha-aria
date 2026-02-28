# Lesson: ML Model Serialization Embeds Absolute Install Paths — Breaks on Deploy

**Date:** 2026-02-28
**System:** community (yzhao062/pyod)
**Tier:** lesson
**Category:** data-model
**Keywords:** joblib, pickle, serialization, absolute path, deploy, model loading, SUOD, cross-environment, inference
**Source:** https://github.com/yzhao062/pyod/issues/367

---

## Observation (What Happened)

A SUOD (Scalable Unsupervised Outlier Detection) model trained in one environment (training compute) and serialized with `joblib.dump()` failed to load in a different environment (inference compute) because the serialized model embedded an absolute path to a pre-built prediction file (`/azureml-envs/.../bps_prediction.joblib`) that was only present at the original training install location.

## Analysis (Root Cause — 5 Whys)

**Why #1:** `FileNotFoundError` on `bps_prediction.joblib` when loading the serialized model.
**Why #2:** SUOD's constructor loaded a bundled model file by absolute path using `__file__` resolution — the path was baked into the serialized object state.
**Why #3:** `joblib.dump()` / `pickle` serializes the object's `__dict__`, which included the resolved absolute path string, not a relative path or a lazy-load reference.
**Why #4:** The model was designed for single-environment use; cross-environment deployment was not a design consideration.
**Why #5:** No test verified model round-trip (train → serialize → load in different path) before release.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Any ML model that references bundled files must use `importlib.resources` or path resolution relative to the package at load time, not at train time | proposed | community | issue #367 |
| 2 | Test model serialization round-trips with a different working directory than the one used during training | proposed | community | issue #367 |
| 3 | Document whether a model artifact is self-contained or has external file dependencies — if external, the deployment package must include those files | proposed | community | issue #367 |

## Key Takeaway

`joblib.dump()` serializes the object's current state including any resolved absolute paths — models that reference bundled files must defer path resolution to load time, not bake it in at training time, or they break in every deployment environment.
