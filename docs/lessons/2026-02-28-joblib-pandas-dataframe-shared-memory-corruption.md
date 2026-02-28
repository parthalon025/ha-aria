# Lesson: joblib Shared Memory Memmaps Corrupt Pandas DataFrame Internal Buffers

**Date:** 2026-02-28
**System:** community (joblib/joblib)
**Tier:** lesson
**Category:** reliability
**Keywords:** joblib, shared-memory, memmap, pandas, dataframe, data-corruption, parallel, numpy, mmap-mode
**Source:** https://github.com/joblib/joblib/issues/451

---

## Observation (What Happened)

A Pandas DataFrame was `joblib.dump()`-ed and loaded with `mmap_mode='r'`, then passed to `Parallel(n_jobs=2)` workers. Column names and the index survived intact, but the underlying numerical data (`df.values`) was corrupted into random float values. The same workflow with a bare NumPy array (`df.values`) worked correctly. The corruption only manifested when: (a) the data was numerical, (b) the DataFrame exceeded a size threshold that triggered memmap, and (c) more than one worker was used.

## Analysis (Root Cause — 5 Whys)

`joblib.dump()` serializes a DataFrame by pickling its metadata (columns, index) and memory-mapping its underlying NumPy arrays. When Pandas wraps those arrays back into a DataFrame object, internal buffer references are stored differently than in a plain NumPy array — Pandas may hold references to multiple block arrays whose alignment the memmap doesn't preserve correctly across worker process boundaries on some platforms. The plain NumPy path skips the block-manager reconstruction and reads the memmap directly, avoiding the corruption.

## Corrective Actions

- Never pass a Pandas DataFrame via `joblib` memmap to parallel workers; convert to a NumPy array (`df.values` or `df.to_numpy()`) before passing and reconstruct the DataFrame inside the worker if needed.
- If a full DataFrame must be shared, use `multiprocessing.shared_memory` with explicit layout control, not `joblib.dump` + `mmap_mode`.
- Add a smoke test: after any parallel job that receives a large DataFrame argument, assert `np.allclose(result, expected)` on a known-value subset before consuming the output.

## Key Takeaway

Pass NumPy arrays (not Pandas DataFrames) through joblib's shared-memory memmap layer — DataFrame block-manager reconstruction across process boundaries corrupts numerical data silently.
