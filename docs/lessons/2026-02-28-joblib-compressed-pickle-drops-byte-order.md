# Lesson: joblib Compressed Pickle Silently Converts Array dtype to Native Endianness

**Date:** 2026-02-28
**System:** community (joblib/joblib)
**Tier:** lesson
**Category:** data-model
**Keywords:** joblib, pickle, endianness, dtype, numpy, cross-platform, serialization, byte-order, compressed-pickle
**Source:** https://github.com/joblib/joblib/issues/280

---

## Observation (What Happened)

A NumPy array created with explicit big-endian dtype (`'>i8'`) was dumped with `joblib.dump(..., compress=True)` and loaded on a little-endian machine. The loaded array had dtype `'<i8'` (native little-endian). Without compression (`compress=False`), the original byte-order was preserved. The dtype silently changed only when compression was enabled.

## Analysis (Root Cause — 5 Whys)

joblib's compressed pickle path routes through a NumpyPickler that converts arrays to native byte order before compression to improve compression ratio (non-native arrays often compress worse). This is an implicit normalization step — it is not documented behavior, and no warning is emitted. When loaded on the same platform, the values are numerically identical. The bug surfaces when: (a) models are trained on big-endian systems, (b) the serialized artifact is deployed on little-endian hardware, or (c) downstream code branches on `arr.dtype.byteorder` for format negotiation.

## Corrective Actions

- Never rely on byte-order metadata surviving a joblib compressed dump/load cycle; always call `np.ascontiguousarray(arr)` or `arr.newbyteorder('=')` to normalize to native byte order before saving.
- After loading any joblib artifact that contains NumPy arrays, assert `arr.dtype.byteorder in ('<', '=', 'native', sys.byteorder[0])` if endianness matters.
- In ARIA's model serialization: scikit-learn models saved with `joblib.dump(compress=3)` have this normalization applied silently — document that cross-platform model portability is safe but byte-order metadata is not preserved.

## Key Takeaway

`joblib.dump(..., compress=True)` silently normalizes all NumPy array dtypes to native byte order; never branch on dtype.byteorder after a compressed joblib round-trip.
