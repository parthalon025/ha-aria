# Lesson: prometheus_client write_to_textfile Leaves Orphaned Temp Files on Write Error

**Date:** 2026-02-28
**System:** community (prometheus/client_python)
**Tier:** lesson
**Category:** reliability
**Keywords:** prometheus, write_to_textfile, temp file, atomic write, cleanup, exception, file leak, pushgateway
**Source:** https://github.com/prometheus/client_python/issues/1044

---

## Observation (What Happened)

`prometheus_client.write_to_textfile("", registry=registry)` raised `FileNotFoundError` when attempting the final `os.rename()` to an empty path. The temporary file created at the start of the write (e.g., `.969277.139657657282624`) was left on disk permanently. Any exception during the rename phase — permission error, full disk, invalid path — produced orphaned temp files that accumulated silently.

## Analysis (Root Cause — 5 Whys)

The implementation wrote to a temp file and then renamed it (correct atomic-write pattern), but did not wrap the rename in `try/finally` to ensure the temp file was deleted on any exception. The write-then-rename pattern is only safe if the cleanup is guaranteed — a `finally: os.unlink(tmppath)` is mandatory when the rename fails. Atomic file writes have two phases (write + rename), and both must be covered by cleanup.

## Corrective Actions

- Implement atomic writes with guaranteed cleanup:
  ```python
  tmppath = path + ".tmp"
  try:
      with open(tmppath, "w") as f:
          f.write(data)
      os.replace(tmppath, path)
  except Exception:
      try:
          os.unlink(tmppath)
      except OSError:
          pass
      raise
  ```
- Use `os.replace()` instead of `os.rename()` — it is atomic on POSIX and handles existing destination files.
- For pushgateway-style batch jobs, validate the output path before beginning the write to fail fast before creating any temp files.

## Key Takeaway

Atomic write patterns (write-to-temp + rename) must include `finally: os.unlink(tmppath)` — any exception between write and rename leaves the temp file permanently.
