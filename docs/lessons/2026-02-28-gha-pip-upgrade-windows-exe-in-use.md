# Lesson: `pip install --upgrade pip` Fails on Windows CI Because pip.exe Cannot Self-Replace While Running

**Date:** 2026-02-28
**System:** community (pypa/cibuildwheel)
**Tier:** lesson
**Category:** ci-cd
**Keywords:** github-actions, windows, pip, upgrade, self-replace, exe-in-use, cross-platform, python-m-pip
**Source:** https://github.com/pypa/cibuildwheel/issues/2536

---

## Observation (What Happened)

cibuildwheel's venv setup ran `pip install --upgrade pip` on Windows CI, which failed because Windows locks the `pip.exe` binary while it is running — you cannot replace an executable that is currently executing on Windows. This broke builds after pip 25.2 was released. The cross-platform fix is `python -m pip install --upgrade pip`, which replaces the pip module files rather than the running `.exe`.

## Analysis (Root Cause — 5 Whys)

**Why #1:** The upgrade command was `pip install --upgrade pip` — calling pip directly launches `pip.exe`, which then tries to overwrite itself.
**Why #2:** Windows file locking prevents in-use executables from being replaced — the OS raises a PermissionError when pip tries to overwrite `pip.exe` while that binary is the running process.
**Why #3:** On Linux/macOS, `pip install --upgrade pip` works because the OS does not lock executables on disk (it copies the inode); on Windows, the file handle is held exclusive.
**Why #4:** The fix (`python -m pip`) delegates to the Python interpreter to load pip as a module and overwrite the files — the `.exe` is not the running process, Python is, so the lock is not triggered.
**Why #5:** CI scripts copied from Linux-centric documentation lack the Windows-specific invocation pattern.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Always use `python -m pip install --upgrade pip` instead of `pip install --upgrade pip` in CI scripts that target Windows | proposed | community | https://github.com/pypa/cibuildwheel/issues/2536 |
| 2 | Add this as a standard CI script linting rule — flag bare `pip install --upgrade pip` in cross-platform workflows | proposed | community | — |
| 3 | Apply the same pattern to `pip install --upgrade setuptools` and similar self-referential upgrades | proposed | community | — |

## Key Takeaway

`pip install --upgrade pip` fails on Windows CI because Windows locks the running executable — always use `python -m pip install --upgrade pip` for cross-platform pip self-upgrades.
