# Lesson: EnvironmentFile Silently Ignores Invalid Lines — No Error, No Warning
**Date:** 2026-02-28
**System:** community (systemd/systemd)
**Tier:** lesson
**Category:** configuration
**Keywords:** systemd, EnvironmentFile, environment, silent failure, configuration, service, env vars
**Source:** https://github.com/systemd/systemd/issues/2836
---
## Observation (What Happened)
A service unit referenced an `EnvironmentFile` containing a shell command (`ulimit -n 2048`) rather than simple `KEY=VALUE` assignments. systemd loaded the file without error, started the service, and reported success — but the environment variable was never set and no diagnostic was emitted. The operator had no signal that the EnvironmentFile was partially or fully ignored.

## Analysis (Root Cause — 5 Whys)
systemd's EnvironmentFile parser only accepts `KEY=VALUE` pairs; shell syntax (ulimit, export, variable expansion) is not executed and is silently skipped. The parser does not differentiate between "file contains no vars" (empty/valid) and "file contains unrecognized syntax" (misconfiguration). Because no error is returned, `systemctl start` reports success, making the failure invisible to operators and monitoring systems. This problem commonly surfaces when migrating services from SysV/sysconfig files where shell commands in `/etc/sysconfig/` were historically valid.

## Corrective Actions
- Validate EnvironmentFile contents before deploying: every line must be `KEY=VALUE`, `# comment`, or blank. Shell syntax (export, ulimit, $(subst)) is always wrong.
- Add a startup assertion: log `os.environ.get("EXPECTED_KEY")` at process init to fail loudly if required env vars are absent rather than silently running with wrong config.
- Use `systemd-analyze verify` to catch unit-level problems; supplement with a custom lint step that greps EnvironmentFiles for non-`KEY=VALUE` lines in CI.
- In Python services: `if not os.environ.get("REQUIRED_VAR"): raise RuntimeError("REQUIRED_VAR not set — check EnvironmentFile")`.

## Key Takeaway
EnvironmentFile is not a shell script — it only parses `KEY=VALUE`; anything else is silently dropped, so validate the file format and assert env var presence at startup.
