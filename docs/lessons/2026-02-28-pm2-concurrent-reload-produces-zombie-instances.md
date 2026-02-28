# Lesson: Concurrent PM2 Reload Calls Produce Zombie Instances Not Visible in pm2 list
**Date:** 2026-02-28
**System:** community (Unitech/pm2)
**Tier:** lesson
**Category:** reliability
**Keywords:** pm2, reload, cluster mode, concurrent, zombie process, race condition, instance count, rolling restart
**Source:** https://github.com/Unitech/pm2/issues/2951
---
## Observation (What Happened)
Two `pm2 reload app` commands were issued in rapid succession to a cluster-mode application with 8 instances and a `kill_timeout` of 10,000ms. The second reload interrupted the first mid-flight. PM2 logged `[PM2][ERROR] Process 6 not found` and `Unknown id _old_6` repeatedly. After the reloads completed, `pm2 list` showed 8 instances but `ps aux` showed more than 8 Node.js processes — the extras were ghost instances from the interrupted first reload, serving traffic from the old code version.

## Analysis (Root Cause — 5 Whys)
PM2's rolling reload renames old instances to `_old_<id>` during the swap, then sends SIGTERM after the new instance is ready. When a second reload starts before the first finishes, the process registry contains both `<id>` and `_old_<id>` entries in transitional states. The second reload cannot find `_old_6` because it was already renamed or partially cleaned up, leaving it in limbo. The process remains alive (handling connections) but PM2 loses the reference needed to send SIGTERM. The result is invisible zombie workers running old code.

## Corrective Actions
- Treat PM2 reload as non-reentrant: never issue a second `pm2 reload` until `pm2 list` shows all instances in ONLINE state with 0 `_old_` entries.
- Implement a reload lock: before reloading in CI/CD, check `pm2 jlist | jq '.[].pm2_env.status' | grep -c starting` and wait for 0.
- After every reload: run `ps aux | grep <app> | grep -v grep | wc -l` and compare to expected instance count; alert if they differ.
- For zero-downtime deployments: prefer `pm2 reload` over `pm2 restart` but serialize reload triggers — use a queue or mutex in the deployment pipeline.

## Key Takeaway
Concurrent PM2 reload calls produce invisible zombie processes serving old code — serialize reload triggers and verify instance count via `ps` (not just `pm2 list`) after every reload.
