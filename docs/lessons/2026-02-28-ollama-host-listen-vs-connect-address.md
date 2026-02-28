# Lesson: OLLAMA_HOST=0.0.0.0 Is a Listen Address — Client Must Connect to 127.0.0.1

**Date:** 2026-02-28
**System:** community (ollama/ollama-python)
**Tier:** lesson
**Category:** integration
**Keywords:** ollama, OLLAMA_HOST, 0.0.0.0, listen-address, connect-address, environment-variable, client, networking
**Source:** https://github.com/ollama/ollama-python/issues/471

---

## Observation (What Happened)

After setting `OLLAMA_HOST=0.0.0.0` (to make Ollama listen on all interfaces), `ollama.generate()` and `ollama.chat()` started failing with `ConnectError: [WinError 10049] The requested address is not valid in its context`. Direct HTTP calls to `http://localhost:11434/api/generate` continued to work.

## Analysis (Root Cause — 5 Whys)

**Why #1:** The Python client stopped working after setting `OLLAMA_HOST=0.0.0.0`.
**Why #2:** The ollama-python client reads `OLLAMA_HOST` and uses it directly as the connection target — it sent requests to `http://0.0.0.0:11434`.
**Why #3:** `0.0.0.0` is a valid bind address (means "listen on all interfaces") but is not a valid connection target for a TCP client.
**Why #4:** The library did not distinguish between a listen-address wildcard and a connect-address, treating the env var as a single URL for both purposes.
**Why #5:** The env var name (`OLLAMA_HOST`) implies a host, but the user set it to a bind wildcard — a collision in semantics with no validation or documentation warning.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | When `OLLAMA_HOST=0.0.0.0`, the Python client should substitute `127.0.0.1` as the actual connect address | proposed | community | issue #471 |
| 2 | Document explicitly: `OLLAMA_HOST` controls the server's bind address; use `OLLAMA_BASE_URL` or explicit client kwarg to set the client connect target | proposed | community | issue #471 |
| 3 | Add validation: if client is initialized with `0.0.0.0` as host, warn and substitute loopback | proposed | community | defensive |

## Key Takeaway

`OLLAMA_HOST=0.0.0.0` is a server bind wildcard — any client reading that env var and using it literally as a TCP connection target will fail; the client must substitute `127.0.0.1` when it sees the wildcard address.
