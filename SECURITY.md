# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.x     | Yes       |

## Reporting a Vulnerability

If you discover a security vulnerability in ARIA, please report it responsibly.

**Do not open a public issue.** Instead, email [parthalon025@gmail.com](mailto:parthalon025@gmail.com) with:

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if you have one)

You should receive an acknowledgment within 48 hours. We will work with you to understand and address the issue before any public disclosure.

## Scope

ARIA runs locally and connects to your Home Assistant instance. Security concerns include:

- **API token exposure** — HA long-lived access tokens must never be logged or committed
- **WebSocket authentication** — all HA WebSocket connections require auth
- **Local-only binding** — the hub binds to `127.0.0.1` by default; exposing to `0.0.0.0` is not recommended without a reverse proxy
- **SQLite cache** — contains entity states; protect `hub.db` with filesystem permissions
