---
name: lint-before-commit
enabled: true
event: bash
pattern: git\s+commit
---

**Run ruff before committing.** This project enforces lint compliance.

Before committing, you MUST:
1. Run `.venv/bin/ruff check aria/ tests/ --fix` to auto-fix what's possible
2. Run `.venv/bin/ruff check aria/ tests/` to verify zero violations remain
3. Fix any remaining violations manually
4. Only then commit

If ruff reports violations and you commit anyway, CI will reject the PR.
