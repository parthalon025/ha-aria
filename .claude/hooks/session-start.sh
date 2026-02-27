#!/bin/bash
set -euo pipefail

# Only run on Claude Code web sessions
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

# Install package with dev dependencies
pip install -e ".[dev]"
