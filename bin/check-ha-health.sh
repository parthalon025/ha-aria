#!/usr/bin/env bash
# bin/check-ha-health.sh — Pre-flight check before batch engine commands.
# Exit 0 if HA is healthy, exit 1 if not.
# Used as ExecStartPre in systemd timer services.

set -euo pipefail

source ~/.env

# Check HA API is reachable and returning real data
RESPONSE=$(curl -sf -m 10 \
  -H "Authorization: Bearer ${HA_TOKEN}" \
  "${HA_URL}/api/states" 2>/dev/null) || {
    echo "ARIA guard: HA API unreachable at ${HA_URL}" >&2
    exit 1
}

# Check we got actual entity data (not an error page)
ENTITY_COUNT=$(echo "$RESPONSE" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null) || {
    echo "ARIA guard: HA API returned invalid JSON" >&2
    exit 1
}

if [ "$ENTITY_COUNT" -lt 100 ]; then
    echo "ARIA guard: HA returned only $ENTITY_COUNT entities (expected 3000+), likely restarting" >&2
    exit 1
fi

echo "ARIA guard: HA healthy ($ENTITY_COUNT entities)"
exit 0
