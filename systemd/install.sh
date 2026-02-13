#!/bin/bash
# ARIA systemd unit installer
#
# Copies unit files to ~/.config/systemd/user/ and reloads systemd.
# Does NOT enable or start any units — do that manually after review.
#
# Usage:
#   ./systemd/install.sh
#
# After install, enable units with:
#   systemctl --user enable aria-hub.service
#   systemctl --user enable aria-snapshot.timer aria-full.timer aria-retrain.timer ...
#   systemctl --user start aria-hub.service
#   systemctl --user start aria-snapshot.timer aria-full.timer ...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${HOME}/.config/systemd/user"

echo "ARIA systemd installer"
echo "======================"
echo "Source:  ${SCRIPT_DIR}"
echo "Target:  ${TARGET_DIR}"
echo ""

# Ensure target directory exists
mkdir -p "${TARGET_DIR}"

# Ensure log directory exists
mkdir -p "${HOME}/.local/log"

# Copy all unit files
COPIED=0
for unit in "${SCRIPT_DIR}"/*.service "${SCRIPT_DIR}"/*.timer; do
    [ -f "${unit}" ] || continue
    name="$(basename "${unit}")"
    # Skip install.sh itself
    [ "${name}" = "install.sh" ] && continue
    cp "${unit}" "${TARGET_DIR}/${name}"
    echo "  Installed: ${name}"
    COPIED=$((COPIED + 1))
done

echo ""
echo "Installed ${COPIED} unit files."

# Reload systemd
systemctl --user daemon-reload
echo "systemd daemon reloaded."

echo ""
echo "Units are installed but NOT enabled. To enable:"
echo ""
echo "  # Hub service (long-running)"
echo "  systemctl --user enable --now aria-hub.service"
echo ""
echo "  # All timers"
echo "  systemctl --user enable --now \\"
echo "    aria-snapshot.timer \\"
echo "    aria-full.timer \\"
echo "    aria-retrain.timer \\"
echo "    aria-check-drift.timer \\"
echo "    aria-intraday.timer \\"
echo "    aria-correlations.timer \\"
echo "    aria-sequences.timer \\"
echo "    aria-suggest-automations.timer \\"
echo "    aria-meta-learn.timer \\"
echo "    aria-prophet.timer"
echo ""
echo "  # Verify timers"
echo "  systemctl --user list-timers 'aria-*'"
