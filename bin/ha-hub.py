#!/usr/bin/env python3
"""Legacy entry point — use 'aria serve' instead."""
import sys
sys.argv = ["aria", "serve"]
from aria.cli import main
main()
