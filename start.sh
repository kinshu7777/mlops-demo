#!/bin/bash
# Double-click this file (or run ./start.sh) to start the project.
# It automatically runs from wherever THIS file is located, so folder
# navigation mistakes can't happen with this launcher.

cd "$(dirname "$0")"

echo "Starting Delivery Time Predictor..."
echo ""

if command -v python3 &> /dev/null; then
    python3 control.py start
else
    python control.py start
fi
