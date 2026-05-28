#!/usr/bin/env bash
set -euo pipefail

# Build a standalone GUI binary for the DemoTracking launcher.
# Run from the repository root.

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"

if ! "$PYTHON_BIN" -c "import PyInstaller" >/dev/null 2>&1; then
    echo "PyInstaller is not available for $PYTHON_BIN. Install with: $PYTHON_BIN -m pip install pyinstaller"
    exit 1
fi

"$PYTHON_BIN" -m PyInstaller \
    --noconfirm \
    --clean \
    demo-tracking-launcher.spec

echo "Built binary: dist/demo-tracking-launcher"