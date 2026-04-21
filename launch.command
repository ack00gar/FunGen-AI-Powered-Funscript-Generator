#!/bin/bash
# FunGen launcher (macOS Finder double-click). Pauses on exit so users see
# any error message before the Terminal window closes.
set -e
cd "$(dirname "$0")"

export PATH="/opt/homebrew/bin:$PATH"
export PYTHONNOUSERSITE=1
export KMP_DUPLICATE_LIB_OK=TRUE
export YOLO_TELEMETRY=False
export YOLO_OFFLINE=True
export YOLO_CONFIG_DIR="$(pwd)/config/ultralytics"

unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL

VENV_PY=".venv/bin/python"

if [ ! -x "$VENV_PY" ]; then
    echo "FunGen environment missing, running installer (one-time, ~2 min)..."
    if ! command -v uv >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    fi
    if command -v uv >/dev/null 2>&1; then
        uv run --no-project --python 3.11 install.py || true
    else
        echo "Could not install uv. Open Terminal here and run ./install.sh."
    fi
fi

if [ -x "$VENV_PY" ]; then
    "$VENV_PY" main.py "$@"
else
    echo
    echo "Install failed. See output above."
    echo "Open Terminal here and run:  ./install.sh"
fi

echo
read -n 1 -s -r -p "Press any key to close..."
echo
