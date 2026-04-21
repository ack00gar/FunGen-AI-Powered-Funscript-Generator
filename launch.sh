#!/bin/bash
# FunGen launcher (Linux + macOS Terminal). Self-heals: if .venv is missing
# or broken, runs install.py before launching.
set -e
cd "$(dirname "$0")"

# Homebrew on Apple Silicon ships things (mpv, ffmpeg) we want on PATH
export PATH="/opt/homebrew/bin:$PATH"

# Don't pull in user-site packages from a stray ~/.local install
export PYTHONNOUSERSITE=1

# pip torch wheels also bundle libomp on macOS; this prevents the duplicate-libomp crash
# regardless of whether we're under conda, venv, or a plain system Python.
export KMP_DUPLICATE_LIB_OK=TRUE

# Ultralytics: telemetry + offline + isolated config dir per project
export YOLO_TELEMETRY=False
export YOLO_OFFLINE=True
export YOLO_CONFIG_DIR="$(pwd)/config/ultralytics"

# Drop any active conda env vars so nothing leaks into our venv interpreter
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
        echo "Could not install uv. Run ./install.sh manually." >&2
    fi
fi

if [ ! -x "$VENV_PY" ]; then
    echo "Install failed. See output above. Re-run ./install.sh and report the issue." >&2
    exit 1
fi

exec "$VENV_PY" main.py "$@"
