#!/bin/bash
# FunGen installer (Linux + macOS). Bootstraps uv, then runs the latest
# install.py directly from GitHub. install.py handles cloning the repo +
# building .venv.
set -e
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv (one-time, ~15 MB)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

exec uv run --no-project --python 3.11 \
    https://raw.githubusercontent.com/ack00gar/FunGen-AI-Powered-Funscript-Generator/main/install.py "$@"
