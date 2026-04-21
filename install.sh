#!/bin/bash
# FunGen installer (Linux + macOS). Bootstraps uv, then uses uv to provide
# Python 3.11 and run install.py. Avoids depending on a system Python at all
# (parity with the Windows shim, where the Microsoft Store python alias
# makes system-Python detection unreliable).
set -e
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv (one-time, ~15 MB download from astral.sh)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # uv installer modifies shell rc files; export the common install
    # locations for THIS session so the line below finds uv without restart.
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "uv install failed. astral.sh may be blocked. Install uv manually" >&2
    echo "(see https://astral.sh/uv) then re-run ./install.sh." >&2
    exit 1
fi

exec uv run --no-project --python 3.11 install.py "$@"
