#!/bin/bash
# Thin shim: run the real installer (install.py) with a system Python.
# install.py uses uv to manage a Python 3.11 + the FunGen .venv.
set -e
cd "$(dirname "$0")"

for py in python3 python; do
    if command -v "$py" >/dev/null 2>&1; then
        exec "$py" install.py "$@"
    fi
done

echo "Python is required but was not found on PATH." >&2
echo "Install Python from https://www.python.org/ and re-run this script." >&2
exit 1
