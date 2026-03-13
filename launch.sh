#!/bin/bash
cd "$(dirname "$0")"
export PATH="/opt/homebrew/bin:$PATH"

# Prevent pip/python from using user-site packages
export PYTHONNOUSERSITE=1

# Disable Ultralytics telemetry for privacy
export YOLO_TELEMETRY=False

# Isolate Ultralytics config to project directory (prevents cross-project corruption)
export YOLO_CONFIG_DIR="$(dirname "$0")/config/ultralytics"

# Prevent Ultralytics from phoning home during normal usage
export YOLO_OFFLINE=True

# Activate environment (skip if already active to avoid double-activation)
if [ "$CONDA_DEFAULT_ENV" != "FunGen" ]; then
    echo "Activating FunGen environment..."
    source "$HOME/miniconda3/bin/activate" FunGen
else
    echo "FunGen environment already active."
fi
echo "Starting FunGen..."
python main.py "$@"
