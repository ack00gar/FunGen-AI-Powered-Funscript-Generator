#!/bin/bash
cd "$(dirname "$0")"
export PATH="/opt/homebrew/bin:$PATH"

# Prevent pip/python from using user-site packages
export PYTHONNOUSERSITE=1

# Disable Ultralytics telemetry for privacy
export YOLO_TELEMETRY=False

# Prevent OMP duplicate libomp crash (conda + torch both ship libomp)
export KMP_DUPLICATE_LIB_OK=TRUE

# Activate environment (skip if already active to avoid double-activation)
if [ "$CONDA_DEFAULT_ENV" != "FunGen" ]; then
    echo "Activating FunGen environment..."
    source "/Users/k00gar/miniconda3/bin/activate" FunGen
else
    echo "FunGen environment already active."
fi
echo "Starting FunGen..."
python main.py "$@"

echo ""
read -p "Press Enter to close..."
