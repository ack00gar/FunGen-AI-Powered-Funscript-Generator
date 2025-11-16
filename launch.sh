#!/bin/bash
cd "$(dirname "$0")"
export PATH="/opt/homebrew/bin:$PATH"
echo "Activating FunGen environment..."
source "/Users/k00gar/miniconda3/bin/activate" FunGen
echo "Starting FunGen..."
python main.py "$@"
