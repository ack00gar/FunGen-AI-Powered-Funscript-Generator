#!/bin/bash
# FunGen Uninstaller for Linux/macOS
# Downloads and runs the universal uninstaller

echo "=========================================="
echo "     FunGen Uninstaller for Linux/macOS"
echo "=========================================="
echo ""

# Check if Python is available
if ! command -v python3 >/dev/null 2>&1 && ! command -v python >/dev/null 2>&1; then
    echo "ERROR: Python not found. Cannot run uninstaller."
    echo "Please install Python or use manual uninstall instructions."
    exit 1
fi

# Determine Python command
PYTHON_CMD="python3"
if ! command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python"
fi

# Try to find the uninstaller in current directory first
if [ -f "fungen_uninstall.py" ]; then
    echo "Found uninstaller in current directory."
    $PYTHON_CMD fungen_uninstall.py "$@"
    exit $?
fi

# Download uninstaller from GitHub
echo "Downloading FunGen uninstaller..."
UNINSTALLER_URL="https://raw.githubusercontent.com/ack00gar/FunGen-AI-Powered-Funscript-Generator/main/fungen_uninstall.py"
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Function to download files (tries multiple methods)
download_file() {
    local url=$1
    local output=$2
    
    # Try curl first
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "$url" -o "$output"
        return $?
    fi
    
    # Try wget as fallback
    if command -v wget >/dev/null 2>&1; then
        wget -q "$url" -O "$output"
        return $?
    fi
    
    # Try python as last resort
    $PYTHON_CMD -c "
import urllib.request
urllib.request.urlretrieve('$url', '$output')
"
    return $?
}

if ! download_file "$UNINSTALLER_URL" "$TEMP_DIR/fungen_uninstall.py"; then
    echo "ERROR: Failed to download uninstaller"
    echo "Please download fungen_uninstall.py manually from GitHub"
    exit 1
fi

echo "Running uninstaller..."
$PYTHON_CMD "$TEMP_DIR/fungen_uninstall.py" "$@"