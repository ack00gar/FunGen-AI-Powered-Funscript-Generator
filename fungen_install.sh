#!/bin/bash
# FunGen Universal Bootstrap Installer for Linux/macOS
# This script requires ZERO dependencies - only uses POSIX shell built-ins
# Downloads and runs the full Python installer

set -e  # Exit on any error

echo "=========================================="
echo "     FunGen Bootstrap Installer"
echo "=========================================="
echo ""
echo "This installer will download and install everything needed:"
echo "  - Python 3.11 (Miniconda)"
echo "  - Git"
echo "  - FFmpeg/FFprobe"
echo "  - FunGen AI and all dependencies"
echo ""

# Detect OS and architecture
OS=$(uname -s)
ARCH=$(uname -m)

case $OS in
    Linux*)
        PLATFORM="Linux"
        PYTHON_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
        if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
            PYTHON_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"
        fi
        ;;
    Darwin*)
        PLATFORM="macOS"
        PYTHON_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
        if [ "$ARCH" = "arm64" ]; then
            PYTHON_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
        fi
        ;;
    *)
        echo "ERROR: Unsupported operating system: $OS"
        exit 1
        ;;
esac

echo "Detected: $PLATFORM ($ARCH)"
echo ""

# Create temp directory
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Configuration
INSTALLER_URL="https://raw.githubusercontent.com/ack00gar/FunGen-AI-Powered-Funscript-Generator/main/fungen_universal_installer.py"
PYTHON_INSTALLER="$TEMP_DIR/miniconda-installer.sh"
UNIVERSAL_INSTALLER="$TEMP_DIR/fungen_universal_installer.py"
MINICONDA_PATH="$HOME/miniconda3"

# Function to download files (tries multiple methods)
download_file() {
    local url=$1
    local output=$2
    local description=$3
    
    echo "  Downloading $description..."
    
    # Try curl first (most common)
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "$url" -o "$output"
        return $?
    fi
    
    # Try wget as fallback
    if command -v wget >/dev/null 2>&1; then
        wget -q "$url" -O "$output"
        return $?
    fi
    
    # Try python if available (unlikely but possible)
    if command -v python3 >/dev/null 2>&1; then
        python3 -c "
import urllib.request
urllib.request.urlretrieve('$url', '$output')
"
        return $?
    fi
    
    echo "ERROR: No download tool available (curl, wget, or python3)"
    return 1
}

echo "[1/4] Downloading Miniconda installer..."
if ! download_file "$PYTHON_URL" "$PYTHON_INSTALLER" "Miniconda"; then
    echo "ERROR: Failed to download Miniconda installer"
    exit 1
fi
echo "    Miniconda installer downloaded successfully"

echo ""
echo "[2/4] Installing Miniconda..."
echo "    This may take a few minutes..."
chmod +x "$PYTHON_INSTALLER"
"$PYTHON_INSTALLER" -b -p "$MINICONDA_PATH"
if [ $? -ne 0 ]; then
    echo "ERROR: Miniconda installation failed"
    exit 1
fi
echo "    Miniconda installed successfully"

# Add conda to PATH for this session
export PATH="$MINICONDA_PATH/bin:$PATH"

echo ""
echo "[3/4] Downloading FunGen universal installer..."
if ! download_file "$INSTALLER_URL" "$UNIVERSAL_INSTALLER" "FunGen universal installer"; then
    echo "ERROR: Failed to download universal installer"
    exit 1
fi
echo "    Universal installer downloaded successfully"

echo ""
echo "[4/4] Running FunGen universal installer..."
echo "    The universal installer will now handle the complete setup..."
echo ""

# Run the universal Python installer
python "$UNIVERSAL_INSTALLER" --dir "$(pwd)"
INSTALL_RESULT=$?

if [ $INSTALL_RESULT -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "    Bootstrap Installation Complete!"
    echo "=========================================="
    echo ""
    echo "FunGen has been successfully installed."
    echo "Check above for launcher instructions."
else
    echo ""
    echo "=========================================="
    echo "      Installation Failed"
    echo "=========================================="
    echo ""
    echo "Please check the error messages above."
    if [ "$PLATFORM" = "macOS" ]; then
        echo "You may need to install Xcode Command Line Tools:"
        echo "  xcode-select --install"
    fi
fi

exit $INSTALL_RESULT