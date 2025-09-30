#!/bin/bash
# FunGen Universal Bootstrap Installer for Linux/macOS
# Version: 1.0.2
# This script requires ZERO dependencies - only uses POSIX shell built-ins
# Downloads and runs the full Python installer

set -e  # Exit on any error

BOOTSTRAP_VERSION="1.0.2"

# Check for help or common invalid flags
for arg in "$@"; do
    case $arg in
        -h|--help)
            echo "FunGen Bootstrap Installer"
            echo "Usage: $0 [options]"
            echo ""
            echo "This script downloads and installs FunGen automatically."
            echo "All options are passed to the universal installer."
            echo ""
            echo "Common options:"
            echo "  --force     Force reinstallation"
            echo "  --uninstall Run uninstaller instead"
            echo "  --help      Show this help"
            echo ""
            exit 0
            ;;
        -u)
            echo "ERROR: '-u' is not a valid option."
            echo "Did you mean '--uninstall' or '--force'?"
            echo "Run '$0 --help' for available options."
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "     FunGen Bootstrap Installer"
echo "            v${BOOTSTRAP_VERSION}"
echo "=========================================="
echo ""
echo "This installer will download and install everything needed:"
echo "  - Python 3.11 (Miniconda)"
echo "  - Git"
echo "  - FFmpeg/FFprobe"
echo "  - FunGen AI and all dependencies"
echo ""
echo "Note: You may be prompted for your password to install system packages"
echo "      (Git, FFmpeg) via your system's package manager."
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
        # Recommend Rosetta 2 for Apple Silicon if running x86_64 Python
        if [ "$ARCH" = "arm64" ]; then
            echo "NOTE: Apple Silicon (M1/M2/M3) detected!"
            echo "      If you encounter issues with x86_64 packages,"
            echo "      consider installing Rosetta 2:"
            echo "      softwareupdate --install-rosetta"
            echo ""
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
INSTALLER_URL="https://raw.githubusercontent.com/ack00gar/FunGen-AI-Powered-Funscript-Generator/main/install.py"
PYTHON_INSTALLER="$TEMP_DIR/miniconda-installer.sh"
UNIVERSAL_INSTALLER="$TEMP_DIR/install.py"
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

# Function to handle interactive package installations
handle_interactive_install() {
    local cmd=("$@")
    echo "  Note: This installation may require your interaction (password, agreement acceptance)"
    echo "  Running: ${cmd[*]}"
    if ! "${cmd[@]}"; then
        echo "  Installation failed. You may need to run this command manually:"
        echo "    sudo ${cmd[*]}"
        return 1
    fi
    return 0
}

echo "[1/4] Checking Python installation..."
if [ -d "$MINICONDA_PATH" ]; then
    echo "    Miniconda already installed, skipping download..."
else
    echo "    Downloading Miniconda installer..."
    if ! download_file "$PYTHON_URL" "$PYTHON_INSTALLER" "Miniconda"; then
        echo "ERROR: Failed to download Miniconda installer"
        exit 1
    fi
    echo "    Miniconda installer downloaded successfully"
fi

echo ""
echo "[2/4] Installing Miniconda..."
if [ -d "$MINICONDA_PATH" ]; then
    echo "    Miniconda already installed at $MINICONDA_PATH"
    echo "    Using existing installation..."
else
    echo "    This may take a few minutes..."
    chmod +x "$PYTHON_INSTALLER"
    "$PYTHON_INSTALLER" -b -p "$MINICONDA_PATH"
    if [ $? -ne 0 ]; then
        echo "ERROR: Miniconda installation failed"
        exit 1
    fi
    echo "    Miniconda installed successfully"
fi

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

# Pass through any command line arguments to the universal installer
# Use the conda python explicitly to avoid system python issues
CONDA_PYTHON="$MINICONDA_PATH/bin/python"
if [ ! -f "$CONDA_PYTHON" ]; then
    echo "WARNING: Conda python not found at $CONDA_PYTHON"
    echo "         Falling back to PATH python"
    CONDA_PYTHON="python"
fi

if [ $# -gt 0 ]; then
    echo "    Passing arguments: $@"
    "$CONDA_PYTHON" "$UNIVERSAL_INSTALLER" --dir "$(pwd)" --bootstrap-version "$BOOTSTRAP_VERSION" "$@"
else
    "$CONDA_PYTHON" "$UNIVERSAL_INSTALLER" --dir "$(pwd)" --bootstrap-version "$BOOTSTRAP_VERSION"
fi
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
        echo ""
        echo "For Apple Silicon systems, you may need Rosetta 2:"
        echo "  softwareupdate --install-rosetta"
    fi
fi

exit $INSTALL_RESULT