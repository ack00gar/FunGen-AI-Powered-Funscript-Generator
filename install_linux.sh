#!/bin/bash

# --- START OF USER CONFIGURATION ---
REPO_URL="YOUR_REPO_URL_HERE"
PROJECT_DIR_NAME="MyProject" # Optional: if empty, uses repo name
GOFILE_DIRECT_LINK="YOUR_GOFILE_DIRECT_LINK_HERE"
MODEL_FILENAME_ON_DISK="your_model.file"
ENV_NAME="my_project_env"
PYTHON_VERSION="3.9" # Used if environment.yml is not found
MAIN_PYTHON_SCRIPT="main.py"
# --- END OF USER CONFIGURATION ---

echo "Starting Project Installation for Linux..."

# Set Project Directory
if [ -z "$PROJECT_DIR_NAME" ]; then
    PROJECT_DIR_NAME=$(basename "$REPO_URL" .git)
fi
PROJECT_PATH="$(pwd)/$PROJECT_DIR_NAME"

# Helper function to check if a command exists
command_exists () {
    command -v "$1" >/dev/null 2>&1
}

# 1. Check for prerequisites
for cmd in git curl; do
    if ! command_exists $cmd; then
        echo "$cmd is not installed."
        echo "Please install it using your distribution's package manager (e.g., sudo apt install $cmd or sudo yum install $cmd) and re-run this script."
        exit 1
    fi
done
echo "Git and curl found."

# 2. Clone Repository
if [ -d "$PROJECT_PATH" ]; then
    echo "Project directory $PROJECT_PATH already exists. Skipping clone."
else
    echo "Cloning repository $REPO_URL into $PROJECT_PATH..."
    git clone "$REPO_URL" "$PROJECT_PATH"
    if [ $? -ne 0 ]; then
        echo "Failed to clone repository."
        exit 1
    fi
fi
cd "$PROJECT_PATH" || exit

# 3. Check for Conda / Install Miniconda
MINICONDA_INSTALL_PATH="$HOME/miniconda3" # Default
CONDA_EXE="$MINICONDA_INSTALL_PATH/bin/conda"
CONDA_ACTIVATE_SCRIPT="$MINICONDA_INSTALL_PATH/bin/activate"

if ! command_exists conda && ! [ -f "$CONDA_EXE" ]; then
    echo "Conda not found. Attempting to install Miniconda..."
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    MINICONDA_INSTALLER="Miniconda3-latest-Linux.sh"

    echo "Downloading Miniconda installer..."
    curl -L "$MINICONDA_URL" -o "$MINICONDA_INSTALLER"
    if [ $? -ne 0 ]; then
        echo "Failed to download Miniconda installer. Please download and install manually from anaconda.com."
        rm -f "$MINICONDA_INSTALLER"
        exit 1
    fi

    echo "Starting Miniconda installation (will install to $MINICONDA_INSTALL_PATH)..."
    bash "$MINICONDA_INSTALLER" -b -p "$MINICONDA_INSTALL_PATH"
    rm -f "$MINICONDA_INSTALLER"

    echo "Miniconda installation complete. Please close and reopen your terminal, or source your shell configuration file (e.g., source ~/.bashrc or source ~/.zshrc) for changes to take effect."
    echo "Then, re-run this script."
    # Attempt to initialize for current shell if possible
    eval "$($MINICONDA_INSTALL_PATH/bin/conda shell.bash hook)"
    if ! command_exists conda && ! [ -f "$CONDA_EXE" ]; then
        echo "Failed to initialize Conda in the current session. Please restart your terminal and re-run this script."
        exit 1
    fi
else
    echo "Conda found."
     # If conda command exists but install path is different from default
    if command_exists conda && [ ! -f "$CONDA_EXE" ]; then
        CONDA_BASE_DIR=$(conda info --base)
        CONDA_EXE="$CONDA_BASE_DIR/bin/conda"
        CONDA_ACTIVATE_SCRIPT="$CONDA_BASE_DIR/bin/activate"
    fi
    # Initialize conda for the script's session if needed
    if ! type conda > /dev/null 2>&1; then
      eval "$($CONDA_EXE shell.bash hook)"
    fi
fi

# 4. Create Conda Environment
echo "Checking for Conda environment: $ENV_NAME..."
if ! "$CONDA_EXE" env list | grep -q "^$ENV_NAME\s"; then
    echo "Creating Conda environment $ENV_NAME..."
    if [ -f "environment.yml" ]; then
        echo "Found environment.yml. Creating environment from file..."
        "$CONDA_EXE" env create -f environment.yml -n "$ENV_NAME"
    else
        echo "environment.yml not found. Creating environment with Python $PYTHON_VERSION..."
        "$CONDA_EXE" create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
        if [ $? -ne 0 ]; then
            echo "Failed to create Conda environment."
            exit 1
        fi
        echo "Activating environment to install requirements..."
        source "$CONDA_ACTIVATE_SCRIPT" "$ENV_NAME"
        pip install uv
        uv sync
        echo "Attempting to install FFmpeg via Conda..."
        conda install -c conda-forge ffmpeg -y
        conda deactivate
    fi
    if [ $? -ne 0 ]; then
        echo "Failed to create or set up Conda environment $ENV_NAME."
        exit 1
    fi
else
    echo "Conda environment $ENV_NAME already exists. Skipping creation."
    echo "To update, run: $CONDA_EXE env update -f environment.yml --name $ENV_NAME --prune"
fi

# 5. Create "models" subfolder
if [ ! -d "models" ]; then
    echo "Creating 'models' subdirectory..."
    mkdir "models"
fi

# 6. Download file from Gofile
echo "Downloading model file from $GOFILE_DIRECT_LINK..."
if [ -z "$GOFILE_DIRECT_LINK" ]; then
    echo "GOFILE_DIRECT_LINK is not set. Skipping download."
    echo "Please download the model manually and place it in the 'models' folder as '$MODEL_FILENAME_ON_DISK'."
else
    curl -L "$GOFILE_DIRECT_LINK" -o "models/$MODEL_FILENAME_ON_DISK"
    if [ $? -ne 0 ]; then
        echo "Failed to download model file. This might be due to an incorrect or non-direct link."
        echo "Please download it manually from $GOFILE_DIRECT_LINK"
        echo "And place it in the 'models' folder as '$MODEL_FILENAME_ON_DISK'."
    else
        echo "Model downloaded to models/$MODEL_FILENAME_ON_DISK."
    fi
fi

# 7. Create Launcher
LAUNCHER_SCRIPT_NAME="launch_linux.sh"
DESKTOP_LAUNCHER_NAME="${PROJECT_DIR_NAME,,}-launcher.desktop" # Lowercase project name
echo "Creating launcher script ($LAUNCHER_SCRIPT_NAME)..."
cat << EOF > "$LAUNCHER_SCRIPT_NAME"
#!/bin/bash
SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
CONDA_BASE_DIR="\$(conda info --base)" # More robust way to find conda

echo "Activating Conda environment $ENV_NAME..."
source "\$CONDA_BASE_DIR/bin/activate" "$ENV_NAME"

echo "Launching $MAIN_PYTHON_SCRIPT from \$SCRIPT_DIR..."
python "\$SCRIPT_DIR/$MAIN_PYTHON_SCRIPT" "\$@"

# Optional: Deactivate environment after script finishes
# conda deactivate
EOF
chmod +x "$LAUNCHER_SCRIPT_NAME"
if [ $? -ne 0 ]; then
    echo "Failed to create or set permissions for launcher script $LAUNCHER_SCRIPT_NAME."
else
    echo "Launcher script '$LAUNCHER_SCRIPT_NAME' created in $PROJECT_PATH."
fi

# Create .desktop file for GUI launcher
echo "Creating .desktop file ($DESKTOP_LAUNCHER_NAME)..."
cat << EOF > "$DESKTOP_LAUNCHER_NAME"
[Desktop Entry]
Version=1.0
Type=Application
Name=$PROJECT_DIR_NAME Launcher
Comment=Launch $PROJECT_DIR_NAME application
Exec=gnome-terminal -- bash -c "'$PROJECT_PATH/$LAUNCHER_SCRIPT_NAME'; exec bash"
Icon= # You can specify an icon path here, e.g., /path/to/icon.png
Path=$PROJECT_PATH
Terminal=false # The script itself will run in a terminal launched by Exec
Categories=Application;Development;
EOF
chmod +x "$DESKTOP_LAUNCHER_NAME"
echo "Desktop launcher '$DESKTOP_LAUNCHER_NAME' created in $PROJECT_PATH."
echo "You may need to move it to ~/.local/share/applications/ or /usr/share/applications/ for it to appear in your application menu."
echo "You might also need to right-click it and select 'Allow Launching'."


echo ""
echo "Installation process complete."
echo "To run the application:"
echo "1. Open a new terminal."
echo "2. Navigate to $PROJECT_PATH"
echo "3. Run: ./$LAUNCHER_SCRIPT_NAME"
echo "Alternatively, try using the $DESKTOP_LAUNCHER_NAME file (you might need to move it and/or mark it as trusted)."

exit 0
