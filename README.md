# FunGen

FunGen is a Python-based tool that uses AI to generate Funscript files from VR and 2D POV videos. It enables fully automated funscript creation for individual scenes or entire folders of videos.

Join the **Discord community** for discussions and support: [Discord Community](https://discord.gg/WYkjMbtCZA)

---

### DISCLAIMER

This project is still at the early stages of development. It is not intended for commercial use. Please, do not use this project for any commercial purposes without prior consent from the author. It is for individual use only.

---

## v0.8.0 Highlights

- **Simplified GUI** - Removed Simple Mode (Run tab is now clean enough for everyone). Control panel reduced to Run + Metadata. Settings, Undo, and Performance tabs moved to the Info panel on the right
- **Toast Notifications** - Non-blocking popup notifications for saves, errors, and plugin results. Replaces old modal dialogs and status bar spam
- **Ultimate Autotune Popup** - Opens with parameter sliders and live preview overlay. Adjust settings and see the result before applying
- **Streamlined Menus** - Flattened View menu, added shortcut hints throughout, removed unused gauges and movement bar. All display toggles show their keyboard shortcuts
- **Cleaner Tracker Settings** - Stripped broken/dead settings from live trackers. Only working, user-relevant controls exposed
- **Add-on Terminology** - Updated from "Supporter" to ko-fi add-on language with correct purchase URLs
- **First-run Wizard** - Reduced from 6 to 5 steps (no mode selection needed)
- **Model Download Button** - Re-download AI models anytime from Settings > AI Models
- **Auto-populated Metadata** - Creator and Title fields auto-fill from FunGen version and video filename

## v0.7.5 Highlights

- **VR Hybrid Chapter-Aware Tracker** - New offline tracker combining sparse YOLO chapter detection with per-chapter ROI optical flow
- **Preprocessed Video Infrastructure** - Hardware-accelerated encoding, automatic reuse on re-run
- **Batch Mode Preprocessed Video** - Opt-in setting for faster re-runs in batch processing

## v0.6.0 Highlights

- **Multi-Axis Funscript Support** - OFS-compatible axis system (stroke, roll, pitch, surge, sway, twist)
- **14+ Built-in Filter Plugins** - Ultimate Autotune, RDP Simplify, Savitzky-Golay, and more
- **Device Control and VR Streaming Add-ons** - OSR/Buttplug hardware control, HereSphere/Quest 3 streaming (available at ko-fi.com/k00gar)
- **Batch Processing** - Process entire folders (available as monthly Ko-fi add-on)

---

## Quick Installation (Recommended)

**Automatic installer that handles everything for you:**

### Windows
1. Download: [install.bat](https://raw.githubusercontent.com/ack00gar/FunGen-AI-Powered-Funscript-Generator/main/install.bat)
2. Double-click to run (or run from command prompt)
3. Wait for automatic installation of Python, Git, FFmpeg, and FunGen

### Linux/macOS
```bash
curl -fsSL https://raw.githubusercontent.com/ack00gar/FunGen-AI-Powered-Funscript-Generator/main/install.sh | bash
```

The installer automatically:
- Installs Python 3.11 (Miniconda)
- Installs Git and FFmpeg/FFprobe  
- Downloads and sets up FunGen AI
- Installs all required dependencies
- Creates launcher scripts for easy startup
- Detects your GPU and optimizes PyTorch installation

**That's it!** The installer creates launch scripts - just run them to start FunGen.

---

## Manual Installation

If you prefer manual installation or need custom configuration:

### Prerequisites

Before using this project, ensure you have the following installed:

- **Git** https://git-scm.com/downloads/ or 'winget install --id Git.Git -e --source winget' from a command prompt for Windows users as described below for easy install of Miniconda.
- **FFmpeg** added to your PATH or specified under the settings menu (https://www.ffmpeg.org/download.html)
- **Miniconda** (https://www.anaconda.com/docs/getting-started/miniconda/install)

Easy install of Miniconda for Windows users:
Open Command Prompt and run: `winget install -e --id Anaconda.Miniconda3`

### Start a miniconda command prompt
After installing Miniconda look for a program called "Anaconda prompt (miniconda3)" in the start menu (on Windows) and open it

### Create the necessary miniconda environment and activate it
```bash
conda create -n VRFunAIGen python=3.11
conda activate VRFunAIGen
```
- Please note that any pip or python commands related to this project must be run from within the VRFunAIGen virtual environment.

### Clone the repository
Open a command prompt and navigate to the folder where you'd like FunGen to be located. For example, if you want it in C:\FunGen, navigate to C:\ ('cd C:\'). Then run
```bash
git clone --branch main https://github.com/ack00gar/FunGen-AI-Powered-Funscript-Generator.git FunGen
cd FunGen
```

### Install the core python requirements
```bash
pip install -r requirements/core.requirements.txt
```

### NVIDIA GPU Setup (CUDA Required)

**Quick Setup:**
1. **Install NVIDIA Drivers**: [Download here](https://www.nvidia.com/Download/index.aspx)
2. **Install CUDA 12.8**: [Download here](https://developer.nvidia.com/cuda-downloads)
3. **Install cuDNN for CUDA 12.8**: [Download here](https://developer.nvidia.com/cudnn) (requires free NVIDIA account)

**Install Python Packages:**

**For 20xx, 30xx and 40xx-series NVIDIA GPUs:**
```bash
pip install -r requirements/cuda.requirements.txt
pip install tensorrt
```

**For 50xx series NVIDIA GPUs (RTX 5070, 5080, 5090):**
```bash
pip install -r requirements/cuda.50series.requirements.txt
pip install tensorrt
```

**Note:** NVIDIA 10xx series GPUs are not supported.

**Verify Installation:**
```bash
nvidia-smi                    # Check GPU and driver
nvcc --version               # Check CUDA version  
python -c "import torch; print(torch.cuda.is_available())"  # Check PyTorch CUDA
python -c "import torch; print(torch.backends.cudnn.is_available())"  # Check cuDNN
```

### If your GPU doesn't support cuda
```bash
pip install -r requirements/cpu.requirements.txt
```

### AMD GPU acceleration (ROCm for Linux Only)
ROCm is supported for AMD GPUs on Linux. To install the required packages, run:
```bash
pip install -r requirements/rocm.requirements.txt
```

## Download the YOLO models

The necessary YOLO models will be automatically downloaded on the first startup. If you want to use a specific model, you can download it from our Discord and place it in the `models/` sub-directory. If you aren't sure, you can add all the models and let the app decide the best option for you.

### Start the app
```bash
python main.py
```

We support multiple model formats across Windows, macOS, and Linux.

### Recommendations
- NVIDIA Cards: we recommend the .engine model
- AMD Cards: we recommend .pt (requires ROCm see below)
- Mac: we recommend .mlmodel

### Models
- **.pt (PyTorch)**: Requires CUDA (for NVIDIA GPUs) or ROCm (for AMD GPUs) for acceleration.
- **.onnx (ONNX Runtime)**: Best for CPU users as it offers broad compatibility and efficiency.
- **.engine (TensorRT)**: For NVIDIA GPUs: Provides very significant efficiency improvements (this file needs to be build by running "Generate TensorRT.bat" after adding the base ".pt" model to the models directory)
- **.mlpackage (Core ML)**: Optimized for macOS users. Runs efficiently on Apple devices with Core ML.

In most cases, the app will automatically detect the best model from your models directory at launch, but if the right model wasn't present at this time or the right dependencies where not installed, you might need to override it under settings. The same applies when we release a new version of the model.


### Troubleshooting CUDA Installation

**Common Issues:**
- **Driver version mismatch**: Ensure NVIDIA drivers are compatible with your CUDA version
- **PATH issues**: Make sure CUDA bin directory is in your system PATH
- **Version conflicts**: Ensure all components (driver, CUDA, cuDNN, PyTorch) are compatible versions

**Verification Commands:**
```bash
nvidia-smi                    # Check GPU and driver
nvcc --version               # Check CUDA version  
python -c "import torch; print(torch.cuda.is_available())"  # Check PyTorch CUDA
python -c "import torch; print(torch.backends.cudnn.is_available())"  # Check cuDNN
```

## GUI

FunGen launches with a streamlined interface. The control panel (left) has Run and Metadata tabs, plus add-on tabs for Device Control, Streamer, and Batch Processing. The info panel (right) has Info, Settings, Undo, and Performance tabs. All settings are searchable from the Settings tab. Use View > Show Advanced Options to reveal developer controls.

-----

## GitHub Token Setup (Optional)

FunGen includes an update system that allows you to download and switch between different versions of the application. To use this feature, you'll need to set up a GitHub Personal Access Token. This is optional and only required for the update functionality.

### Why a GitHub Token?

GitHub's API has rate limits:
- **Without a token**: 60 requests per hour
- **With a token**: 5,000 requests per hour

This allows FunGen to fetch commit information, changelogs, and version data without hitting rate limits.

### How to Get a GitHub Token

1. **Go to GitHub Settings**:
   - Visit [GitHub Settings](https://github.com/settings)
   - Sign in to your GitHub account

2. **Navigate to Developer Settings**:
   - Click your GitHub avatar (top right) → "Settings"
   - Scroll down to the bottom left of the Settings page
   - Click "Developer settings" in the left menu list

3. **Create a Personal Access Token**:
   - Click "Personal access tokens" → "Tokens (classic)"
   - Click "Generate new token" → "Generate new token (classic)"

4. **Confirm Access**
   - If you created a 2FA you will be prompted to eter it
   - If you have _not_ yet created a 2FA you will be prompted to do so

5. **Configure the Token**:
   - **Note**: Give it a descriptive name like "FunGen Updates"
   - **Expiration**: Choose an appropriate expiration (30 days, 60 days, etc.)
   - **Scopes**: Select only these scopes:
     - `public_repo` (to read public repository information)
     - `read:user` (to read your user information for validation)

6. **Generate and Copy**:
   - Click "Generate token"
   - **Important**: Copy the token immediately - you won't be able to see it again!

### Setting the Token in FunGen

1. **Open FunGen** and go to the **Updates** menu
2. **Click "Select Update Commit"**
3. **Go to the "GitHub Token" tab**
4. **Paste your token** in the text field
5. **Click "Test Token"** to verify it works
6. **Click "Save Token"** to store it

### What the Token is Used For

The GitHub token enables these features in FunGen:
- **Version Selection**: Browse and download specific commits from the `main` branch
- **Changelog Display**: View detailed changes between versions
- **Update Notifications**: Check for new versions and updates
- **Rate Limit Management**: Avoid hitting GitHub's API rate limits

### Security Notes

- The token is stored locally in `github_token.ini`
- Only `public_repo` and `read:user` permissions are required
- The token is used only for reading public repository data
- You can revoke the token anytime from your GitHub settings

-----

# Command Line Usage

FunGen can be run in two modes: a graphical user interface (GUI) or a command-line interface (CLI) for automation and batch processing.

**To start the GUI**, simply run the script without any arguments:

```bash
python main.py
```

**To use the CLI mode**, you must provide an input path to a video or a folder.

### CLI Examples

**To generate a script for a single video with default settings:**

```bash
python main.py "/path/to/your/video.mp4"
```

**To process an entire folder of videos recursively using a specific mode and overwrite existing funscripts:**

```bash
python main.py "/path/to/your/folder" --mode <your_mode> --overwrite --recursive
```

**To run multiple instances on different GPUs (e.g. 10-bit on QSV, rest on CUDA):**

```bash
python main.py "/path/to/10bit_videos" --hwaccel qsv &
python main.py "/path/to/other_videos" --hwaccel cuda &
```

### Command-Line Arguments

| Argument | Short | Description |
|---|---|---|
| `input_path` | | **Required for CLI mode.** Path to a single video file or a folder containing videos. |
| `--mode` | | Sets the processing mode. The available modes are discovered dynamically. |
| `--overwrite`| | Forces the app to re-process and overwrite any existing funscripts. By default, it skips videos that already have a funscript. |
| `--no-autotune`| | Disables the automatic application of Ultimate Autotune after generation. |
| `--no-copy` | | Prevents saving a copy of the final funscript next to the video file. It will only be saved in the application's output folder. |
| `--generate-roll` | | Generates a secondary axis funscript file (e.g. `.roll.funscript`) for supported multi-axis devices. |
| `--save-preprocessed` | | Keeps the preprocessed (resized/unwarped) video for each processed file. Off by default in batch/CLI to save disk space. |
| `--hwaccel` | | Override hardware acceleration method for this run (e.g. `cuda`, `qsv`, `auto`, `none`). Useful for running multiple instances on different GPUs. |
| `--recursive`| `-r` | If the input path is a folder, this flag enables scanning for videos in all its subdirectories. |

---

# Modular Systems

FunGen features a modular architecture for both funscript filtering and motion tracking, allowing for easy extension and customization.

## Filter Plugin System

Plugins are accessible from the **Plugins** dropdown in the timeline toolbar. Each plugin opens a popup with adjustable parameters and live preview. Available plugins:

- **Amplify:** Amplifies or reduces position values around a center point.
- **Autotune SG:** Automatically finds optimal Savitzky-Golay filter parameters.
- **Clamp:** Clamps all positions to a specific value.
- **Invert:** Inverts position values (0 becomes 100, etc.).
- **Keyframes:** Simplifies the script to significant peaks and valleys.
- **Resample:** Resamples the funscript at regular intervals while preserving peak timing.
- **Simplify (RDP):** Simplifies the funscript by removing redundant points using the RDP algorithm.
- **Smooth (SG):** Applies a Savitzky-Golay smoothing filter.
- **Speed Limiter:** Limits speed and adds vibrations for hardware device compatibility.
- **Threshold Clamp:** Clamps positions to 0/100 based on thresholds.
- **Ultimate Autotune:** Comprehensive 8-stage enhancement pipeline with live preview.

## Tracking System

The tracker system is responsible for analyzing the video and generating the raw motion data. Trackers are organized into categories based on their functionality.

### Offline Trackers (Recommended)

- **VR Hybrid Chapter-Aware** - Single-pass chapter detection + per-chapter ROI optical flow. Best quality for VR videos.
- **Contact Analysis (2-Stage)** - YOLO-based contact detection and analysis.
- **Guided Flow (3-Stage)** - Chapter-aware dense optical flow with per-position ROI strategies.

### Live Trackers

- **2D POV and VR Hybrid Flow** - YOLO ROI detection with DIS optical flow. Dual axis (stroke + roll).
- **Oscillation Detector** - Grid-based motion detection with decay mechanism.
- **YOLO ROI Tracker** - Automatic ROI detection with optical flow.
- **User ROI Tracker** - Manual ROI definition with sub-tracking.

### Community Trackers

Community trackers are auto-discovered from the `tracker/tracker_modules/community/` folder. See the example tracker for how to create your own.

---

# Performance & Parallel Processing

Our pipeline's current bottleneck lies in the Python code within YOLO.track (the object detection library we use), which is challenging to parallelize effectively in a single process.

However, when you have high-performance hardware you can use the command line (see above) to processes multiple videos simultaneously. Alternatively you can launch multiple instances of the GUI.

We tested speeds of about 60 to 110 fps for 8k 8bit vr videos when running a single process. Which translates to faster then realtime processing already. However, running in parallel mode we tested
speeds of about 160 to 190 frames per second (for object detection). Meaning processing times of about 20 to 30 minutes for 8bit 8k VR videos for the complete process. More then twice the speed of realtime!

Keep in mind your results may vary as this is very dependent on your hardware. Cuda capable cards will have an advantage here. However, since the pipeline is largely CPU and video decode bottlenecked
a top of the line card like the 4090 is not required to get similar results. Having enough VRAM to run 3-6 processes, paired with a good CPU, will speed things up considerably though.

**Important considerations:**

- Each instance requires the YOLO model to load which means you'll need to keep checks on your VRAM to see how many you can load.
- The optimal number of instances depends on a combination of factors, including your CPU, GPU, RAM, and system configuration. So experiment with different setups to find the ideal configuration for your hardware! 😊

---

# Miscellaneous

- For VR only sbs (side by side) **Fisheye** and **Equirectangular** 180° videos are supported at the moment
- 2D POV videos are supported but work best when they are centered properly
- 2D / VR is automatically detected as is fisheye / equirectangular and FOV (make sure you keep the file format information in the filename _FISHEYE190, _MKX200, _LR_180, etc.)
- Detection settings can also be overwritten in the UI if the app doesn't detect it properly

---

# Output Files

FunGen generates the following files in a dedicated subfolder within your output directory:

- **`.funscript`** - The final funscript file for the primary (stroke) axis
- **`.roll.funscript` / `.twist.funscript`** - Secondary axis funscript (if dual-axis tracker is used)
- **`_t1_raw.funscript`** - Raw unprocessed funscript before any post-processing
- **`_preprocessed.mkv`** - Preprocessed video for faster re-runs (optional, off by default)
- **`.fgnproj`** - FunGen project file containing settings, chapters, and metadata

-----

# About the project

## Pipeline Overview

Each tracker implements its own pipeline. The VR Hybrid tracker (recommended) works as follows:

1.  **Chapter Detection** - Sparse YOLO detection at 2fps classifies the video into chapters (cowgirl, missionary, blowjob, etc.)
2.  **Per-Chapter Analysis** - Dense YOLO + ROI optical flow per chapter, with position-specific amplitude targets
3.  **Funscript Generation** - Motion signal smoothing, peak detection, and keyframe extraction
4.  **Optional Post-Processing** - Apply Ultimate Autotune or individual plugins from the timeline's Plugins menu

## Project Genesis and Evolution

This project started as a dream to automate Funscript generation for VR videos. Here’s a brief history of its development:

- **Initial Approach (OpenCV Trackers)**: The first version relied on OpenCV trackers to detect and track objects in the video. While functional, the approach was slow (8–20 FPS) and struggled with occlusions and complex scenes.
- **Transition to YOLO**: To improve accuracy and speed, the project shifted to using YOLO object detection. A custom YOLO model was trained on a dataset of 1000nds annotated VR video frames, significantly improving detection quality.
- **Original Post**: For more details and discussions, check out the original post on EroScripts:
  [VR Funscript Generation Helper (Python + CV/AI)](https://discuss.eroscripts.com/t/vr-funscript-generation-helper-python-now-cv-ai/202554)

----

# Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes.
4. Submit a pull request.

---

# License

This project is licensed under the **Non-Commercial License**. You are free to use the software for personal, non-commercial purposes only. Commercial use, redistribution, or modification for commercial purposes is strictly prohibited without explicit permission from the copyright holder.

This project is not intended for commercial use, nor for generating and distributing in a commercial environment.

For commercial use, please contact me.

See the [LICENSE](LICENSE) file for full details.

---

# Acknowledgments

- **YOLO**: Thanks to the Ultralytics team for the YOLO implementation.
- **FFmpeg**: For video processing capabilities.
- **Eroscripts Community**: For the inspiration and use cases.

---

# Troubleshooting

## Installation Issues

### "unknown@unknown" or Git Permission Errors

If you see `[unknown@unknown]` in the application logs or git errors like "returned non-zero exit status 128":

**Cause:** The installer was run with administrator privileges, causing git permission/ownership issues.

**Solution 1 - Fix git permissions:**
```cmd
cd "C:\path\to\your\FunGen\FunGen"
git config --add safe.directory .
```

**Solution 2 - Reinstall as normal user:**
1. Redownload `install.bat`
2. Run it as a **normal user** (NOT as administrator)
3. Use the launcher script created by the installer instead of `python main.py`

### FFmpeg/FFprobe Not Found

If you get "ffmpeg/ffprobe not found" errors:

1. **Use the launcher script** (`launch.bat` or `launch.sh`) instead of running `python main.py` directly
2. **Rerun the installer** to get updated launcher scripts with FFmpeg PATH fixes
3. The launcher automatically adds FFmpeg to PATH

### General Installation Problems

1. **Always use launcher scripts** - Don't run `python main.py` directly
2. **Run installer as normal user** - Avoid administrator mode
3. **Rerun installer for updates** - Get latest fixes by rerunning the installer
4. **Check working directory** - Make sure you're in the FunGen project folder

---

# Support

If you encounter any issues or have questions, please open an issue on GitHub.

Join the **Discord community** for discussions and support:
[Discord Community](https://discord.gg/WYkjMbtCZA)

Support the project on **Ko-fi** (one-time add-on purchases) or **Patreon** (monthly supporter features).

---