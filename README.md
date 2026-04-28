# FunGen

FunGen is a Python-based tool that uses AI to generate Funscript files from VR and 2D POV videos. It enables fully automated funscript creation for individual scenes or entire folders of videos.

Join the **Discord community** for discussions and support: [Discord Community](https://discord.gg/WYkjMbtCZA)

---

### DISCLAIMER

This project is still at the early stages of development. It is not intended for commercial use. Please, do not use this project for any commercial purposes without prior consent from the author. It is for individual use only.

---

## v1.0.0 Highlights

- **One-shim installer (uv + venv replaces miniconda)**. Download a single `install.bat` / `install.sh`, double-click, done. The shim bootstraps `uv`, builds a self-contained `.venv`, auto-detects your GPU, and writes launcher scripts. ~500 MB on disk, no admin rights, no PATH surgery. `ffmpeg` and `mpv` are auto-installed via the OS package manager (winget, brew, apt/dnf/pacman) when missing.
- **Six PyTorch channels, auto-selected**: `cuda_blackwell` (RTX 50-series, cu129), `cuda_stable` (RTX 20/30/40, cu128), `cuda_legacy` (driver 525-559, cu124), `cpu`, `mps` (Apple Silicon), `rocm` (AMD on Linux). Detection runs in `install.py`; you can override by re-running with the channel name.
- **VR dewarp shader with adaptive supersample**. Runtime-compiled GLSL replaces the CPU `v360` filter for in-GUI playback. Adaptive resolution scales the shader FBO to display * supersample, with anisotropic filter cap and free IGN dither. Embedded fullscreen keeps the shader and adaptive quality active. Plain non-shader playback stays clamped to a sensible CPU budget.
- **GUI perf sweep**. Timeline draws via `rect_filled` instead of `circle_filled` (~2x cheaper); oscillation grid activation vectorized (1.4x); plugin runtimes fixed at the algorithm level (Resample 8.4x via `math.cos` in the scalar loop, Keyframes 5.2x and Dynamic Amplify 3.1x via `O(n log n)` flat arrays); cached u32 colors / chapter text widths / spline math throughout the draw loop; LOD-A density envelope dropped (zero CPU saving in bench).
- **Async tracker lifecycle**. YOLO model preloads off the UI thread; `stop_tracking` tears down asynchronously; post-session funscript save + autotune is async; mpv pause/resume is balanced across stop / display-mode reload; mpv `hwdec` defaults to `auto-safe`.
- **Animated splash with 17 themes**. Random per launch (or pin one with `FUNGEN_SPLASH_THEME=<name>`). Themes: matrix, terminator, tron, starwars, breaking, invaders, mars, clippy, tetris, pacman, blade, bsod, sonic, xfiles, tmnt, et, mario.
- **Cock Hero Beat Tracker (offline)**. Audio-beat-driven funscript generator. Picks beats from the audio track and emits alternating peak/valley keyframes - useful for music-video edits where visual flow alone is unreliable.
- **`--watch` actually processes videos**. The watch-folder CLI now spawns `main.py` workers per queued item, up to `--max-parallel N` (default 1), reaps on exit, terminates inflight on Ctrl-C. Previously the queue filled forever with nothing draining it.
- **Async navigation**. Arrow-key seeks fetch via a dedicated worker; tooltip dict refs are captured before async hover-cancel; scrub cache keyed by requested frame index avoids respawning the FFmpeg source on hover-seek.
- **Internal restructure (no behavior change)**. `app_logic` split into 8 lifecycle modules (`tracking_lifecycle`, `project_lifecycle`, `settings_lifecycle`, `video_session`, `first_run_setup`, `hardware_accel`, `log_config`, `shortcut_mapper`); video display split into `_core` / `controls` / `display_route` / `overlays`; gui components reorganized.

## v0.9.0 Highlights

- **Video backend rewrite** - PyAV is gone. Frame decode runs through a dedicated FFmpeg subprocess frame source; the GUI display uses libmpv via its render API for smooth playback. Each video-touching subsystem (thumbnails, proxy encode, metadata probe, audio) is a purpose-built FFmpeg or ffprobe path, not a shared in-process filter graph.
- **New nav buffer** - 1 GB byte-budgeted LRU frame cache replaces the old contiguous deque. Survives seeks, so bouncing between regions of the video reuses previously decoded frames for free until the byte budget forces LRU eviction. Hit-rate and fill percentage visible in the Expert -> Developer Perf panel.
- **Anticipatory prefetcher** - While paused and idle, a background thread watches the arrow-nav pattern and warms the cache around the likely next target (bidirectional fill on "landed", forward/backward fill on sustained trend). Gated off during playback (the loop pumps the cache itself) and during tracking (tracker owns the decoder).
- **Progressive arrow-hold playback** - Tap right arrow = one frame forward. Hold >= 0.25 s = REALTIME playback. Hold >= 3 s = MAX_SPEED. Release stops playback. Left arrow = one frame back on tap, auto-repeat step-back on hold (engine has no reverse playback).
- **Faster texture upload** - `GL_BGR` native upload eliminates the per-frame `cv2.cvtColor` allocation; PBO-backed streaming with `glBufferData` orphaning lets the driver overlap the DMA copy with the rest of the render pass. Automatic fallback to direct upload if PBO fails.
- **Better diagnostics** - FFmpeg subprocess deaths now surface the return code and stderr tail so misconfigured hwaccel or filter chains are identifiable from the log. MpvDisplay load failures surface to the user via toast + expert panel instead of showing a silent black frame. Expert panel exposes a "Debug nav logging" toggle that emits a one-line NAV trace per arrow/scrub event.

## v0.8.0 Highlights

- **Simplified GUI** - Removed Simple Mode (Run tab is now clean enough for everyone). Control panel reduced to Run + Metadata. Settings, Undo, and Performance tabs moved to the Info panel on the right
- **Toast Notifications** - Non-blocking popup notifications for saves, errors, and plugin results. Replaces old modal dialogs and status bar spam
- **Ultimate Autotune Popup** - Opens with parameter sliders and live preview overlay. Adjust settings and see the result before applying
- **Streamlined Menus** - Flattened View menu, added shortcut hints throughout, removed unused gauges and movement bar. All display toggles show their keyboard shortcuts
- **Cleaner Tracker Settings** - Stripped broken/dead settings from live trackers. Only working, user-relevant controls exposed
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
- **Device Control and VR Streaming Add-ons** - OSR/Buttplug hardware control, Quest 3 streaming (available at paypal.me/k00gar)
- **Batch Processing** - Process entire folders (available as monthly PayPal add-on)

---

## Quick Installation (Recommended)

**One installer for every platform. No miniconda, no admin rights, ~500 MB on disk.**

### Windows
1. Download: [install.bat](https://raw.githubusercontent.com/ack00gar/FunGen-AI-Powered-Funscript-Generator/main/install.bat)
2. Double-click to run.

### Linux / macOS
```bash
curl -fsSL https://raw.githubusercontent.com/ack00gar/FunGen-AI-Powered-Funscript-Generator/main/install.sh | bash
```

The installer:
- Installs [uv](https://docs.astral.sh/uv/) once (~15 MB) if you don't already have it
- Creates a self-contained `.venv` next to FunGen with Python 3.11
- Detects your GPU (NVIDIA Blackwell / NVIDIA stable / AMD ROCm / Apple MPS / CPU) and installs the matching PyTorch wheels
- Creates launcher scripts: `launch.bat` (Windows), `launch.command` (macOS, Finder), `launch.sh` (Linux / macOS Terminal)
- Note: `ffmpeg` and `libmpv` are system packages, install them via your OS package manager (the Windows launcher bundles them)

If a previous FunGen install used miniconda, the installer detects it at the end and asks (once) whether to clean it up. It never touches conda silently and never removes other conda envs.

**That's it.** Double-click the launcher for your OS to start FunGen.

### Migrating from a previous miniconda install

Just `git pull` (or use the in-app updater) and click the launcher again. The launcher self-heals: it sees there's no `.venv`, runs `install.py` once (~2 min), then starts the app. The old `~/miniconda3/envs/FunGen` is left in place until you confirm the new env works. You will get a one-time prompt at the end of the installer asking what to do with it.

---

## Manual Installation

If the automatic installer doesn't fit your setup, you can do it by hand.

### Prerequisites

- Any Python 3.x on PATH (uv will install Python 3.11 itself if you don't have it)
- `git` (https://git-scm.com/downloads/, or `winget install --id Git.Git -e --source winget` on Windows)
- `ffmpeg` + `ffprobe` on PATH (https://www.ffmpeg.org/download.html, `brew install ffmpeg`, `apt install ffmpeg`)
- `libmpv` for in-GUI video playback (Homebrew: `brew install mpv`, Debian/Ubuntu: `sudo apt install libmpv-dev`, Windows: bundled with the launcher)

### Install
```bash
git clone --branch main https://github.com/ack00gar/FunGen-AI-Powered-Funscript-Generator.git FunGen
cd FunGen
./install.sh        # macOS / Linux
install.bat         # Windows
```

The shim bootstraps `uv` if needed, then runs `install.py` with `uv run --no-project --python 3.11`. `install.py` is stdlib-only, picks the right requirements file from `requirements/` based on your GPU, and installs into `.venv/` via uv.

### Requirements files (one per channel)

| File | When |
|---|---|
| `requirements/base.txt` | Always installed: torch-independent deps (opencv, ultralytics, moderngl, etc.) |
| `requirements/cuda_stable.txt` | NVIDIA RTX 20/30/40-series, A/H/L-series datacenter cards (cu128, driver 555+) |
| `requirements/cuda_blackwell.txt` | NVIDIA RTX 50-series (RTX 5070/5080/5090), Blackwell (cu129, driver 560+) |
| `requirements/cuda_legacy.txt` | Older NVIDIA drivers in the 525-559 range (cu124, torch 2.6.0) |
| `requirements/cpu.txt` | Linux + Windows CPU-only |
| `requirements/mps.txt` | macOS Apple Silicon (MPS / Metal) |
| `requirements/rocm.txt` | AMD ROCm on Linux |

If the GPU detection picks the wrong file, you can override it by running:
```bash
.venv/bin/python -m pip install -r requirements/base.txt -r requirements/<channel>.txt
```
(replace `bin/python` with `Scripts/python.exe` on Windows).

**NVIDIA 10xx-series GPUs are not supported.** ROCm is Linux-only.

### Verify the install (NVIDIA only)
```bash
nvidia-smi
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
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
- **Driver too old for installed CUDA wheels**: NVIDIA Studio Driver 555+ is recommended for cu128 (RTX 20/30/40-series); 560+ for cu129 (RTX 50-series Blackwell).
- **PATH issues**: The system CUDA toolkit is not required (the torch wheels ship their own CUDA libs). Just having `nvidia-smi` work is enough.
- **Right channel?** The installer auto-detects, but you can verify by checking `.venv/bin/python -c "import torch; print(torch.version.cuda)"` matches expectations (12.8 for stable, 12.9 for Blackwell).

**Verification Commands:**
```bash
nvidia-smi
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```
On Windows replace `.venv/bin/python` with `.venv\Scripts\python.exe`.

## GUI

FunGen launches with a streamlined interface. The control panel (left) has Run and Metadata tabs, plus add-on tabs for Device Control, Streamer, and Batch Processing. The info panel (right) has Info, Settings, Undo, and Performance tabs. All settings are searchable from the Settings tab. Use View > Show Advanced Options to reveal developer controls.

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
- **Cock Hero Beat Tracker** - Audio-beat-driven script generator. Detects beats in the audio track and emits alternating peak/valley keyframes; works well for music-video edits where visual flow alone is unreliable.

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
- The optimal number of instances depends on a combination of factors, including your CPU, GPU, RAM, and system configuration. So experiment with different setups to find the ideal configuration for your hardware!

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

This project started as a dream to automate Funscript generation for VR videos. Here's a brief history of its development:

- **Initial Approach (OpenCV Trackers)**: The first version relied on OpenCV trackers to detect and track objects in the video. While functional, the approach was slow (8–20 FPS) and struggled with occlusions and complex scenes.
- **Transition to YOLO**: To improve accuracy and speed, the project shifted to using YOLO object detection. A custom YOLO model was trained on a dataset of 1000nds annotated VR video frames, significantly improving detection quality.
- **v0.9.0 Video Backend Rewrite**: PyAV and its in-process libav filter graph were replaced by a purpose-built FFmpeg subprocess frame source for analysis and a libmpv render-API display for playback. Removed a major source of C-level crashes, cut cold-start latency, and unblocked the GPU texture-upload and nav-cache improvements shipped in the same release.
- **v1.0.0 Installer + Render Pipeline**: miniconda was replaced by `uv + .venv` so a fresh install is one click instead of a multi-step toolchain bring-up. The VR display path moved from CPU `v360` to a runtime-compiled GLSL dewarp shader with adaptive supersampling, and the GUI perf path was swept (timeline draw, oscillation grid, plugin algorithms, async tracker lifecycle).
- **Original Post**: For more details and discussions, check out the original post on EroScripts:
  [VR Funscript Generation Helper (Python + CV/AI)](https://discuss.eroscripts.com/t/vr-funscript-generation-helper-python-now-cv-ai/202554)

---

# License

This project is licensed under the **Non-Commercial License**. You are free to use the software for personal, non-commercial purposes only. Commercial use, redistribution, or modification for commercial purposes is strictly prohibited without explicit permission from the copyright holder.

This project is not intended for commercial use, nor for generating and distributing in a commercial environment.

For commercial use, please contact me.

See the [LICENSE](LICENSE) file for full details.

---

# Acknowledgments

- **YOLO** - Ultralytics for the detection framework.
- **FFmpeg / libavfilter** - decode, v360 dewarp, proxy encoding, audio, thumbnails.
- **libmpv** - smooth in-GUI playback via the render API.
- **uv** - the Python installer / venv builder that replaced miniconda in v1.0.0.
- **Eroscripts Community** - inspiration and real-world use cases.

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
3. Use the launcher script (`launch.bat` on Windows, `launch.command` on macOS, `launch.sh` on Linux) instead of `python main.py`

### FFmpeg/FFprobe Not Found

If you get "ffmpeg/ffprobe not found" errors:

1. **Use the launcher script** (`launch.bat` on Windows, `launch.command` on macOS, `launch.sh` on Linux) instead of running `python main.py` directly
2. **Rerun the installer** if FFmpeg is not on your system PATH (the launcher relies on system PATH to find it; install via `brew install ffmpeg` on macOS, `apt install ffmpeg` on Debian/Ubuntu, or [ffmpeg.org](https://www.ffmpeg.org/download.html) on Windows)

### libmpv not found

If the log shows `libmpv not available; MpvDisplay disabled` and video playback stutters:

- **macOS**: `brew install mpv`
- **Linux (Debian/Ubuntu)**: `sudo apt install libmpv2` (or `libmpv1` on older distros)
- **Windows**: reinstall using the installer, which bundles libmpv DLLs

Analysis, batch, and CLI modes do not require libmpv; only the in-GUI smooth-playback path does. Without it, the app falls back to CPU texture uploads per frame.

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

Support the project on **PayPal** (one-time add-on purchases or monthly subscription).

---
