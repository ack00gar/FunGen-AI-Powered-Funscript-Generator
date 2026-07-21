# FunGen 1 — version history

Release notes for the legacy Python FunGen (v0.6.0 → v1.0.0). FunGen 1 is no
longer developed — see [fungen.app](https://fungen.app) for FunGen 2.

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

