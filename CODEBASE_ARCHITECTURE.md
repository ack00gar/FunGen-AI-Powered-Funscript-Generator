# FunGen Codebase Architecture Overview

## Executive Summary

FunGen is a comprehensive Python application for automated funscript generation from videos using AI. The codebase consists of **~94,000 lines of Python code** across **157 files**, organized into well-defined modules with clear separation of concerns.

**Key Statistics:**
- Total Python Code: 93,761 lines
- Python Files: 157
- Total Size: 4.14 MB (source code)
- Main Language: Python (GUI via ImGui)
- Entry Point: `main.py`
- Architecture: Modular with specialized domains

---

## 1. OVERALL DIRECTORY AND FILE STRUCTURE

### Root-Level Organization

```
FunGen-AI-Powered-Funscript-Generator/
├── main.py                          # Application entry point (CLI/GUI dispatcher)
├── core.requirements.txt            # Core Python dependencies
├── cuda.requirements.txt            # NVIDIA CUDA-specific dependencies
├── rocm.requirements.txt            # AMD ROCm dependencies
├── macos.requirements.txt           # macOS-specific dependencies
├── cpu.requirements.txt             # CPU-only dependencies
├── environment.yml                  # Conda environment configuration
│
├── application/                     # Main application logic (49,818 lines, 75 files)
│   ├── logic/                       # Business logic & processing
│   ├── gui_components/              # UI rendering and components
│   ├── classes/                     # Model classes and data structures
│   ├── utils/                       # Utility functions
│   └── gpu_rendering/               # GPU acceleration code
│
├── tracker/                         # Tracker system (20,101 lines, 30 files)
│   ├── tracker_modules/             # Modular tracker implementations
│   │   ├── core/                    # Base tracker classes & security
│   │   ├── live/                    # Live/real-time trackers
│   │   ├── offline/                 # Offline processing trackers
│   │   ├── experimental/            # Experimental tracker variants
│   │   ├── helpers/                 # Visualization & signal processing
│   │   ├── templates/               # Community tracker templates
│   │   └── community/               # Community-contributed trackers
│   └── tracker_manager.py           # Tracker orchestration (983 lines)
│
├── funscript/                       # Funscript generation & processing (6,955 lines, 21 files)
│   ├── plugins/                     # Filter/transformation plugins
│   ├── user_plugins/                # User-defined plugin templates
│   ├── dual_axis_funscript.py       # Dual-axis support (1,352 lines)
│   └── [plugins]/                   # Individual filters (autotune, clamp, etc.)
│
├── detection/                       # Object detection & contour tracking (6,937 lines, 10 files)
│   └── cd/                          # Contact Detection pipeline
│       ├── stage_1_cd.py            # Initial contour detection (1,061 lines)
│       ├── stage_2_cd.py            # Contact analysis (2,698 lines)
│       ├── stage_3_of_processor.py   # Optical flow processing (596 lines)
│       ├── stage_3_mixed_processor.py # Mixed tracking (1,097 lines)
│       └── data_structures/         # Data models (segments, boxes, frames)
│
├── video/                           # Video processing (5,212 lines, 6 files)
│   ├── video_processor.py           # Main video processor (2,831 lines)
│   ├── gpu_unwarp_worker.py         # GPU-accelerated VR unwarp (1,035 lines)
│   ├── dual_frame_processor.py       # Dual-frame VR processing (596 lines)
│   ├── vr_format_detector_ml_real.py # VR format detection (498 lines)
│   └── thumbnail_extractor.py       # Video thumbnails (247 lines)
│
├── config/                          # Configuration & constants (1,664 lines, 6 files)
│   ├── constants.py                 # Application constants (657 lines)
│   ├── tracker_discovery.py         # Dynamic tracker discovery (368 lines)
│   ├── constants_colors.py          # Color definitions (265 lines)
│   ├── element_group_colors.py      # UI color scheme (259 lines)
│   └── theme_manager.py             # Theme management (106 lines)
│
├── common/                          # Shared utilities (487 lines, 5 files)
│   ├── exceptions.py                # Custom exceptions
│   ├── result.py                    # Result/response wrapper
│   ├── http_client_manager.py       # HTTP client management
│   ├── temp_manager.py              # Temporary file handling
│   └── __init__.py
│
└── assets/                          # Static resources
    ├── branding/                    # Logo, icons
    ├── splash/                      # Splash screen assets
    └── ui/                          # UI icons and buttons
```

---

## 2. MAIN ENTRY POINTS AND APPLICATION FLOW

### Entry Point: `main.py` (257 lines)

```python
main()
  ├── _setup_bootstrap_logger()           # Initialize logging
  ├── check_and_install_dependencies()    # Verify all requirements
  ├── set_start_method('spawn')           # Multiprocessing setup
  ├── parse_command_line_arguments()      # CLI argument parsing
  └── Decision Point:
      ├── If input_path provided → run_cli(args)
      └── Else → run_gui()
```

### GUI Mode Flow

```
run_gui()
  └── ApplicationLogic(is_cli=False)  [app_logic.py]
      ├── Initialize tracker discovery
      ├── Load application state (settings, projects)
      ├── Initialize audio/video processors
      ├── Create GPU pools
      └── GUI(app_logic=core_app)      [app_gui.py]
          ├── Initialize ImGui/GLFW
          ├── Create UI components:
          │   ├── control_panel_ui.py    [5,841 lines] - Main processing controls
          │   ├── video_display_ui.py    [2,391 lines] - Video rendering
          │   ├── video_navigation_ui.py [2,090 lines] - Timeline navigation
          │   ├── toolbar_ui.py          [988 lines]   - Toolbar & buttons
          │   └── Other UI panels...
          ├── Start preview/heatmap workers
          └── gui.run() → OpenGL event loop
```

### CLI Mode Flow

```
run_cli(args)
  └── ApplicationLogic.run_cli(args)
      ├── Determine mode (3-stage, live, etc.)
      ├── Load/parse input videos/funscripts
      ├── For each file:
      │   ├── Stage 1: Contact Detection      [stage_1_cd.py]
      │   ├── Stage 2: Contact Analysis      [stage_2_cd.py]
      │   ├── Stage 3: Tracking (OD/OF)     [stage_3_*.py]
      │   ├── Apply funscript filters
      │   └── Save output
      └── Report results
```

### Key Processing Stages

**Stage 1: Contact Detection**
- Input: Video frames
- Process: Contour detection, image analysis
- Output: Detected contact regions

**Stage 2: Contact Analysis**
- Input: Contact regions, frame history
- Process: Signal processing, analysis
- Output: Quantified contact data

**Stage 3: Tracking**
- Input: Contact data, optical flow
- Process: Multi-tracker processing, motion detection
- Output: Time-position mappings

**Stage 4: Filtering (Funscript Plugins)**
- Input: Raw funscript
- Plugins: Autotune, clamp, amplify, speed-limit, RDP simplify, etc.
- Output: Polished funscript

---

## 3. KEY MODULES AND THEIR RESPONSIBILITIES

### Application Module (49,818 lines, 75 files)

#### Logic Submodule (11 files)
- **app_logic.py** (2,334 lines) - Core orchestration, state management
- **app_stage_processor.py** (2,357 lines) - Coordinates all 3 processing stages
- **app_funscript_processor.py** (1,635 lines) - Funscript manipulation, chapters
- **app_file_manager.py** (893 lines) - Video/file I/O operations
- **app_event_handlers.py** (847 lines) - UI event processing
- **app_calibration.py** (165 lines) - Device calibration
- **app_energy_saver.py** (123 lines) - Power management
- **app_utility.py** (393 lines) - Miscellaneous helpers
- **app_state_ui.py** (1,017 lines) - UI state synchronization
- **tensorrt_compiler_logic.py** (280 lines) - TensorRT optimization

#### GUI Components Submodule (17 files)
- **control_panel_ui.py** (5,841 lines) - **LARGEST GUI FILE** - Processing controls, filters, settings
- **app_gui.py** (2,541 lines) - Main GUI window orchestration
- **video_display_ui.py** (2,391 lines) - Video rendering, heatmaps, preview
- **video_navigation_ui.py** (2,090 lines) - Timeline, scrubbing, navigation
- **splash_screen.py** (1,944 lines) - Startup splash, initialization display
- **info_graphs_ui.py** (1,878 lines) - Statistics, graphs, analysis visualization
- **toolbar_ui.py** (988 lines) - Button toolbar, playback controls
- **device_control_ui.py** (941 lines) - Device simulator controls
- **keyboard_shortcuts_dialog.py** (569 lines) - Keyboard mapping UI
- **chapter_type_manager_ui.py** (436 lines) - Chapter type configuration
- **dynamic_tracker_ui.py** (239 lines) - Runtime tracker selection
- **fullscreen_display.py** (365 lines) - Fullscreen video display mode
- **autotuner_window.py** (141 lines) - Autotune filter UI
- **generated_file_manager_window.py** (195 lines) - Output file browser

#### Classes Submodule (17 files)
- **interactive_timeline.py** (3,269 lines) - **LARGEST CLASS FILE** - Timeline rendering, funscript editing
- **menu.py** (1,404 lines) - Menu system implementation
- **simulator_3d.py** (750 lines) - 3D device simulator
- **chapter_manager.py** (547 lines) - Chapter/segment management
- **chapter_type_manager.py** (544 lines) - Chapter type definitions
- **file_dialog.py** (525 lines) - File browser dialog
- **plugin_ui_manager.py** (523 lines) - Plugin UI orchestration
- **project_manager.py** (410 lines) - Project save/load
- **plugin_ui_renderer.py** (400 lines) - Plugin UI rendering
- **plugin_preview_renderer.py** (360 lines) - Plugin preview display
- **settings_manager.py** (322 lines) - User settings storage
- **shortcut_manager.py** (233 lines) - Keyboard shortcuts
- **chapter_thumbnail_cache.py** (236 lines) - Thumbnail caching
- **movement_bar.py** (175 lines) - Movement indicator UI
- **gauge.py** (129 lines) - Value gauge widget
- **undo_redo_manager.py** (101 lines) - Undo/redo stack

#### Utils Submodule (22 files, 6,476 lines)
- **updater.py** (1,692 lines) - Auto-update system
- **processing_thread_manager.py** (633 lines) - Thread pool management
- **dependency_checker.py** (658 lines) - Dependency verification
- **logger.py** - Logging configuration
- **tensorrt_compiler.py** (619 lines) - Model compilation
- **system_monitor.py** - Performance monitoring
- **model_pool.py** - Model instance caching
- **video_segment.py** - Video segment data structure
- **time_format.py** - Time formatting utilities
- **checkpoint_manager.py** - Progress checkpointing
- And 12+ more utility modules

#### GPU Rendering Submodule
- GPU-accelerated texture rendering for video display
- OpenGL integration utilities

---

### Tracker Module (20,101 lines, 30 files)

#### Core Submodule (base_tracker.py, base_offline_tracker.py, security.py)
- **BaseTracker** - Abstract base for all tracker implementations
- **BaseOfflineTracker** - Base for offline/batch processing trackers
- **TrackerMetadata** - Tracker capability information
- **TrackerResult** - Standardized result format
- **Security validation** - Sandbox execution & validation

#### Live Trackers Submodule (Real-time processing)
- **yolo_roi.py** (1,505 lines) - YOLO-based region of interest tracking
- **user_roi.py** (803 lines) - User-defined region tracking
- **oscillation_legacy.py** (624 lines) - Legacy oscillation detection
- **oscillation_experimental_2.py** (587 lines) - Experimental oscillation v2

#### Offline Trackers Submodule (Batch processing)
- **stage3_mixed.py** (823 lines) - Mixed optical flow & detection fusion
- **stage3_optical_flow.py** (727 lines) - Optical flow-based tracking
- **stage2_contact_analysis.py** (603 lines) - Contact data analysis

#### Experimental Trackers Submodule (R&D implementations)
- **hybrid_intelligence.py** (3,339 lines) - **MOST COMPLEX TRACKER** - Multi-modal fusion
- **relative_distance.py** (2,287 lines) - Distance-based tracking
- **axis_projection_enhanced.py** (987 lines) - 3D axis projection
- **beat_marker.py** (642 lines) - Music/beat synchronization
- And backups/variants

#### Helpers Submodule
- **visualization.py** (735 lines) - Tracker output visualization
- **signal_amplifier.py** - Signal enhancement utilities

#### TrackerManager (983 lines)
- Orchestrates tracker instantiation and execution
- Discovers available trackers dynamically
- Manages tracker lifecycle and cleanup
- Interfaces between GUI/CLI and tracker modules

---

### Funscript Module (6,955 lines, 21 files)

#### Core
- **dual_axis_funscript.py** (1,352 lines) - Primary funscript implementation
  - Handles single-axis and dual-axis (roll) funscript generation
  - Point management, validation, serialization
  - Metadata handling (duration, ranges, etc.)

#### Plugins System (17 plugin files)
Plugin architecture allows modular post-processing filters:

**Signal Modification:**
- **autotune_plugin.py** (314 lines) - Automatic timing adjustment
- **ultimate_autotune_plugin.py** (336 lines) - Advanced autotune variant
- **amplify_plugin.py** (260 lines) - Amplitude scaling
- **dynamic_amplify_plugin.py** (302 lines) - Dynamic amplitude adjustment

**Signal Smoothing:**
- **savgol_filter_plugin.py** (295 lines) - Savitzky-Golay smoothing
- **clamp_plugin.py** (449 lines) - Clamp values to range

**Data Optimization:**
- **rdp_simplify_plugin.py** (518 lines) - Ramer-Douglas-Peucker simplification
- **speed_limiter_plugin.py** (537 lines) - Limit motion speed
- **resample_plugin.py** (354 lines) - Change sample rate

**Special Operations:**
- **keyframe_plugin.py** (464 lines) - Keyframe extraction
- **invert_plugin.py** (214 lines) - Invert motion direction
- **anti_jerk_plugin.py** (194 lines) - Remove jerky motion
- **time_shift_plugin.py** (160 lines) - Temporal adjustment

**Plugin Infrastructure:**
- **plugin_loader.py** (368 lines) - Dynamic plugin discovery and loading
- **base_plugin.py** (311 lines) - Abstract plugin base class

#### User Plugins
- **template_plugin.py** (103 lines) - Simple template for custom plugins
- **advanced_template_plugin.py** (382 lines) - Advanced plugin template
- **PLUGIN_DEVELOPMENT_GUIDE.md** - Documentation for plugin development

---

### Detection Module (6,937 lines, 10 files)

**3-Stage Detection Pipeline:**

1. **Stage 1: Contact Detection** (stage_1_cd.py, 1,061 lines)
   - Frame preprocessing
   - Contour detection
   - Initial motion identification
   - Input: Raw video frames
   - Output: Contact region masks

2. **Stage 2: Contact Analysis** (stage_2_cd.py, 2,698 lines) **[LARGEST DETECTION FILE]**
   - Time-series analysis of contact regions
   - Signal processing and enhancement
   - SQLite storage of intermediate results
   - Input: Contact masks over time
   - Output: Quantified contact strength/position

3. **Stage 3: Fusion Processors**
   - **stage_3_mixed_processor.py** (1,097 lines) - Combines optical flow + detection
   - **stage_3_of_processor.py** (596 lines) - Pure optical flow processing
   - Input: Optical flow data, contact detection results
   - Output: Consolidated position data

**Data Structures** (data_structures/)
- **segments.py** - Time segment representation
- **box_records.py** - Bounding box tracking
- **frame_objects.py** - Per-frame object data
- **__init__.py** - Module initialization

**Storage:**
- **stage_2_sqlite_storage.py** (785 lines) - SQLite DB for intermediate processing results

---

### Video Module (5,212 lines, 6 files)

- **video_processor.py** (2,831 lines) - Main video processing pipeline
  - Frame extraction and buffering
  - Format detection and handling
  - VR unwarp coordinate calculations
  - Multi-threaded frame processing
  
- **gpu_unwarp_worker.py** (1,035 lines) - GPU-accelerated VR format unwarp
  - Fisheye/equirectangular distortion correction
  - GPU texture operations
  - Real-time unwarp for VR videos
  
- **dual_frame_processor.py** (596 lines) - Dual-frame VR handling
  - Side-by-side stereo processing
  - Per-eye frame extraction
  
- **vr_format_detector_ml_real.py** (498 lines) - ML-based VR format detection
  - Automatically identifies video format type
  - Classifier-based detection
  
- **thumbnail_extractor.py** (247 lines) - Video thumbnail generation

---

### Config Module (1,664 lines, 6 files)

- **constants.py** (657 lines) - Application-wide constants
  - Model paths and URLs
  - Processing parameters
  - File extensions and formats
  - Default values

- **tracker_discovery.py** (368 lines) - Dynamic tracker discovery system
  - Scans for available trackers
  - Metadata caching
  - CLI mode enumeration

- **constants_colors.py** (265 lines) - Color definitions
- **element_group_colors.py** (259 lines) - UI element color schemes
- **theme_manager.py** (106 lines) - Theme switching system

---

### Common Module (487 lines, 5 files)

Shared utilities across the application:

- **exceptions.py** - Custom exception definitions
- **result.py** (104 lines) - Result wrapper class for method returns
- **http_client_manager.py** (124 lines) - HTTP client singleton
- **temp_manager.py** (209 lines) - Temporary file and directory management
- **__init__.py**

---

## 4. CURRENT SEPARATION OF CONCERNS

### Strengths (Good Separation)

1. **Clear Module Boundaries**
   - `tracker/` - Tracker implementations (pluggable)
   - `funscript/` - Funscript generation/filtering
   - `detection/` - Object detection pipeline
   - `video/` - Video I/O and processing
   - `application/` - GUI/CLI and orchestration

2. **Plugin Architecture**
   - Funscript filters use base class inheritance
   - Dynamic loader allows runtime extension
   - User can create custom plugins

3. **Tracker System**
   - Auto-discovery via registry pattern
   - Each tracker is independent module
   - Security sandboxing for community trackers
   - Metadata-driven capability advertisement

4. **Configuration Management**
   - Centralized constants in `config/`
   - Theme/color management separated
   - Dynamic tracker discovery system

### Areas of Concern (Mixed/Tight Coupling)

1. **Large God Classes**
   - `control_panel_ui.py` (5,841 lines) - Single file handles all control UI
   - `interactive_timeline.py` (3,269 lines) - Timeline rendering + editing
   - `app_stage_processor.py` (2,357 lines) - Multiple responsibilities
   - `app_logic.py` (2,334 lines) - Core orchestration has mixed concerns

2. **GUI-Logic Coupling**
   - UI components directly interact with app_logic
   - Some business logic exists in UI files
   - Threading management spread across multiple files

3. **Inconsistent Organization**
   - Some utilities in `/application/utils/` (22 files)
   - Some utilities in `/common/` (5 files)
   - Similar utilities scattered across modules

4. **Incomplete Abstraction**
   - Video module tightly coupled to YOLO models
   - Some tracker-specific code in stage processors
   - Hard-coded paths/constants in some places

5. **Complex Interdependencies**
   - TrackerManager depends on detection module results
   - GUI state manager depends on all processing stages
   - Circular references possible between GUI components and logic

---

## 5. FILE SIZES AND COMPLEXITY INDICATORS

### Size Distribution

**Largest Files (Code Complexity Risk):**
```
1.  control_panel_ui.py          5,841 lines    [VERY HIGH RISK]
2.  interactive_timeline.py      3,269 lines    [VERY HIGH RISK]
3.  app_stage_processor.py       2,357 lines    [HIGH RISK]
4.  app_logic.py                 2,334 lines    [HIGH RISK]
5.  video_display_ui.py          2,391 lines    [HIGH RISK]
6.  splash_screen.py             1,944 lines    [MODERATE RISK]
7.  info_graphs_ui.py            1,878 lines    [MODERATE RISK]
8.  dual_axis_funscript.py       1,352 lines    [MODERATE RISK]
9.  menu.py                      1,404 lines    [MODERATE RISK]
10. video_processor.py           2,831 lines    [HIGH RISK]
```

### Complexity Indicators

**By Feature/Responsibility:**
- **GUI Components:** 19,300+ lines (41% of code)
- **Processing Logic:** 12,000+ lines (25% of code)
- **Video Processing:** 5,200+ lines (11% of code)
- **Tracker System:** 20,100+ lines (21% of code)
- **Utilities:** 6,500+ lines (14% of code)

**Most Complex Single Components:**
1. **hybrid_intelligence.py** (3,339 lines) - Multi-modal tracking fusion
2. **relative_distance.py** (2,287 lines) - Distance-based tracking
3. **control_panel_ui.py** (5,841 lines) - GUI control panel
4. **stage_2_cd.py** (2,698 lines) - Contact analysis pipeline

---

## 6. DEPENDENCIES BETWEEN DIFFERENT PARTS

### Dependency Graph

```
main.py
  ├─→ application.logic.app_logic         [Core orchestrator]
  │    ├─→ application.classes.*          [Data models]
  │    ├─→ application.utils.*            [Helper functions]
  │    ├─→ tracker.tracker_manager        [Tracker orchestration]
  │    │    ├─→ tracker.tracker_modules.* [Tracker implementations]
  │    │    └─→ funscript.dual_axis_funscript
  │    ├─→ video.video_processor          [Video I/O]
  │    │    ├─→ video.gpu_unwarp_worker
  │    │    ├─→ video.vr_format_detector_ml_real
  │    │    └─→ video.dual_frame_processor
  │    ├─→ detection.cd.*                 [3-stage pipeline]
  │    │    ├─→ detection.cd.stage_1_cd
  │    │    ├─→ detection.cd.stage_2_cd
  │    │    ├─→ detection.cd.stage_3_*
  │    │    └─→ detection.cd.data_structures.*
  │    ├─→ funscript.dual_axis_funscript
  │    └─→ funscript.plugins.plugin_loader
  │         └─→ funscript.plugins.*       [Filter plugins]
  │
  ├─→ application.gui_components.GUI      [GUI renderer]
  │    ├─→ application.gui_components.*   [UI components]
  │    │    ├─→ control_panel_ui
  │    │    ├─→ video_display_ui
  │    │    ├─→ video_navigation_ui
  │    │    └─→ [other UI modules]
  │    ├─→ application.classes.*          [UI models & state]
  │    └─→ application.logic.app_logic    [Back to core logic]
  │
  └─→ config.*                            [Configuration]
       ├─→ constants.py                   [App constants]
       ├─→ tracker_discovery.py           [Tracker enumeration]
       └─→ [color definitions]
```

### Critical Path Dependencies

1. **Video Loading Path:**
   ```
   GUI → app_logic → video_processor
         → gpu_unwarp_worker
         → vr_format_detector_ml_real
   ```

2. **Processing Pipeline:**
   ```
   app_logic → app_stage_processor
            → stage_1_cd
            → stage_2_cd
            → stage_3_mixed / stage_3_of
            → tracker_manager
            → funscript generation
   ```

3. **Filtering/Post-Processing:**
   ```
   app_funscript_processor → plugin_loader
                          → individual plugins
   ```

4. **UI Rendering:**
   ```
   GUI → control_panel_ui, video_display_ui, video_navigation_ui
      → interactive_timeline
      → app_logic (for state updates)
   ```

### Cross-Cutting Concerns

**Logging:**
- All modules import from `application.utils.logger`
- Centralized logging configuration

**Configuration:**
- All modules reference `config.constants`
- Dynamic discovery via `config.tracker_discovery`

**Utilities:**
- Many modules depend on `application.utils.*`
- Video segment handling in multiple places

**State Management:**
- app_logic serves as central state holder
- Multiple GUI components read/write to app_logic
- Could benefit from proper state management pattern (Redux-like)

---

## 7. ENTRY POINTS AND EXECUTION MODES

### Mode 1: GUI Mode (Default)
```
main.py → run_gui()
  ├── ApplicationLogic(is_cli=False)
  └── GUI.run()
```
- Interactive video processing
- Real-time preview and visualization
- Manual funscript editing
- Settings management
- Project save/load

### Mode 2: CLI Mode - Video Processing
```
main.py [--mode MODE] input_path
  └── ApplicationLogic.run_cli()
    ├── Stage 1 (detection) - CLI progress callbacks
    ├── Stage 2 (analysis) - CLI progress callbacks
    ├── Stage 3 (tracking) - CLI progress callbacks
    └── Plugin filters - Apply settings
```
- Batch processing support
- Recursive folder processing
- Custom filter chains
- Available modes dynamically discovered

### Mode 3: CLI Mode - Funscript Filtering
```
main.py --funscript-mode --filter FILTER_NAME funscript_path
  └── ApplicationLogic.run_cli()
    ├── Load funscript(s)
    └── Apply selected filter
```
- Post-process existing funscripts
- Batch filter application

---

## 8. KEY ARCHITECTURAL PATTERNS

### 1. Plugin Architecture (Funscript Filters)
- **Base Class:** `funscript.plugins.base_plugin.BasePlugin`
- **Loader:** `funscript.plugins.plugin_loader`
- **Registry:** Dynamic discovery system
- **Usage:** Chained post-processing filters

### 2. Registry Pattern (Trackers)
- **Registry Class:** `tracker.tracker_modules.TrackerRegistry`
- **Discovery:** Auto-scan subdirectories
- **Validation:** Security sandboxing
- **Metadata:** Capability advertisement

### 3. Observer Pattern (UI Updates)
- GUI components observe app_logic state changes
- Callback-based event handling
- Event dispatcher in app_event_handlers

### 4. Factory Pattern (Tracker Creation)
- `TrackerManager.create_tracker()` instantiates appropriate tracker
- Metadata-driven selection
- Dependency injection of app reference

### 5. Pipeline Pattern (Video Processing)
- Multi-stage detection pipeline (3 stages)
- Each stage processes output of previous
- Result passing via data structures
- Can skip stages based on results

### 6. Thread Pool Pattern
- `ProcessingThreadManager` manages worker threads
- Decouples long-running tasks from UI thread
- Prevents GUI freezing during processing

### 7. Singleton Pattern
- Logger instance (centralized)
- HTTP client manager (single instance)
- Tracker discovery system

---

## 9. DATA STRUCTURES & MODEL CLASSES

Located in `application/classes/`:

1. **AppSettings** - Application configuration state
2. **ProjectManager** - Project file handling (load/save)
3. **ShortcutManager** - Keyboard shortcut definitions
4. **UndoRedoManager** - Undo/redo stack management
5. **ChapterManager** - Video chapter/segment handling
6. **ChapterTypeManager** - Chapter classification system
7. **InteractiveFunscriptTimeline** - Timeline editing model
8. **SimulatorState** - 3D device simulator state

Detection pipeline data structures in `detection/cd/data_structures/`:
1. **Segment** - Time segment representation
2. **BoxRecord** - Bounding box tracking data
3. **FrameObjects** - Per-frame detection results

---

## 10. TECHNOLOGY STACK

**Core Frameworks:**
- **ImGui (OpenGL)** - GUI rendering (immediate-mode)
- **GLFW** - Window management
- **OpenGL** - Graphics rendering

**Processing Libraries:**
- **OpenCV** - Image processing
- **NumPy/SciPy** - Numerical computing
- **Ultralytics YOLO** - Object detection
- **scikit-learn** - Machine learning utilities

**Video Handling:**
- **imageio** - Video I/O
- **FFmpeg** (external) - Video codec support

**Data & Storage:**
- **msgpack** - Binary serialization
- **orjson** - JSON serialization
- **aiosqlite** - Async SQLite
- **pickle** - Python object serialization

**Utilities:**
- **tqdm** - Progress bars
- **colorama** - Colored console output
- **Pillow** - Image manipulation
- **simplification** - RDP curve simplification

**Optional (GPU Acceleration):**
- **PyTorch** + CUDA/ROCm
- **TensorRT** - Model compilation/optimization

---

## 11. CONFIGURATION SYSTEM

### Configuration Files
- **core.requirements.txt** - Core dependencies
- **cuda.requirements.txt** - NVIDIA GPU support
- **rocm.requirements.txt** - AMD GPU support
- **macos.requirements.txt** - macOS-specific
- **cpu.requirements.txt** - CPU-only
- **environment.yml** - Conda specification

### Runtime Configuration
- **Stored in:** ~/.config/FunGen/ or OS equivalent
- **Project files:** *.fungen
- **Settings:** JSON format with encryption support

---

## 12. SUMMARY OF ARCHITECTURE QUALITY

### Positive Aspects
- Clear separation between GUI and processing logic
- Modular tracker system with auto-discovery
- Plugin architecture for funscript filters
- Multi-stage processing pipeline with clear interfaces
- Comprehensive error handling
- Cross-platform support (Windows, macOS, Linux)
- GPU acceleration support
- Checkpoint/resume capability

### Areas for Improvement
- Some files exceed 5,000 lines (refactor candidates)
- GUI-logic coupling could be reduced with better state management
- Utility functions scattered across multiple locations
- Some duplicate code between tracker variants
- Documentation could be more comprehensive
- Test coverage unknown

### Recommended Refactoring Priority
1. **High:** Split control_panel_ui.py (5,841 lines)
2. **High:** Refactor interactive_timeline.py (3,269 lines)
3. **Medium:** Extract logic from app_stage_processor.py
4. **Medium:** Consolidate utilities from app/utils and common
5. **Low:** Decouple GUI components from app_logic

---

## Conclusion

FunGen is a well-structured, feature-rich application with a clear separation between domain concerns (video processing, tracking, funscript generation) and presentation logic (GUI/CLI). The codebase demonstrates mature architectural patterns (plugins, registry, pipeline) but has some opportunities for refactoring to reduce file sizes and coupling.

The application successfully handles complex tasks including:
- Real-time video processing with GPU acceleration
- Multi-stage AI-powered analysis pipeline
- Extensible plugin system for post-processing
- Modular tracker framework for different algorithms
- Cross-platform GUI using ImGui/OpenGL

Total metrics:
- **93,761 lines** of Python code
- **157 Python files**
- **14 major modules**
- **3-stage processing pipeline**
- **8+ tracker implementations**
- **11 funscript filter plugins**
- **GUI + CLI interfaces**
