# FunGen Refactoring Roadmap
## Code Architecture Analysis & Improvement Plan

**Analysis Date:** 2025-11-18
**Codebase Size:** 93,761 lines across 157 Python files
**Focus:** Scalability, Performance, Lightweight, Ease of Use, User Experience

---

## Executive Summary

FunGen has a solid architectural foundation with clear module boundaries and good design patterns (Plugin, Registry, Pipeline). However, several files have grown excessively large (5,000+ lines), leading to maintenance challenges, reduced performance, and difficult onboarding for new developers.

This roadmap prioritizes refactoring from **quick wins** (immediate improvements) to **major architectural changes** (long-term transformation).

---

## Table of Contents

1. [Current Architecture Assessment](#current-architecture-assessment)
2. [Quick Wins (1-2 weeks)](#phase-1-quick-wins-1-2-weeks)
3. [Medium Complexity (3-6 weeks)](#phase-2-medium-complexity-3-6-weeks)
4. [Major Refactoring (2-3 months)](#phase-3-major-refactoring-2-3-months)
5. [Performance & Optimization](#phase-4-performance-optimization-ongoing)
6. [Implementation Guidelines](#implementation-guidelines)

---

## Current Architecture Assessment

### Strengths ✅

1. **Clear Module Boundaries**
   - Distinct separation: `tracker/`, `funscript/`, `detection/`, `video/`, `application/`
   - Each module has well-defined responsibilities

2. **Extensibility Patterns**
   - Plugin architecture for funscript filters (11 implementations)
   - Auto-discovery registry for trackers (8+ variants)
   - Community contribution support

3. **Processing Pipeline**
   - 4-stage pipeline with clear interfaces
   - Checkpointing and resume capability
   - Multi-threaded execution

4. **Cross-Platform Support**
   - Windows, macOS, Linux compatibility
   - GPU acceleration (CUDA, ROCm, Metal)
   - CPU fallback support

### Critical Issues ⚠️

1. **God Classes / Monolithic Files**
   - `control_panel_ui.py` - **5,841 lines** (CRITICAL)
   - `interactive_timeline.py` - **3,269 lines** (CRITICAL)
   - `app_stage_processor.py` - **2,357 lines** (HIGH)
   - `app_logic.py` - **2,334 lines** (HIGH)
   - `video_display_ui.py` - **2,391 lines** (HIGH)

2. **Scattered Utilities**
   - 22 files in `application/utils/` (~8,634 lines)
   - 5 files in `common/` (~487 lines)
   - Overlapping responsibilities and unclear organization

3. **GUI-Logic Coupling**
   - UI components directly reference `app_logic` state
   - Business logic embedded in UI rendering code
   - Difficult to test UI independently

4. **Complex Dependencies**
   - Circular references possible between GUI and logic
   - Hard-coded paths in some modules
   - Tight coupling between video module and YOLO models

---

## Phase 1: Quick Wins (1-2 weeks)

**Goal:** Immediate improvements with minimal risk, high impact on code maintainability.

### 1.1 Consolidate Utility Modules

**Priority:** HIGH
**Effort:** 3-5 days
**Impact:** Improved discoverability, reduced confusion

#### Current State
```
application/utils/
  ├── 22 files (~8,634 lines)
  └── Mixed concerns: logging, threading, UI helpers, network, etc.

common/
  ├── 5 files (~487 lines)
  └── Overlapping with app/utils
```

#### Target Structure
```
utils/
  ├── __init__.py
  ├── core/                      # Core utilities
  │   ├── logger.py
  │   ├── exceptions.py
  │   ├── result.py
  │   └── temp_manager.py
  ├── ui/                        # UI-specific utilities
  │   ├── icon_texture.py
  │   ├── logo_texture.py
  │   ├── button_styles.py
  │   └── keyboard_layout_detector.py
  ├── network/                   # Network operations
  │   ├── http_client_manager.py
  │   ├── network_utils.py
  │   └── github_token_manager.py
  ├── processing/                # Processing utilities
  │   ├── processing_thread_manager.py
  │   ├── checkpoint_manager.py
  │   ├── stage_output_validator.py
  │   └── stage2_signal_enhancer.py
  ├── video/                     # Video utilities
  │   ├── video_segment.py
  │   ├── time_format.py
  │   └── generated_file_manager.py
  ├── system/                    # System utilities
  │   ├── system_monitor.py
  │   ├── system_scaling.py
  │   ├── feature_detection.py
  │   ├── write_access.py
  │   └── dependency_checker.py
  ├── ml/                        # ML/AI utilities
  │   ├── model_pool.py
  │   ├── tensorrt_compiler.py
  │   └── tensorrt_export_engine_model.py
  └── app/                       # Application utilities
      ├── updater.py
      └── rts_smoother.py
```

#### Benefits
- **Discoverability:** Developers can quickly find utilities by category
- **Maintainability:** Clear separation of concerns
- **Testing:** Easier to write focused unit tests
- **Import Clarity:** `from utils.network import http_client` vs `from application.utils.http_client_manager import HTTPClientManager`

#### Migration Steps
1. Create new `utils/` structure at project root
2. Move files from `common/` and `application/utils/` to new locations
3. Update all imports using find-and-replace
4. Run tests to verify no breakage
5. Remove old directories

---

### 1.2 Extract Helper Functions from UI Files

**Priority:** HIGH
**Effort:** 2-3 days
**Impact:** Reduced file size, improved reusability

#### Files to Extract From
1. **control_panel_ui.py** (5,841 lines)
   - Extract: `_tooltip_if_hovered`, `_readonly_input`, `_DisabledScope`
   - Target: `utils/ui/imgui_helpers.py`

2. **video_display_ui.py** (2,391 lines)
   - Extract: Texture loading helpers, OpenGL utilities
   - Target: `utils/ui/opengl_helpers.py`

3. **interactive_timeline.py** (3,269 lines)
   - Extract: Hash computation, array caching utilities
   - Target: `utils/ui/timeline_helpers.py`

#### Example Extraction

**Before:** `control_panel_ui.py`
```python
def _tooltip_if_hovered(text):
    if imgui.is_item_hovered():
        imgui.set_tooltip(text)

class _DisabledScope:
    # ... 30 lines ...
```

**After:** `utils/ui/imgui_helpers.py`
```python
"""ImGui UI helper functions and context managers."""

def tooltip_if_hovered(text: str) -> None:
    """Show tooltip when item is hovered."""
    if imgui.is_item_hovered():
        imgui.set_tooltip(text)

class DisabledScope:
    """Context manager for disabled UI elements."""
    # ... implementation ...
```

**After:** `control_panel_ui.py`
```python
from utils.ui.imgui_helpers import tooltip_if_hovered, DisabledScope

# Use imported helpers
```

#### Benefits
- **File Size Reduction:** 200-500 lines per file
- **Reusability:** Helpers available across all UI modules
- **Testing:** Easier to unit test helpers in isolation
- **Performance:** Can cache/optimize helpers without affecting UI logic

---

### 1.3 Create Configuration Registry

**Priority:** MEDIUM
**Effort:** 2-4 days
**Impact:** Centralized configuration, easier testing

#### Current Issues
- Constants scattered across `config/constants.py` (657 lines)
- Hard-coded paths in various modules
- Difficult to override for testing

#### Target Structure
```python
# config/registry.py
from typing import Any, Dict
from pathlib import Path

class ConfigRegistry:
    """Centralized configuration registry with environment overrides."""

    _instance = None
    _config: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with fallback."""
        return self._config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self._config[key] = value

    def load_from_env(self) -> None:
        """Load overrides from environment variables."""
        # ... implementation ...

    @property
    def models_dir(self) -> Path:
        """Get models directory path."""
        return Path(self.get('MODELS_DIR', DEFAULT_MODELS_DIR))
```

#### Benefits
- **Testability:** Easy to mock configurations in tests
- **Flexibility:** Environment-based overrides
- **Type Safety:** Typed property access
- **Documentation:** Self-documenting configuration

---

### 1.4 Add Type Hints to Core Modules

**Priority:** MEDIUM
**Effort:** 3-5 days
**Impact:** Better IDE support, early error detection

#### Target Files (Priority Order)
1. `funscript/dual_axis_funscript.py` - Core data structure
2. `tracker/tracker_manager.py` - Tracker orchestration
3. `detection/cd/data_structures/*.py` - Data models
4. `video/video_processor.py` - Video processing

#### Example

**Before:**
```python
def process_frame(self, frame, frame_idx):
    result = self._detect_contours(frame)
    return result
```

**After:**
```python
from typing import Tuple, Optional
import numpy as np

def process_frame(
    self,
    frame: np.ndarray,
    frame_idx: int
) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
    """Process a single frame for contour detection.

    Args:
        frame: Input frame as numpy array (H, W, C)
        frame_idx: Frame index in video

    Returns:
        Tuple of (processed_frame, metadata) or None if processing failed
    """
    result = self._detect_contours(frame)
    return result
```

#### Benefits
- **IDE Support:** Better autocomplete and error detection
- **Documentation:** Type hints serve as inline documentation
- **Refactoring Safety:** Easier to refactor with type checking
- **Onboarding:** New developers understand interfaces faster

---

### 1.5 Document Public APIs

**Priority:** LOW (but valuable)
**Effort:** 2-3 days
**Impact:** Improved developer experience

#### Create API Documentation Files
```
docs/
  ├── api/
  │   ├── tracker_api.md        # Tracker plugin development
  │   ├── funscript_filter_api.md  # Filter plugin development
  │   ├── video_processing_api.md  # Video processing pipeline
  │   └── gui_components_api.md    # GUI component development
  └── architecture/
      ├── processing_pipeline.md
      ├── state_management.md
      └── threading_model.md
```

---

## Phase 2: Medium Complexity (3-6 weeks)

**Goal:** Break down monolithic files, improve modularity.

### 2.1 Split `control_panel_ui.py` (5,841 lines)

**Priority:** CRITICAL
**Effort:** 2-3 weeks
**Impact:** Massive maintainability improvement

#### Current Structure Analysis
The file contains:
- ~90 slot attributes (initialization complexity)
- Multiple tab rendering methods (Input, Preprocessing, Detection, Tracking, Post-Processing, Advanced)
- Device control logic (supporter feature)
- Live tracker integration
- Plugin UI management
- Performance optimization code

#### Target Structure
```
application/gui_components/control_panel/
  ├── __init__.py                     # Main ControlPanelUI orchestrator
  ├── base.py                         # Base class with common functionality (~300 lines)
  ├── tabs/
  │   ├── __init__.py
  │   ├── input_tab.py                # Input configuration tab (~600 lines)
  │   ├── preprocessing_tab.py        # Preprocessing settings (~500 lines)
  │   ├── detection_tab.py            # Detection settings (~700 lines)
  │   ├── tracking_tab.py             # Tracker selection/config (~800 lines)
  │   ├── postprocessing_tab.py       # Funscript filtering (~600 lines)
  │   ├── advanced_tab.py             # Advanced settings (~500 lines)
  │   └── device_control_tab.py       # Device control (supporter) (~600 lines)
  ├── widgets/
  │   ├── __init__.py
  │   ├── tracker_selector.py         # Dynamic tracker UI (~200 lines)
  │   ├── model_selector.py           # AI model selection (~150 lines)
  │   ├── range_slider.py             # Range input widget (~100 lines)
  │   └── file_path_input.py          # File path widget (~100 lines)
  ├── managers/
  │   ├── __init__.py
  │   ├── device_manager.py           # Device control logic (~400 lines)
  │   ├── live_tracker_bridge.py      # Live tracker integration (~300 lines)
  │   └── video_playback_bridge.py    # Video playback bridge (~200 lines)
  └── utils/
      ├── __init__.py
      ├── validation.py               # Input validation (~150 lines)
      └── state_sync.py               # UI-app state sync (~200 lines)
```

#### Refactoring Strategy

**Step 1: Extract Base Class (Week 1)**
```python
# application/gui_components/control_panel/base.py
class ControlPanelBase:
    """Base class for control panel with common functionality."""

    def __init__(self, app):
        self.app = app
        self._init_colors()
        self._init_performance_cache()

    def _init_colors(self):
        """Initialize color schemes."""
        self.ControlPanelColors = config.ControlPanelColors
        self.GeneralColors = config.GeneralColors

    def _init_performance_cache(self):
        """Initialize performance optimization attributes."""
        self._last_tab_hash = None
        self._cached_tab_content = {}
        self._widget_visibility_cache = {}
```

**Step 2: Extract Tab Classes (Week 1-2)**
```python
# application/gui_components/control_panel/tabs/input_tab.py
class InputTab:
    """Input configuration tab for video/funscript selection."""

    def __init__(self, app, parent_ui):
        self.app = app
        self.parent = parent_ui

    def render(self) -> None:
        """Render the input configuration tab."""
        self._render_video_input_section()
        self._render_funscript_input_section()
        self._render_output_path_section()

    def _render_video_input_section(self):
        """Render video file selection."""
        # ... implementation ...
```

**Step 3: Create Main Orchestrator (Week 2)**
```python
# application/gui_components/control_panel/__init__.py
from .base import ControlPanelBase
from .tabs import InputTab, PreprocessingTab, DetectionTab, TrackingTab, PostProcessingTab, AdvancedTab

class ControlPanelUI(ControlPanelBase):
    """Main control panel UI orchestrator."""

    def __init__(self, app):
        super().__init__(app)

        # Initialize tab components
        self.input_tab = InputTab(app, self)
        self.preprocessing_tab = PreprocessingTab(app, self)
        self.detection_tab = DetectionTab(app, self)
        self.tracking_tab = TrackingTab(app, self)
        self.postprocessing_tab = PostProcessingTab(app, self)
        self.advanced_tab = AdvancedTab(app, self)

    def render(self):
        """Render the control panel with tabs."""
        if imgui.begin_tab_bar("ControlPanelTabs"):
            if imgui.begin_tab_item("Input")[0]:
                self.input_tab.render()
                imgui.end_tab_item()

            if imgui.begin_tab_item("Preprocessing")[0]:
                self.preprocessing_tab.render()
                imgui.end_tab_item()

            # ... other tabs ...

            imgui.end_tab_bar()
```

**Step 4: Gradual Migration (Week 3)**
- Update imports in `app_gui.py`
- Test each tab individually
- Ensure no regressions in functionality

#### Benefits
- **File Size:** 5,841 lines → ~300 lines (main) + 6 files (~500-800 lines each)
- **Maintainability:** Each tab is independent and testable
- **Performance:** Lazy loading of tabs (only render active tab)
- **Team Collaboration:** Multiple developers can work on different tabs simultaneously
- **Extensibility:** Easy to add new tabs or widgets

---

### 2.2 Refactor `interactive_timeline.py` (3,269 lines)

**Priority:** CRITICAL
**Effort:** 2 weeks
**Impact:** Improved performance, better separation of concerns

#### Current Issues
- Rendering logic mixed with data manipulation
- Complex caching mechanisms
- Multiple responsibilities: rendering, editing, plugin preview, selection

#### Target Structure
```
application/classes/timeline/
  ├── __init__.py
  ├── timeline.py                     # Main timeline class (~400 lines)
  ├── rendering/
  │   ├── __init__.py
  │   ├── renderer.py                 # Base rendering (~300 lines)
  │   ├── dense_envelope_renderer.py  # Dense visualization (~200 lines)
  │   ├── waveform_renderer.py        # Waveform rendering (~150 lines)
  │   └── selection_renderer.py       # Selection highlighting (~150 lines)
  ├── editing/
  │   ├── __init__.py
  │   ├── point_editor.py             # Add/delete/move points (~300 lines)
  │   ├── selection_manager.py        # Selection logic (~250 lines)
  │   ├── clipboard_manager.py        # Copy/paste operations (~200 lines)
  │   └── undo_manager.py             # Undo/redo integration (~150 lines)
  ├── plugins/
  │   ├── __init__.py
  │   ├── plugin_preview.py           # Plugin preview rendering (~300 lines)
  │   ├── plugin_menu.py              # Plugin selection menu (~200 lines)
  │   └── ultimate_autotune.py        # Ultimate autotune integration (~250 lines)
  ├── cache/
  │   ├── __init__.py
  │   ├── cache_manager.py            # Cache coordination (~200 lines)
  │   ├── array_cache.py              # NumPy array caching (~150 lines)
  │   └── preview_cache.py            # Preview data caching (~150 lines)
  └── utils/
      ├── __init__.py
      ├── geometry.py                 # Coordinate calculations (~150 lines)
      ├── selection_filters.py        # Top/bottom/mid point filters (~200 lines)
      └── hash_utils.py               # Hash computation (~100 lines)
```

#### Refactoring Example

**Before:** Monolithic render method
```python
def render(self, timeline_y_start_coord, timeline_render_height, view_mode):
    # 500+ lines of rendering logic
    # Mixed: data fetching, rendering, event handling, caching
```

**After:** Separated concerns
```python
# timeline.py
class InteractiveFunscriptTimeline:
    def __init__(self, app_instance, timeline_num):
        self.app = app_instance
        self.timeline_num = timeline_num

        # Composition over monolithic class
        self.renderer = TimelineRenderer(self)
        self.editor = PointEditor(self)
        self.selection_mgr = SelectionManager(self)
        self.clipboard = ClipboardManager(self)
        self.plugin_preview = PluginPreview(self)
        self.cache = CacheManager(self)

    def render(self, y_start, height, view_mode):
        """Main rendering entry point."""
        # Get cached data
        data = self.cache.get_or_compute_data()

        # Delegate rendering
        self.renderer.render(data, y_start, height, view_mode)

        # Handle editing events
        if self.editor.has_edit_event():
            self.editor.process_edit()

        # Update plugin preview if needed
        if self.plugin_preview.should_update():
            self.plugin_preview.update()

# rendering/renderer.py
class TimelineRenderer:
    def render(self, data, y_start, height, view_mode):
        """Render timeline visualization."""
        # Focused rendering logic only
        if view_mode == 'expert':
            self._render_dense_envelope(data)
        else:
            self._render_simple_line(data)
```

#### Benefits
- **Performance:** Specialized renderers can be optimized independently
- **Testability:** Each component can be unit tested
- **Caching:** Simplified cache management with dedicated cache classes
- **Extensibility:** Easy to add new rendering modes or editing features

---

### 2.3 Decompose `app_stage_processor.py` (2,357 lines)

**Priority:** HIGH
**Effort:** 1.5 weeks
**Impact:** Better testability, clearer pipeline logic

#### Current Issues
- Manages all 3 processing stages
- Checkpoint logic
- Progress callbacks
- Settings management
- Thread coordination

#### Target Structure
```
application/logic/processing/
  ├── __init__.py
  ├── pipeline.py                     # Main pipeline orchestrator (~300 lines)
  ├── stages/
  │   ├── __init__.py
  │   ├── base_stage.py               # Abstract base stage (~150 lines)
  │   ├── stage1_processor.py         # Stage 1 execution (~400 lines)
  │   ├── stage2_processor.py         # Stage 2 execution (~500 lines)
  │   └── stage3_processor.py         # Stage 3 execution (~400 lines)
  ├── checkpointing/
  │   ├── __init__.py
  │   ├── checkpoint_manager.py       # Checkpoint CRUD (~300 lines)
  │   └── resume_handler.py           # Resume logic (~200 lines)
  ├── progress/
  │   ├── __init__.py
  │   ├── progress_tracker.py         # Progress tracking (~150 lines)
  │   └── callbacks.py                # Callback system (~100 lines)
  └── validation/
      ├── __init__.py
      └── artifact_validator.py       # Validate processed artifacts (~150 lines)
```

#### Refactoring Example

**Before:** Single class handles everything
```python
class AppStageProcessor:
    def start_full_analysis(self, processing_mode, **kwargs):
        # 400+ lines handling all stages

    def _execute_stage1_logic(self, frame_range, **kwargs):
        # 200+ lines

    def _execute_stage2_logic(self, **kwargs):
        # 300+ lines

    def _execute_stage3_optical_flow_module(self, **kwargs):
        # 200+ lines
```

**After:** Separated pipeline and stages
```python
# pipeline.py
class ProcessingPipeline:
    """Orchestrates the multi-stage processing pipeline."""

    def __init__(self, app_logic):
        self.app = app_logic
        self.stage1 = Stage1Processor(self)
        self.stage2 = Stage2Processor(self)
        self.stage3 = Stage3Processor(self)
        self.checkpoint_mgr = CheckpointManager(self)
        self.progress = ProgressTracker(self)

    def execute(self, mode: str, **settings) -> bool:
        """Execute the full processing pipeline."""
        try:
            # Check for resumable checkpoint
            checkpoint = self.checkpoint_mgr.find_latest()
            if checkpoint:
                return self._resume_from_checkpoint(checkpoint)

            # Execute stages sequentially
            if not self.stage1.execute(settings):
                return False

            if not self.stage2.execute(settings):
                return False

            if not self.stage3.execute(settings):
                return False

            self.checkpoint_mgr.cleanup()
            return True

        except Exception as e:
            self.checkpoint_mgr.save()
            raise

# stages/stage1_processor.py
class Stage1Processor(BaseStageProcessor):
    """Handles Stage 1: Contact Detection."""

    def execute(self, settings: Dict) -> bool:
        """Execute stage 1 processing."""
        # Focused stage 1 logic
        video_path = settings['video_path']
        frame_range = settings.get('frame_range')

        # Initialize detection module
        detector = self._create_detector(settings)

        # Process frames
        results = detector.process_video(
            video_path,
            frame_range=frame_range,
            progress_callback=self.progress_callback
        )

        # Save results
        self._save_stage1_results(results)
        return True
```

#### Benefits
- **Clarity:** Each stage is self-contained
- **Testing:** Easy to test stages individually
- **Reusability:** Stages can be run independently (e.g., stage 1 only)
- **Maintainability:** Changes to one stage don't affect others

---

### 2.4 Introduce State Management Layer

**Priority:** HIGH
**Effort:** 2 weeks
**Impact:** Reduced GUI-logic coupling, better testability

#### Current Issues
- GUI components directly access `app_logic` attributes
- State changes scattered across multiple files
- Difficult to track state mutations
- No centralized state history

#### Target Structure: Redux-like Architecture
```
application/state/
  ├── __init__.py
  ├── store.py                        # Central state store (~200 lines)
  ├── actions.py                      # Action creators (~300 lines)
  ├── reducers/
  │   ├── __init__.py
  │   ├── video_reducer.py            # Video state (~150 lines)
  │   ├── processing_reducer.py       # Processing state (~200 lines)
  │   ├── funscript_reducer.py        # Funscript state (~150 lines)
  │   └── ui_reducer.py               # UI state (~150 lines)
  ├── selectors/
  │   ├── __init__.py
  │   ├── video_selectors.py          # Video state queries (~100 lines)
  │   └── processing_selectors.py     # Processing state queries (~100 lines)
  └── middleware/
      ├── __init__.py
      ├── logging_middleware.py       # Log state changes (~50 lines)
      └── persistence_middleware.py   # Auto-save state (~100 lines)
```

#### Implementation Example

```python
# state/store.py
from typing import Dict, Any, Callable, List
from dataclasses import dataclass, field

@dataclass
class AppState:
    """Immutable application state."""
    video: Dict[str, Any] = field(default_factory=dict)
    processing: Dict[str, Any] = field(default_factory=dict)
    funscript: Dict[str, Any] = field(default_factory=dict)
    ui: Dict[str, Any] = field(default_factory=dict)

class Store:
    """Central state store with immutable updates."""

    def __init__(self, initial_state: AppState):
        self._state = initial_state
        self._subscribers: List[Callable] = []
        self._middleware: List[Callable] = []

    def get_state(self) -> AppState:
        """Get current state (immutable)."""
        return self._state

    def dispatch(self, action: Dict[str, Any]) -> None:
        """Dispatch action to update state."""
        # Apply middleware
        for middleware in self._middleware:
            action = middleware(action, self._state)

        # Apply reducers
        new_state = self._reduce(action, self._state)

        # Update state if changed
        if new_state != self._state:
            self._state = new_state
            self._notify_subscribers()

    def subscribe(self, callback: Callable) -> Callable:
        """Subscribe to state changes."""
        self._subscribers.append(callback)
        return lambda: self._subscribers.remove(callback)

    def _reduce(self, action: Dict, state: AppState) -> AppState:
        """Apply action to state using reducers."""
        return AppState(
            video=video_reducer(state.video, action),
            processing=processing_reducer(state.processing, action),
            funscript=funscript_reducer(state.funscript, action),
            ui=ui_reducer(state.ui, action),
        )

# state/actions.py
class VideoActions:
    """Action creators for video operations."""

    @staticmethod
    def load_video(path: str) -> Dict:
        return {
            'type': 'VIDEO_LOAD_REQUESTED',
            'payload': {'path': path}
        }

    @staticmethod
    def video_loaded(metadata: Dict) -> Dict:
        return {
            'type': 'VIDEO_LOADED',
            'payload': metadata
        }

# state/reducers/video_reducer.py
def video_reducer(state: Dict, action: Dict) -> Dict:
    """Handle video-related actions."""
    action_type = action['type']

    if action_type == 'VIDEO_LOADED':
        return {
            **state,
            'loaded': True,
            'metadata': action['payload'],
            'current_frame': 0,
        }

    elif action_type == 'VIDEO_SEEK':
        return {
            **state,
            'current_frame': action['payload']['frame'],
        }

    return state

# Usage in GUI
class ControlPanelUI:
    def __init__(self, app):
        self.store = app.store
        self.store.subscribe(self._on_state_change)

    def _on_state_change(self):
        """React to state changes."""
        state = self.store.get_state()
        # Update UI based on new state
        self._update_video_info(state.video)

    def _handle_load_video_button(self):
        """Handle video load button click."""
        path = self._show_file_dialog()
        self.store.dispatch(VideoActions.load_video(path))
```

#### Benefits
- **Predictability:** All state changes go through actions/reducers
- **Debugging:** Easy to log all state changes
- **Time Travel:** Can implement undo/redo at state level
- **Testing:** Pure functions (reducers) are easy to test
- **Decoupling:** GUI doesn't directly modify app state

---

### 2.5 Refactor Video Processing Module

**Priority:** MEDIUM
**Effort:** 1 week
**Impact:** Reduced coupling, better performance

#### Current Issues
- `video_processor.py` is 2,831 lines
- Tight coupling with YOLO models
- Multiple responsibilities: frame extraction, format detection, VR unwarp

#### Target Structure
```
video/
  ├── __init__.py
  ├── processor.py                    # Main processor (~400 lines)
  ├── readers/
  │   ├── __init__.py
  │   ├── base_reader.py              # Abstract reader (~100 lines)
  │   ├── standard_reader.py          # Standard video (~200 lines)
  │   └── vr_reader.py                # VR video (~200 lines)
  ├── formats/
  │   ├── __init__.py
  │   ├── detector.py                 # Format detection (~300 lines)
  │   └── vr_formats.py               # VR format definitions (~150 lines)
  ├── unwarp/
  │   ├── __init__.py
  │   ├── unwarp_engine.py            # Unwarp coordination (~200 lines)
  │   ├── gpu_unwarp_worker.py        # GPU implementation (existing)
  │   └── cpu_unwarp_worker.py        # CPU fallback (~300 lines)
  ├── frames/
  │   ├── __init__.py
  │   ├── frame_buffer.py             # Frame buffering (~200 lines)
  │   ├── frame_cache.py              # Frame caching (~150 lines)
  │   └── dual_frame_processor.py     # Existing dual-frame logic
  └── thumbnails/
      ├── __init__.py
      └── extractor.py                # Existing thumbnail logic
```

#### Refactoring Example

**Before:** Monolithic processor
```python
class VideoProcessor:
    def __init__(self):
        # 100+ lines of initialization
        self.yolo_model = YOLO(...)  # Tight coupling
        self.unwarp_worker = ...
        self.format_detector = ...

    def load_video(self, path):
        # 300+ lines handling everything
```

**After:** Modular processors
```python
# processor.py
class VideoProcessor:
    """Main video processing coordinator."""

    def __init__(self, config: Dict):
        self.config = config
        self.reader: Optional[BaseVideoReader] = None
        self.format_detector = FormatDetector()
        self.unwarp_engine = UnwarpEngine(config)
        self.frame_buffer = FrameBuffer(max_size=config.get('buffer_size', 100))

    def load_video(self, path: str) -> bool:
        """Load video and detect format."""
        # Detect format
        format_info = self.format_detector.detect(path)

        # Create appropriate reader
        if format_info.is_vr:
            self.reader = VRVideoReader(path, format_info, self.unwarp_engine)
        else:
            self.reader = StandardVideoReader(path)

        return self.reader.open()

    def get_frame(self, frame_idx: int) -> np.ndarray:
        """Get frame with caching."""
        # Check cache
        if self.frame_buffer.has(frame_idx):
            return self.frame_buffer.get(frame_idx)

        # Read from reader
        frame = self.reader.read_frame(frame_idx)
        self.frame_buffer.put(frame_idx, frame)
        return frame

# readers/base_reader.py
class BaseVideoReader(ABC):
    """Abstract base for video readers."""

    @abstractmethod
    def open(self) -> bool:
        """Open video file."""
        pass

    @abstractmethod
    def read_frame(self, frame_idx: int) -> np.ndarray:
        """Read specific frame."""
        pass

    @abstractmethod
    def get_metadata(self) -> Dict:
        """Get video metadata."""
        pass
```

#### Benefits
- **Modularity:** Each component has single responsibility
- **Performance:** Specialized readers can optimize differently
- **Testing:** Easy to mock readers/detectors
- **Extensibility:** Easy to add new video formats

---

## Phase 3: Major Refactoring (2-3 months)

**Goal:** Fundamental architectural improvements for long-term scalability.

### 3.1 Implement Clean Architecture

**Priority:** HIGH (Long-term)
**Effort:** 6-8 weeks
**Impact:** Maximum scalability, testability, maintainability

#### Current Architecture
```
GUI → ApplicationLogic → (Video, Tracker, Detection, Funscript)
      ↑                   ↑
      └───────────────────┘
      (Bidirectional coupling)
```

#### Target: Clean Architecture (Hexagonal/Ports & Adapters)
```
┌─────────────────────────────────────────────────────────────┐
│                      Presentation Layer                     │
│  ├─ GUI (ImGui)                                            │
│  └─ CLI                                                    │
└────────────────────┬────────────────────────────────────────┘
                     │ (uses)
┌────────────────────▼────────────────────────────────────────┐
│                    Application Layer                        │
│  ├─ Use Cases (ProcessVideo, GenerateFunscript, etc.)     │
│  ├─ DTOs (Data Transfer Objects)                          │
│  └─ Ports (Interfaces)                                    │
│     ├─ IVideoRepository                                   │
│     ├─ ITrackerService                                    │
│     ├─ IFunscriptGenerator                                │
│     └─ IProgressReporter                                  │
└────────────────────┬────────────────────────────────────────┘
                     │ (implements)
┌────────────────────▼────────────────────────────────────────┐
│                     Domain Layer                            │
│  ├─ Entities (Video, Funscript, Track, etc.)              │
│  ├─ Value Objects (TimeCode, Position, Frame)             │
│  └─ Domain Services (ActionGenerator, SignalProcessor)    │
└─────────────────────────────────────────────────────────────┘
                     ▲
                     │ (used by)
┌────────────────────┴────────────────────────────────────────┐
│                 Infrastructure Layer                        │
│  ├─ Adapters                                               │
│  │  ├─ VideoRepositoryImpl (OpenCV, imageio)              │
│  │  ├─ TrackerServiceImpl (YOLO, OF trackers)             │
│  │  └─ FunscriptGeneratorImpl                             │
│  ├─ External Services                                      │
│  │  ├─ Model downloaders                                  │
│  │  └─ Update checkers                                    │
│  └─ Persistence                                            │
│     ├─ FileSystem                                          │
│     └─ SQLite                                              │
└─────────────────────────────────────────────────────────────┘
```

#### New Directory Structure
```
fungen/
  ├── presentation/                   # Presentation layer
  │   ├── gui/                        # ImGui GUI
  │   │   ├── __init__.py
  │   │   ├── main_window.py
  │   │   ├── components/
  │   │   └── viewmodels/             # MVVM pattern
  │   └── cli/                        # CLI interface
  │       ├── __init__.py
  │       └── commands/
  │
  ├── application/                    # Application layer
  │   ├── __init__.py
  │   ├── use_cases/                  # Business logic use cases
  │   │   ├── __init__.py
  │   │   ├── process_video.py
  │   │   ├── generate_funscript.py
  │   │   ├── apply_filter.py
  │   │   └── export_funscript.py
  │   ├── dtos/                       # Data transfer objects
  │   │   ├── __init__.py
  │   │   ├── video_dto.py
  │   │   └── funscript_dto.py
  │   └── ports/                      # Interfaces (dependency inversion)
  │       ├── __init__.py
  │       ├── repositories.py         # Repository interfaces
  │       └── services.py             # Service interfaces
  │
  ├── domain/                         # Domain layer (core business logic)
  │   ├── __init__.py
  │   ├── entities/                   # Core entities
  │   │   ├── __init__.py
  │   │   ├── video.py
  │   │   ├── funscript.py
  │   │   ├── track.py
  │   │   └── action.py
  │   ├── value_objects/              # Immutable value objects
  │   │   ├── __init__.py
  │   │   ├── timecode.py
  │   │   ├── position.py
  │   │   └── frame.py
  │   └── services/                   # Domain services
  │       ├── __init__.py
  │       ├── action_generator.py
  │       └── signal_processor.py
  │
  └── infrastructure/                 # Infrastructure layer
      ├── __init__.py
      ├── repositories/               # Repository implementations
      │   ├── __init__.py
      │   ├── video_repository.py
      │   ├── funscript_repository.py
      │   └── checkpoint_repository.py
      ├── services/                   # Service implementations
      │   ├── __init__.py
      │   ├── tracker_service.py      # Wraps tracker module
      │   ├── detection_service.py    # Wraps detection module
      │   └── filter_service.py       # Wraps funscript plugins
      ├── external/                   # External service integrations
      │   ├── __init__.py
      │   ├── model_downloader.py
      │   └── update_checker.py
      └── persistence/                # Data persistence
          ├── __init__.py
          ├── filesystem.py
          └── sqlite_storage.py
```

#### Example Use Case

```python
# application/use_cases/process_video.py
from typing import Protocol
from dataclasses import dataclass

# Ports (interfaces for dependency inversion)
class IVideoRepository(Protocol):
    def load(self, path: str) -> Video:
        ...

class ITrackerService(Protocol):
    def track(self, video: Video, settings: Dict) -> List[Track]:
        ...

class IFunscriptGenerator(Protocol):
    def generate(self, tracks: List[Track]) -> Funscript:
        ...

# Use case
@dataclass
class ProcessVideoRequest:
    video_path: str
    tracker_name: str
    settings: Dict

@dataclass
class ProcessVideoResponse:
    funscript: Funscript
    metadata: Dict

class ProcessVideoUseCase:
    """Use case for processing video to generate funscript."""

    def __init__(
        self,
        video_repo: IVideoRepository,
        tracker_service: ITrackerService,
        funscript_generator: IFunscriptGenerator,
        progress_reporter: Optional[IProgressReporter] = None
    ):
        self.video_repo = video_repo
        self.tracker_service = tracker_service
        self.funscript_generator = funscript_generator
        self.progress = progress_reporter

    def execute(self, request: ProcessVideoRequest) -> ProcessVideoResponse:
        """Execute the use case."""
        # Load video (infrastructure concern)
        if self.progress:
            self.progress.report("Loading video...", 0.0)
        video = self.video_repo.load(request.video_path)

        # Track motion (domain logic via service)
        if self.progress:
            self.progress.report("Tracking motion...", 0.3)
        tracks = self.tracker_service.track(video, request.settings)

        # Generate funscript (domain logic)
        if self.progress:
            self.progress.report("Generating funscript...", 0.8)
        funscript = self.funscript_generator.generate(tracks)

        if self.progress:
            self.progress.report("Complete", 1.0)

        return ProcessVideoResponse(
            funscript=funscript,
            metadata={'video_path': request.video_path}
        )
```

#### Benefits
- **Testability:** Business logic independent of UI/infrastructure
- **Flexibility:** Can swap GUI, CLI, web interface without changing core
- **Maintainability:** Clear boundaries between layers
- **Scalability:** Easy to add new features following established patterns

---

### 3.2 Separate GUI from Logic (MVVM Pattern)

**Priority:** HIGH
**Effort:** 4-6 weeks
**Impact:** Better testability, UI flexibility

#### Current State
```python
# GUI directly accesses app_logic
class ControlPanelUI:
    def render(self):
        if imgui.button("Process"):
            self.app.start_processing()  # Direct coupling
```

#### Target: MVVM (Model-View-ViewModel)
```
┌─────────┐         ┌──────────────┐         ┌───────┐
│  View   │ binds   │  ViewModel   │ uses    │ Model │
│ (ImGui) │────────▶│ (Properties) │────────▶│(Logic)│
└─────────┘         └──────────────┘         └───────┘
     │                      │
     │   (UI events)        │ (notifications)
     └──────────────────────┘
```

#### Implementation

```python
# presentation/gui/viewmodels/control_panel_viewmodel.py
from typing import Callable, List
from dataclasses import dataclass, field

@dataclass
class VideoInputViewModel:
    """ViewModel for video input section."""
    video_path: str = ""
    is_valid: bool = False
    error_message: str = ""
    is_loading: bool = False

    # Callbacks for UI events
    on_path_changed: Callable[[str], None] = field(default=lambda x: None)
    on_load_clicked: Callable[[], None] = field(default=lambda: None)

class ControlPanelViewModel:
    """ViewModel for control panel (presentation logic)."""

    def __init__(self, process_video_use_case, load_video_use_case):
        self.process_video = process_video_use_case
        self.load_video = load_video_use_case

        # Observable properties
        self.video_input = VideoInputViewModel()
        self.processing_state = ProcessingStateViewModel()

        # Property change listeners
        self._listeners = []

        # Setup callbacks
        self.video_input.on_path_changed = self._handle_video_path_changed
        self.video_input.on_load_clicked = self._handle_load_video

    def _handle_video_path_changed(self, path: str):
        """Handle video path change (validation logic)."""
        self.video_input.video_path = path
        self.video_input.is_valid = self._validate_video_path(path)
        if not self.video_input.is_valid:
            self.video_input.error_message = "Invalid video path"
        else:
            self.video_input.error_message = ""
        self._notify_change()

    def _handle_load_video(self):
        """Handle load video button click."""
        self.video_input.is_loading = True
        self._notify_change()

        try:
            result = self.load_video.execute(self.video_input.video_path)
            self.video_input.is_loading = False
            # Update other properties...
        except Exception as e:
            self.video_input.error_message = str(e)
            self.video_input.is_loading = False

        self._notify_change()

    def subscribe(self, callback: Callable):
        """Subscribe to property changes."""
        self._listeners.append(callback)

    def _notify_change(self):
        """Notify all listeners of property changes."""
        for listener in self._listeners:
            listener()

# presentation/gui/components/control_panel_view.py
class ControlPanelView:
    """View for control panel (pure UI rendering)."""

    def __init__(self, viewmodel: ControlPanelViewModel):
        self.vm = viewmodel
        self.vm.subscribe(self._on_viewmodel_changed)

    def render(self):
        """Render UI based on ViewModel."""
        self._render_video_input_section()
        self._render_processing_section()

    def _render_video_input_section(self):
        """Render video input (reads from ViewModel, no logic)."""
        imgui.text("Video Path:")

        # Input field bound to ViewModel
        changed, new_value = imgui.input_text(
            "##video_path",
            self.vm.video_input.video_path,
            256
        )
        if changed:
            self.vm.video_input.on_path_changed(new_value)

        # Show error if invalid
        if not self.vm.video_input.is_valid and self.vm.video_input.error_message:
            imgui.text_colored(self.vm.video_input.error_message, 1.0, 0.0, 0.0)

        # Load button (disabled during loading)
        if self.vm.video_input.is_loading:
            imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)
            imgui.button("Loading...")
            imgui.pop_style_var()
        else:
            if imgui.button("Load Video"):
                self.vm.video_input.on_load_clicked()

    def _on_viewmodel_changed(self):
        """Called when ViewModel properties change."""
        # UI will re-render on next frame
        pass
```

#### Benefits
- **Testability:** ViewModel can be tested without UI
- **Separation:** UI code is pure rendering, no business logic
- **Reusability:** ViewModel can be used by different views
- **Debugging:** Easy to inspect ViewModel state

---

### 3.3 Performance Optimization: Lazy Loading & Code Splitting

**Priority:** MEDIUM
**Effort:** 2 weeks
**Impact:** Faster startup, lower memory footprint

#### Current Issues
- All modules loaded at startup (slow)
- Large dependencies loaded even if not used
- Memory footprint grows with features

#### Target: Lazy Loading Strategy

```python
# application/lazy_imports.py
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from tracker.tracker_manager import TrackerManager
    from detection.cd.stage_2_cd import Stage2ContactDetection

class LazyImporter:
    """Lazy import manager for expensive modules."""

    _instances = {}

    @classmethod
    def get_tracker_manager(cls) -> 'TrackerManager':
        """Lazily import and cache tracker manager."""
        if 'tracker_manager' not in cls._instances:
            from tracker.tracker_manager import create_tracker_manager
            cls._instances['tracker_manager'] = create_tracker_manager()
        return cls._instances['tracker_manager']

    @classmethod
    def get_stage2_detector(cls) -> 'Stage2ContactDetection':
        """Lazily import Stage 2 detector."""
        if 'stage2_detector' not in cls._instances:
            from detection.cd.stage_2_cd import Stage2ContactDetection
            cls._instances['stage2_detector'] = Stage2ContactDetection
        return cls._instances['stage2_detector']

    @classmethod
    def get_yolo_model(cls, model_path: str):
        """Lazily load YOLO model (expensive)."""
        cache_key = f'yolo_{model_path}'
        if cache_key not in cls._instances:
            from ultralytics import YOLO
            cls._instances[cache_key] = YOLO(model_path)
        return cls._instances[cache_key]

# Usage
class ProcessingPipeline:
    def execute_stage2(self):
        # Only import when actually needed
        Stage2Detector = LazyImporter.get_stage2_detector()
        detector = Stage2Detector()
        # ... use detector
```

#### Plugin Lazy Loading

```python
# funscript/plugins/plugin_loader.py
class PluginLoader:
    """Lazy plugin loader with dynamic discovery."""

    def __init__(self):
        self._available_plugins = {}  # metadata only
        self._loaded_plugins = {}      # actual instances
        self._discover_plugins()

    def _discover_plugins(self):
        """Discover available plugins (lightweight)."""
        plugin_dir = Path(__file__).parent
        for file in plugin_dir.glob("*_plugin.py"):
            # Just store metadata, don't import yet
            plugin_name = file.stem
            self._available_plugins[plugin_name] = {
                'path': file,
                'loaded': False
            }

    def get_plugin(self, plugin_name: str):
        """Load plugin on-demand."""
        if plugin_name in self._loaded_plugins:
            return self._loaded_plugins[plugin_name]

        if plugin_name not in self._available_plugins:
            raise ValueError(f"Plugin {plugin_name} not found")

        # Import and instantiate
        plugin_path = self._available_plugins[plugin_name]['path']
        module = self._import_module(plugin_path)
        plugin_class = self._find_plugin_class(module)

        instance = plugin_class()
        self._loaded_plugins[plugin_name] = instance
        return instance

    def list_available(self) -> List[str]:
        """List available plugins without loading."""
        return list(self._available_plugins.keys())
```

#### Benefits
- **Startup Speed:** 2-3x faster startup (only load GUI essentials)
- **Memory:** 30-50% lower baseline memory usage
- **Responsiveness:** UI feels snappier
- **Modularity:** Encourages cleaner module boundaries

---

## Phase 4: Performance Optimization (Ongoing)

### 4.1 Optimize Hot Paths

**Priority:** HIGH
**Effort:** 1-2 weeks
**Impact:** Significantly faster processing

#### Identify Hot Paths (Profiling)

```python
# performance/profiler.py
import cProfile
import pstats
from functools import wraps

def profile_function(output_file: str = None):
    """Decorator to profile function execution."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            profiler = cProfile.Profile()
            profiler.enable()
            result = func(*args, **kwargs)
            profiler.disable()

            if output_file:
                profiler.dump_stats(output_file)
            else:
                stats = pstats.Stats(profiler)
                stats.sort_stats('cumulative')
                stats.print_stats(20)

            return result
        return wrapper
    return decorator

# Usage
@profile_function("stage1_profile.stats")
def execute_stage1(video_path):
    # ... processing logic
```

#### Optimize Frame Processing

**Current:** Frame-by-frame processing with Python loops
```python
# Slow: Python loop
for i, frame in enumerate(frames):
    processed = self._process_frame(frame)
    results.append(processed)
```

**Optimized:** Batch processing with NumPy
```python
# Fast: Vectorized operations
frames_batch = np.stack(frames, axis=0)  # (N, H, W, C)
processed_batch = self._process_frames_vectorized(frames_batch)
```

#### Optimize Signal Processing

**Current:** Python list operations
```python
# Slow
smoothed = []
for i in range(len(signal)):
    window = signal[max(0, i-5):i+6]
    smoothed.append(sum(window) / len(window))
```

**Optimized:** NumPy convolution
```python
# Fast
import numpy as np
kernel = np.ones(11) / 11
smoothed = np.convolve(signal, kernel, mode='same')
```

---

### 4.2 Implement Caching Strategy

**Priority:** MEDIUM
**Effort:** 1 week
**Impact:** 3-10x speedup for repeated operations

#### Multi-Level Cache

```python
# utils/caching/cache_manager.py
from typing import Any, Optional, Callable
import hashlib
import pickle
from pathlib import Path

class CacheManager:
    """Multi-level cache manager (memory + disk)."""

    def __init__(self, cache_dir: Path, max_memory_mb: int = 500):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        self.max_memory_bytes = max_memory_mb * 1024 * 1024

        # Level 1: Memory cache (LRU)
        self._memory_cache = {}
        self._memory_usage = 0
        self._access_order = []

    def get_or_compute(
        self,
        key: str,
        compute_fn: Callable,
        ttl_seconds: Optional[int] = None,
        use_disk: bool = True
    ) -> Any:
        """Get from cache or compute and cache."""
        # Check memory cache
        if key in self._memory_cache:
            self._update_access(key)
            return self._memory_cache[key]

        # Check disk cache
        if use_disk:
            disk_value = self._get_from_disk(key, ttl_seconds)
            if disk_value is not None:
                self._put_in_memory(key, disk_value)
                return disk_value

        # Compute and cache
        value = compute_fn()
        self._put_in_memory(key, value)
        if use_disk:
            self._put_on_disk(key, value)

        return value

    def _put_in_memory(self, key: str, value: Any):
        """Put value in memory cache with LRU eviction."""
        import sys
        value_size = sys.getsizeof(value)

        # Evict if necessary
        while self._memory_usage + value_size > self.max_memory_bytes:
            if not self._access_order:
                break
            evict_key = self._access_order.pop(0)
            evict_value = self._memory_cache.pop(evict_key)
            self._memory_usage -= sys.getsizeof(evict_value)

        # Add to cache
        self._memory_cache[key] = value
        self._memory_usage += value_size
        self._update_access(key)

    def _put_on_disk(self, key: str, value: Any):
        """Persist value to disk."""
        cache_path = self.cache_dir / f"{self._hash_key(key)}.cache"
        with open(cache_path, 'wb') as f:
            pickle.dump(value, f)

    def _get_from_disk(self, key: str, ttl_seconds: Optional[int]) -> Optional[Any]:
        """Retrieve value from disk if not expired."""
        cache_path = self.cache_dir / f"{self._hash_key(key)}.cache"

        if not cache_path.exists():
            return None

        # Check TTL
        if ttl_seconds:
            import time
            age = time.time() - cache_path.stat().st_mtime
            if age > ttl_seconds:
                cache_path.unlink()
                return None

        # Load from disk
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            cache_path.unlink()
            return None

    @staticmethod
    def _hash_key(key: str) -> str:
        """Hash key for filename."""
        return hashlib.sha256(key.encode()).hexdigest()[:16]

# Usage in video processor
class VideoProcessor:
    def __init__(self):
        self.cache = CacheManager(Path("cache/video"))

    def get_frame(self, video_path: str, frame_idx: int) -> np.ndarray:
        """Get frame with caching."""
        cache_key = f"{video_path}:{frame_idx}"
        return self.cache.get_or_compute(
            key=cache_key,
            compute_fn=lambda: self._read_frame_from_disk(video_path, frame_idx),
            use_disk=True
        )
```

---

### 4.3 GPU Acceleration Everywhere

**Priority:** MEDIUM
**Effort:** 2-3 weeks
**Impact:** 5-20x speedup for supported operations

#### Identify GPU Opportunities
1. Frame preprocessing (resize, normalize)
2. Optical flow computation
3. Contour detection (edge detection)
4. Signal filtering (convolution)

#### CuPy Integration (CUDA)

```python
# utils/gpu/accelerator.py
import numpy as np
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = np
    GPU_AVAILABLE = False

class GPUAccelerator:
    """Automatic GPU acceleration with CPU fallback."""

    @staticmethod
    def to_gpu(array: np.ndarray):
        """Move array to GPU."""
        if GPU_AVAILABLE:
            return cp.asarray(array)
        return array

    @staticmethod
    def to_cpu(array):
        """Move array to CPU."""
        if GPU_AVAILABLE and isinstance(array, cp.ndarray):
            return cp.asnumpy(array)
        return array

    @staticmethod
    def smooth_signal_gpu(signal: np.ndarray, window_size: int) -> np.ndarray:
        """GPU-accelerated signal smoothing."""
        if not GPU_AVAILABLE:
            # CPU fallback
            return np.convolve(signal, np.ones(window_size)/window_size, mode='same')

        # GPU implementation
        signal_gpu = cp.asarray(signal)
        kernel = cp.ones(window_size) / window_size
        smoothed_gpu = cp.convolve(signal_gpu, kernel, mode='same')
        return cp.asnumpy(smoothed_gpu)

# Usage in signal processing
class SignalProcessor:
    def __init__(self):
        self.gpu = GPUAccelerator()

    def smooth(self, signal: np.ndarray) -> np.ndarray:
        return self.gpu.smooth_signal_gpu(signal, window_size=11)
```

---

### 4.4 Reduce Memory Footprint

**Priority:** MEDIUM
**Effort:** 1-2 weeks
**Impact:** Handle larger videos, reduce crashes

#### Memory Profiling

```python
# performance/memory_profiler.py
import tracemalloc
from functools import wraps

def profile_memory(func):
    """Decorator to profile memory usage."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        result = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"{func.__name__}")
        print(f"  Current memory: {current / 1024 / 1024:.2f} MB")
        print(f"  Peak memory: {peak / 1024 / 1024:.2f} MB")

        return result
    return wrapper
```

#### Streaming Processing

**Current:** Load entire video into memory
```python
# Memory-hungry
all_frames = []
for i in range(total_frames):
    frame = video.read()
    all_frames.append(frame)
# Process all_frames (uses GB of RAM)
```

**Optimized:** Stream processing
```python
# Memory-efficient
def process_video_streaming(video_path, batch_size=32):
    """Process video in batches."""
    video = VideoReader(video_path)

    batch = []
    for frame_idx in range(video.frame_count):
        frame = video.read_frame(frame_idx)
        batch.append(frame)

        if len(batch) == batch_size:
            # Process batch
            results = process_batch(batch)
            yield results

            # Clear batch
            batch.clear()
            del results  # Explicit cleanup

    # Process remaining
    if batch:
        yield process_batch(batch)
```

---

## Implementation Guidelines

### Development Workflow

#### 1. Pre-Refactoring Checklist
- [ ] Create feature branch: `git checkout -b refactor/[component-name]`
- [ ] Document current behavior (tests/screenshots)
- [ ] Identify all dependencies
- [ ] Create rollback plan

#### 2. During Refactoring
- [ ] Write tests BEFORE refactoring (if none exist)
- [ ] Refactor in small, atomic commits
- [ ] Run tests after each commit
- [ ] Update documentation as you go
- [ ] Use feature flags for risky changes

#### 3. Post-Refactoring
- [ ] Run full test suite
- [ ] Performance benchmarks (before vs after)
- [ ] Update API documentation
- [ ] Code review with team
- [ ] Merge to main with descriptive PR

### Testing Strategy

#### Unit Tests
```python
# tests/unit/test_video_processor.py
import pytest
from video.processor import VideoProcessor

def test_load_video_success():
    processor = VideoProcessor()
    result = processor.load_video("test_video.mp4")
    assert result is True
    assert processor.reader is not None

def test_get_frame_returns_correct_shape():
    processor = VideoProcessor()
    processor.load_video("test_video.mp4")
    frame = processor.get_frame(0)
    assert frame.shape == (1080, 1920, 3)
```

#### Integration Tests
```python
# tests/integration/test_processing_pipeline.py
def test_full_pipeline_execution():
    """Test complete video processing pipeline."""
    pipeline = ProcessingPipeline()
    result = pipeline.execute(
        mode="3-stage",
        video_path="test_video.mp4"
    )
    assert result.success is True
    assert result.funscript is not None
```

#### Performance Tests
```python
# tests/performance/test_benchmarks.py
import time

def test_frame_processing_performance():
    """Ensure frame processing meets performance targets."""
    processor = VideoProcessor()
    processor.load_video("benchmark_video.mp4")

    start = time.time()
    for i in range(100):
        processor.get_frame(i)
    elapsed = time.time() - start

    fps = 100 / elapsed
    assert fps >= 30, f"Frame processing too slow: {fps:.2f} FPS"
```

### Code Quality Standards

#### Linting & Formatting
```bash
# Install tools
pip install black flake8 mypy isort

# Format code
black .

# Sort imports
isort .

# Lint
flake8 .

# Type check
mypy .
```

#### Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.1.0
    hooks:
      - id: black

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: ['--max-line-length=120']
```

---

## Migration Strategy: Living with Legacy

During the transition, you'll have both old and new code coexisting. Here's how to manage:

### Strangler Fig Pattern

```
┌─────────────────────────────────────┐
│        New Architecture             │
│  ┌───────────────────────────┐     │
│  │  New Components           │     │
│  │  (Clean Architecture)     │     │
│  └───────────────────────────┘     │
│          ↓ Adapter ↓               │
│  ┌───────────────────────────┐     │
│  │  Legacy Components        │     │
│  │  (Gradually shrinking)    │     │
│  └───────────────────────────┘     │
└─────────────────────────────────────┘
```

#### Example: Gradual Migration

```python
# New clean architecture use case
from application.use_cases import ProcessVideoUseCase

# Adapter for legacy code
class LegacyAdapter:
    """Adapter to use new use cases with legacy GUI."""

    def __init__(self, app_logic):
        self.legacy_app = app_logic

        # Create new use case with legacy dependencies
        self.process_video = ProcessVideoUseCase(
            video_repo=LegacyVideoRepository(app_logic),
            tracker_service=LegacyTrackerService(app_logic),
            funscript_generator=LegacyFunscriptGenerator(app_logic)
        )

    def start_processing(self):
        """Legacy method that now uses new use case."""
        request = ProcessVideoRequest(
            video_path=self.legacy_app.input_video_path,
            tracker_name=self.legacy_app.selected_tracker,
            settings=self.legacy_app.get_settings()
        )

        response = self.process_video.execute(request)

        # Update legacy state with response
        self.legacy_app.output_funscript = response.funscript

# Use in legacy GUI
class ControlPanelUI:
    def __init__(self, app):
        self.app = app
        self.adapter = LegacyAdapter(app)  # Use adapter

    def render(self):
        if imgui.button("Process"):
            self.adapter.start_processing()  # Use new code via adapter
```

---

## Metrics & Success Criteria

### Performance Targets
- **Startup Time:** < 2 seconds (from 5-7 seconds)
- **Frame Processing:** > 30 FPS (from 10-15 FPS)
- **Memory Usage:** < 2 GB for 4K video (from 4-6 GB)
- **GUI Responsiveness:** 60 FPS UI (no freezing)

### Code Quality Targets
- **Test Coverage:** > 70% (currently ~0%)
- **Average File Size:** < 500 lines (from 1,200 lines average)
- **Cyclomatic Complexity:** < 10 per function (currently 20-50)
- **Type Hint Coverage:** > 80% (currently ~5%)

### Developer Experience Targets
- **Onboarding Time:** < 2 days to make first contribution
- **Build Time:** < 30 seconds (full lint + test)
- **Documentation:** 100% of public APIs documented

---

## Risk Mitigation

### Potential Risks

1. **Breaking Changes**
   - **Mitigation:** Comprehensive test suite BEFORE refactoring
   - **Mitigation:** Feature flags for gradual rollout
   - **Mitigation:** Maintain compatibility layer during transition

2. **Performance Regression**
   - **Mitigation:** Performance benchmarks in CI/CD
   - **Mitigation:** Profile before and after refactoring
   - **Mitigation:** Rollback plan for each phase

3. **Team Velocity**
   - **Mitigation:** Dedicate refactoring sprints (no new features)
   - **Mitigation:** Pair programming for knowledge transfer
   - **Mitigation:** Document decisions in ADRs (Architecture Decision Records)

4. **Scope Creep**
   - **Mitigation:** Strict adherence to phase boundaries
   - **Mitigation:** Time-box each refactoring task
   - **Mitigation:** "Stop Loss" threshold (if taking >150% estimated time, reassess)

---

## Conclusion

This refactoring roadmap provides a structured path from quick wins to major architectural improvements. The key principles:

1. **Start Small:** Quick wins build momentum and confidence
2. **Measure Everything:** Metrics guide decisions
3. **Incremental Progress:** Small, safe steps prevent disasters
4. **Living Documentation:** Keep this roadmap updated as you progress

### Recommended Execution Order

**Month 1-2:** Phase 1 (Quick Wins)
- Consolidate utilities
- Extract UI helpers
- Add type hints
- Create configuration registry

**Month 3-5:** Phase 2 (Medium Complexity)
- Split `control_panel_ui.py`
- Refactor `interactive_timeline.py`
- Decompose `app_stage_processor.py`
- Introduce state management

**Month 6-9:** Phase 3 (Major Refactoring)
- Implement clean architecture
- Separate GUI from logic (MVVM)
- Lazy loading & code splitting

**Ongoing:** Phase 4 (Performance)
- Profile and optimize hot paths
- Implement caching
- GPU acceleration
- Memory optimization

---

**Next Steps:**
1. Review this roadmap with the team
2. Prioritize based on current pain points
3. Start with Phase 1, Task 1.1 (Consolidate Utilities)
4. Set up testing infrastructure
5. Begin refactoring!

Good luck! 🚀
