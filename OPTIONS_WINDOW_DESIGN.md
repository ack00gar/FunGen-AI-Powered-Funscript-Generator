# FunGen Options Window - Complete Redesign

## Overview
This document describes the comprehensive Options window redesign that consolidates all 400+ settings and features into a unified interface with vertical and horizontal tabs.

## Design Goals
1. **Unified Settings**: Single Options window for all configuration
2. **Minimal Side Panel**: Keep control panel minimal (Run + Post-Processing only)
3. **Complete Coverage**: All 400+ UI elements organized logically
4. **Search Capability**: Quick access to any setting
5. **Intuitive Navigation**: Vertical tabs for categories, horizontal tabs for subdivisions
6. **Persistence**: Auto-save all changes to settings.json

---

## Window Structure

### Layout
```
┌─────────────────────────────────────────────────────────────┐
│ Options                                              [X]     │
├──────────┬──────────────────────────────────────────────────┤
│          │ [Tab 1] [Tab 2] [Tab 3] [Tab 4]                  │
│ General  │ ┌────────────────────────────────────────────┐  │
│          │ │                                            │  │
│ Display  │ │         Settings Content Area              │  │
│          │ │                                            │  │
│ AI       │ │  [Controls, sliders, checkboxes, etc.]    │  │
│          │ │                                            │  │
│ Tracking │ │                                            │  │
│          │ │                                            │  │
│ Output   │ │                                            │  │
│          │ │                                            │  │
│ Device   │ └────────────────────────────────────────────┘  │
│          │                                                  │
│ Keys     │ [Search: ___________________________] [Clear]   │
│          │                                                  │
│ About    │ [Reset to Defaults] [Apply] [OK] [Cancel]       │
└──────────┴──────────────────────────────────────────────────┘
```

---

## Vertical Tabs (Main Categories)

### 1. 🎨 **General**
Interface, appearance, and application behavior settings

**Horizontal Tabs:**
- **Interface** - UI mode, layout, font scale, toolbar
- **Performance** - Energy saver, GPU rendering, video decoding
- **Autosave** - Project autosave settings, backup intervals

**Settings Count:** ~25

---

### 2. 🖥️ **Display & Windows**
Video display, gauges, timelines, and window management

**Horizontal Tabs:**
- **Video** - Zoom defaults, overlay options, fullscreen
- **Gauges** - Timeline 1/2 gauges, movement bar, 3D simulator
- **Timelines** - Interactive timelines, heatmap, chapter indicators
- **Panels** - Panel visibility, floating window positions

**Settings Count:** ~30

---

### 3. 🤖 **AI Models**
AI model configuration and inference settings

**Horizontal Tabs:**
- **Models** - YOLO detection, pose estimation paths
- **Inference** - Worker threads, batch sizes, hardware acceleration
- **TensorRT** - Optional TensorRT compilation settings

**Settings Count:** ~15

---

### 4. 🎯 **Tracking & Detection**
Live tracking, ROI, optical flow, and oscillation detection

**Horizontal Tabs:**
- **General** - Confidence threshold, tracker mode defaults
- **ROI** - Padding, update interval, smoothing, persistence
- **Optical Flow** - DIS flow settings, sparse flow, finest scale
- **Sensitivity** - Output sensitivity, amplification, class multipliers
- **Oscillation** - Grid size, detection sensitivity, decay settings
- **Filtering** - Class filtering checkboxes, multi-select

**Settings Count:** ~50

---

### 5. 📊 **Funscript Generation**
Funscript generation, range processing, and interactive refinement

**Horizontal Tabs:**
- **General** - Tracking axis mode, range processing
- **User ROI** - ROI selection, amplification controls
- **Refinement** - Scale, center, clamps, smoothing
- **Simplification** - RDP settings, output range
- **Calibration** - Latency calibration, timeline comparison

**Settings Count:** ~35

---

### 6. 🔧 **Post-Processing**
Post-processing plugins, profiles, and final smoothing

**Horizontal Tabs:**
- **Auto Processing** - Enable, per-chapter profiles, final RDP
- **Profiles** - Per-position processing profiles
- **Plugins** - Plugin selection and parameters (dynamic)

**Settings Count:** ~20 (plus dynamic plugin parameters)

---

### 7. 💾 **File Output**
Output folder, file generation options, batch mode

**Horizontal Tabs:**
- **General** - Output folder, .funscript/.roll generation
- **Batch Mode** - Overwrite strategy, autosave to video location
- **Advanced** - Stage 2 database retention, preprocessing cache

**Settings Count:** ~15

---

### 8. 🕹️ **Device Control** *(Supporters)*
Handy, OSR2, Buttplug integration and live tracking

**Horizontal Tabs:**
- **Connection** - Device discovery, backend selection, auto-connect
- **Handy** - Handy-specific settings
- **OSR2** - OSR2-specific settings
- **Buttplug** - Buttplug server address, port, device management
- **Live Tracking** - Device sync, max rate, live tracking options

**Settings Count:** ~25 (conditional)

**Feature Gate:** Only visible if user has supporter/buyer status

---

### 9. 📡 **Streamer** *(Supporters)*
XBVR integration and native sync

**Horizontal Tabs:**
- **XBVR** - Host, port, connection settings
- **Sync** - Sync status, client monitoring
- **Advanced** - Additional streaming options

**Settings Count:** ~10 (conditional)

**Feature Gate:** Only visible if user has supporter/buyer status

---

### 10. ⌨️ **Keyboard Shortcuts**
Customizable keyboard bindings

**Horizontal Tabs:**
- **Navigation** - Playback, seeking, frame navigation
- **Editing** - Undo/redo, point manipulation
- **Project** - New/open/save/export
- **Chapters** - Chapter creation, navigation

**Settings Count:** ~30 shortcuts

---

### 11. ℹ️ **About**
Application info, updates, support

**Horizontal Tabs:**
- **Info** - Version, GitHub link, license
- **Support** - Ko-fi support button, documentation links
- **Updates** - Check for updates, update settings

**Settings Count:** N/A (informational)

---

## Search Functionality

**Global Search Bar** (bottom of options window):
- Search across all categories and settings
- Real-time filtering
- Shows matching category/tab
- Highlights matching settings
- Clear button to reset search

**Implementation:**
- Search index built from all setting labels and tooltips
- Jump to matching category/tab when result selected
- Multiple results shown in dropdown

---

## Control Panel Simplification

### New Minimal Control Panel (Right/Left Side)

**2 Tabs Only:**

#### 1. ⚡ **Run** (Tracking & Processing)
- Video info display
- Tracker mode selector (simple/expert)
- Start/Stop/Pause/Resume buttons
- Progress bars (Stage 1/2/3)
- Quick export button
- **"Options..." button** → Opens Options window

#### 2. 🔧 **Post-Processing** (Quick Access)
- Current plugin selector
- Apply button
- **"More Options..."** button → Opens Options window to Post-Processing tab

**Removed from Control Panel:**
- Advanced tab → Moved to Options window
- Device Control tab → Moved to Options window
- Streamer tab → Moved to Options window

---

## Options Window Access Points

### 1. **Menu Bar**
- **Edit** → **Options...** (standard location)
- **Tools** → **Options...** (alternative)

### 2. **Toolbar**
- New "Options" button in TOOLS section
- Icon: ⚙️ gear icon

### 3. **Control Panel**
- "Options..." button in Run tab
- "More Options..." button in Post-Processing tab

### 4. **Keyboard Shortcut**
- **Ctrl+,** (standard options shortcut)
- **F12** (alternative)

---

## Implementation Details

### File Structure
```
/application/gui_components/
├── options_window.py           (NEW - main options window)
├── options_tabs/              (NEW - directory for tab implementations)
│   ├── general_tab.py
│   ├── display_tab.py
│   ├── ai_models_tab.py
│   ├── tracking_tab.py
│   ├── funscript_generation_tab.py
│   ├── post_processing_tab.py
│   ├── file_output_tab.py
│   ├── device_control_tab.py   (reuse from device_control_ui.py)
│   ├── streamer_tab.py         (reuse from control_panel_ui.py)
│   ├── keyboard_shortcuts_tab.py
│   └── about_tab.py
├── control_panel_ui.py         (MODIFIED - simplified to 2 tabs)
├── toolbar_ui.py               (MODIFIED - add Options button)
└── menu.py                     (MODIFIED - add Options menu item)
```

### Settings Persistence
- All settings auto-save to `settings.json` on change
- No separate Apply/OK needed (real-time)
- Cancel button discards changes since window opened
- Reset to Defaults button per-category

### Conditional Visibility
- Device Control tab: Only show if `is_supporter()` returns True
- Streamer tab: Only show if `is_supporter()` returns True
- Advanced controls: Show/hide based on UI mode (Simple vs Expert)
- Hardware options: Show/hide based on system capabilities

### UI Framework
- Use Dear PyGui (DPyG) consistent with existing codebase
- Vertical tabs: Custom widget using selectable buttons + separators
- Horizontal tabs: Standard `dpg.tab_bar()` and `dpg.tab()`
- Search: Text input with filter callback
- Responsive layout: Handle window resize

---

## Migration Plan

### Phase 1: Create Options Window Structure ✓
- [x] Design document (this file)
- [ ] Create `options_window.py` with vertical tabs
- [ ] Implement horizontal tab framework
- [ ] Add search functionality

### Phase 2: Migrate Settings
- [ ] General settings
- [ ] Display & Windows settings
- [ ] AI Models settings
- [ ] Tracking & Detection settings
- [ ] Funscript Generation settings
- [ ] Post-Processing settings
- [ ] File Output settings
- [ ] Device Control (reuse existing UI)
- [ ] Streamer (reuse existing UI)
- [ ] Keyboard Shortcuts (reuse existing dialog)

### Phase 3: Simplify Control Panel
- [ ] Remove Advanced tab
- [ ] Remove nested Device Control tab
- [ ] Remove nested Streamer tab
- [ ] Add "Options..." buttons to Run and Post-Processing tabs

### Phase 4: Integration
- [ ] Add Options button to toolbar
- [ ] Add Options menu items
- [ ] Add keyboard shortcuts
- [ ] Test all settings persistence
- [ ] Verify feature gating

### Phase 5: Testing & Refinement
- [ ] Test all 400+ settings
- [ ] Verify search functionality
- [ ] Check conditional visibility
- [ ] Validate settings persistence
- [ ] User testing feedback

---

## Settings Coverage Checklist

### ✅ All 400+ Settings Covered

**General (25):**
- [x] UI mode, layout mode, font scale
- [x] Auto system scaling, show toolbar
- [x] Energy saver, GPU rendering, video decoding
- [x] Autosave settings

**Display & Windows (30):**
- [x] Video display options
- [x] Gauge windows (T1, T2, movement bar, 3D simulator)
- [x] Timeline visibility, interactive timelines, heatmap
- [x] Chapter indicators, overlays

**AI Models (15):**
- [x] YOLO detection model path
- [x] Pose model path, artifacts directory
- [x] Worker threads (stage 1 & 2)
- [x] Hardware acceleration

**Tracking & Detection (50):**
- [x] Confidence threshold
- [x] ROI configuration (padding, interval, smoothing, persistence)
- [x] Optical flow settings (DIS, sparse, finest scale)
- [x] Output sensitivity, amplification, class multipliers
- [x] Oscillation detector (grid, sensitivity, decay)
- [x] Class filtering checkboxes

**Funscript Generation (35):**
- [x] Tracking axis mode, range processing
- [x] User ROI selection and amplification
- [x] Interactive refinement (scale, center, clamps)
- [x] Smoothing (Savitzky-Golay) settings
- [x] Simplification (RDP) settings
- [x] Calibration and tuning

**Post-Processing (20+):**
- [x] Auto post-processing toggle
- [x] Per-chapter profiles
- [x] Per-position processing profiles
- [x] Final RDP pass
- [x] Plugin parameters (dynamic)

**File Output (15):**
- [x] Output folder path
- [x] Generate .funscript, .roll
- [x] Batch mode overwrite strategy
- [x] Stage 2 database retention
- [x] Autosave to video location

**Device Control (25):**
- [x] Device discovery and selection
- [x] Backend selection (Handy, OSR2, Buttplug)
- [x] Connection settings
- [x] Device-specific controls
- [x] Live tracking integration

**Streamer (10):**
- [x] XBVR host, port
- [x] Connection status
- [x] Client monitoring

**Keyboard Shortcuts (30):**
- [x] All navigation shortcuts
- [x] All editing shortcuts
- [x] All project shortcuts
- [x] All chapter shortcuts

---

## Benefits of New Design

### For Users
1. **Single location** for all settings
2. **Easy discovery** of features via search
3. **Logical organization** by category
4. **Less clutter** in main interface
5. **Faster access** to commonly used settings

### For Developers
1. **Centralized** settings management
2. **Easier maintenance** of related settings
3. **Consistent** UI patterns
4. **Better feature organization**
5. **Cleaner codebase** separation

### For UI
1. **Minimal side panel** (Run + Post-Processing only)
2. **More space** for video and timeline
3. **Modern design** with vertical tabs
4. **Scalable** for future features
5. **Professional appearance**

---

## Example Category: Tracking & Detection

### Visual Layout
```
┌─────────────────────────────────────────────────────────────┐
│ Tracking & Detection                                         │
├──────────────────────────────────────────────────────────────┤
│ [General] [ROI] [Optical Flow] [Sensitivity] [Oscillation]  │
│ ┌────────────────────────────────────────────────────────┐  │
│ │ General Tab                                            │  │
│ │                                                        │  │
│ │ Confidence Threshold: [======●=====] 0.45             │  │
│ │ Tracker Mode Default: [Combo: Auto ▾]                 │  │
│ │                                                        │  │
│ │ ROI Configuration                                      │  │
│ │ ─────────────────                                      │  │
│ │ Padding (px): [50    ] ⓘ                             │  │
│ │ Update Interval (frames): [5     ] ⓘ                 │  │
│ │ Smoothing Factor: [==●=========] 0.2 ⓘ               │  │
│ │ Persistence Frames: [30    ] ⓘ                       │  │
│ │                                                        │  │
│ │ Output Processing                                      │  │
│ │ ────────────────                                       │  │
│ │ Output Sensitivity: [======●====] 1.0 ⓘ              │  │
│ │ Signal Amplification: [====●======] 1.0 ⓘ            │  │
│ │ Output Delay (frames): [0     ] ⓘ                    │  │
│ │                                                        │  │
│ └────────────────────────────────────────────────────────┘  │
│                                                              │
│ Search: [_____________________________] [Clear]              │
└──────────────────────────────────────────────────────────────┘
```

---

## Notes
- Settings marked with ⓘ have tooltips with detailed explanations
- Sliders show current value on right
- All changes auto-save immediately
- Search bar available on every tab
- Vertical tab selection persists between sessions
- Window size/position saved to settings.json

---

## Status: DESIGN COMPLETE ✓
Ready for implementation.
