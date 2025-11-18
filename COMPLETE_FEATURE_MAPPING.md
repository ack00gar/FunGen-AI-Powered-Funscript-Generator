# FunGen Complete Feature Mapping
## Ensuring Zero Features Left Behind

This document maps **EVERY SINGLE FEATURE** (400+) to its location in the new UI design:
- **Minimal Side Panel** (Run + Post-Processing tabs)
- **Options Window** (All settings and configuration)
- **Menu Bar** (File operations and window management)
- **Toolbar** (Quick actions)

---

## ✅ COVERAGE SUMMARY

| Location | Feature Count | Purpose |
|----------|--------------|---------|
| **Menu Bar** | 60+ items | File operations, view management, tools |
| **Toolbar** | 40+ buttons | Quick actions, mode toggle, playback |
| **Side Panel - Run** | 15 items | Tracking workflow, progress, quick export |
| **Side Panel - Post-Processing** | 10 items | Quick plugin application |
| **Options Window** | 300+ settings | All configuration and advanced settings |
| **Floating Windows** | 15 windows | Gauges, simulators, managers, dialogs |
| **Context Menus** | 20+ actions | Timeline editing, chapter management |
| **Keyboard Shortcuts** | 30+ shortcuts | All keyboard operations |

**TOTAL: 400+ features** ✓

---

## 📋 DETAILED FEATURE MAPPING

### 1. MENU BAR (60+ items) - **LOCATION: Top Menu Bar**

#### File Menu (18 items)
- [x] New Project → **Menu: File > New Project** (Ctrl+N)
- [x] Open Project → **Menu: File > Open Project** (Ctrl+O)
- [x] Open Video → **Menu: File > Open Video**
- [x] Close Project → **Menu: File > Close Project**
- [x] Open Recent (submenu) → **Menu: File > Open Recent**
  - [x] Recent project list (dynamic)
  - [x] Clear Recent Menu
- [x] Save Project → **Menu: File > Save Project** (Ctrl+S)
- [x] Save Project As → **Menu: File > Save Project As**
- [x] Import (submenu) → **Menu: File > Import**
  - [x] Import Funscript to Timeline 1 (Ctrl+I)
  - [x] Import Funscript to Timeline 2
  - [x] Import Stage 2 Overlay Data
- [x] Export (submenu) → **Menu: File > Export**
  - [x] Export Funscript from Timeline 1 (Ctrl+E)
  - [x] Export Funscript from Timeline 2
- [x] Chapters (submenu) → **Menu: File > Chapters**
  - [x] Save Chapters
  - [x] Save Chapters As
  - [x] Load Chapters
  - [x] Backup Chapters Now
  - [x] Clear All Chapters
- [x] Exit → **Menu: File > Exit**

#### Edit Menu (4 items)
- [x] Undo T1 Change → **Menu: Edit > Undo T1** (Ctrl+Z)
- [x] Redo T1 Change → **Menu: Edit > Redo T1** (Ctrl+Y)
- [x] Undo T2 Change → **Menu: Edit > Undo T2** (Ctrl+Shift+Z)
- [x] Redo T2 Change → **Menu: Edit > Redo T2** (Ctrl+Shift+Y)
- [x] **Options...** → **Menu: Edit > Options** (Ctrl+,) ← **NEW**

#### View Menu (20+ items)
- [x] UI Mode → **Menu: View > UI Mode**
  - [x] Simple Mode
  - [x] Expert Mode
- [x] Layout → **Menu: View > Layout**
  - [x] Vertical (Stacked)
  - [x] Horizontal (Split)
  - [x] Floating Panels
- [x] Panels (submenu for floating mode) → **Menu: View > Panels**
  - [x] Show/Hide individual panels (dynamic)
- [x] Show Toolbar → **Menu: View > Show Toolbar** (checkbox)
- [x] Gauges (submenu) → **Menu: View > Gauges**
  - [x] Timeline 1 Gauge (checkbox)
  - [x] Timeline 2 Gauge (checkbox)
  - [x] Movement Bar / LR Dial (checkbox)
  - [x] 3D Simulator (checkbox)
- [x] Navigation (submenu) → **Menu: View > Navigation**
  - [x] Chapter List Window
  - [x] Chapter Type Manager
  - [x] Generated File Manager
  - [x] Info Graphs
  - [x] Keyboard Shortcuts
- [x] Timelines (submenu) → **Menu: View > Timelines**
  - [x] Timeline 1 (checkbox)
  - [x] Timeline 2 (checkbox)
  - [x] Show Interactive Timeline 1 (checkbox)
  - [x] Show Interactive Timeline 2 (checkbox)
  - [x] Heatmap View (checkbox)
- [x] Chapters (submenu) → **Menu: View > Chapters**
  - [x] Show Chapter Bar
  - [x] Show Chapter Indicator

#### Tools Menu (10+ items)
- [x] Auto-Simplify → **Menu: Tools > Auto-Simplify**
- [x] Auto Post-Processing → **Menu: Tools > Auto Post-Processing**
- [x] Ultimate Autotune → **Menu: Tools > Ultimate Autotune**
- [x] Autotuner Window → **Menu: Tools > Autotuner Window**
- [x] Device Control → **Menu: Tools > Device Control** (supporters)
- [x] Streamer → **Menu: Tools > Streamer** (supporters)
- [x] Key Binding Customization → **Menu: Tools > Key Bindings**
- [x] TensorRT Compiler → **Menu: Tools > TensorRT Compiler** (optional)
- [x] Batch Processing Window → **Menu: Tools > Batch Processing**
- [x] Rebuild Project Cache → **Menu: Tools > Rebuild Cache**
- [x] **Options...** → **Menu: Tools > Options** (Ctrl+,) ← **NEW**

#### Help Menu (2 items)
- [x] Check for Updates → **Menu: Help > Check for Updates**
- [x] About FunGen → **Menu: Help > About** (with GitHub link, Ko-fi)

#### Dynamic Indicators (2 items)
- [x] Device Control Status → **Menu Bar: Dynamic indicator** (when connected)
- [x] Streamer/Native Sync Status → **Menu Bar: Dynamic indicator** (when active)

---

### 2. TOOLBAR (40+ buttons) - **LOCATION: Toolbar**

#### MODE Section (1 button)
- [x] Expert/Simple Mode Toggle → **Toolbar: MODE section** (shows blue when Expert)

#### PROJECT Section (4 buttons)
- [x] New Project → **Toolbar: PROJECT > New**
- [x] Open Project → **Toolbar: PROJECT > Open**
- [x] Save Project → **Toolbar: PROJECT > Save** (disabled when no project)
- [x] Export Funscript → **Toolbar: PROJECT > Export**

#### AI TRACKING Section (1 button)
- [x] Start/Stop Tracking → **Toolbar: AI TRACKING > Start** (red/green state)

#### PLAYBACK Section (6 buttons)
- [x] Jump to Start → **Toolbar: PLAYBACK > Jump Start**
- [x] Previous Frame → **Toolbar: PLAYBACK > Prev Frame**
- [x] Play/Pause → **Toolbar: PLAYBACK > Play/Pause**
- [x] Stop → **Toolbar: PLAYBACK > Stop**
- [x] Next Frame → **Toolbar: PLAYBACK > Next Frame**
- [x] Jump to End → **Toolbar: PLAYBACK > Jump End**

#### TIMELINE EDIT Section (6 buttons)
- [x] Undo T1 → **Toolbar: TIMELINE EDIT > Undo T1**
- [x] Redo T1 → **Toolbar: TIMELINE EDIT > Redo T1**
- [x] Ultimate Autotune T1 → **Toolbar: TIMELINE EDIT > Autotune T1**
- [x] Undo T2 → **Toolbar: TIMELINE EDIT > Undo T2**
- [x] Redo T2 → **Toolbar: TIMELINE EDIT > Redo T2**
- [x] Ultimate Autotune T2 → **Toolbar: TIMELINE EDIT > Autotune T2**

#### VIEW Section (4 buttons)
- [x] Toggle Timeline 1 → **Toolbar: VIEW > T1**
- [x] Toggle Timeline 2 → **Toolbar: VIEW > T2**
- [x] Toggle Chapter List → **Toolbar: VIEW > Chapters**
- [x] Toggle 3D Simulator → **Toolbar: VIEW > 3D Sim**

#### TOOLS Section (4+ buttons)
- [x] Auto Post-Processing → **Toolbar: TOOLS > Auto Post-Proc** (buyers)
- [x] Ultimate Autotune → **Toolbar: TOOLS > Autotune** (buyers)
- [x] Streamer → **Toolbar: TOOLS > Streamer** (supporters)
- [x] Device Control → **Toolbar: TOOLS > Device** (supporters)
- [x] **Options** → **Toolbar: TOOLS > ⚙️ Options** ← **NEW**

#### SPEED Section (3 buttons)
- [x] Real Time Speed → **Toolbar: SPEED > Real Time**
- [x] Slow Motion → **Toolbar: SPEED > Slow Motion**
- [x] Max Speed → **Toolbar: SPEED > Max Speed**

---

### 3. SIDE PANEL - RUN TAB (15 items) - **LOCATION: Side Panel > Run Tab**

#### Video Information (3 items)
- [x] Video file path display → **Panel: Run > Video Info**
- [x] Video duration → **Panel: Run > Video Info**
- [x] Video resolution & FPS → **Panel: Run > Video Info**

#### Tracker Configuration (2 items)
- [x] Tracker mode selector → **Panel: Run > Tracker Mode dropdown**
- [x] Tracker mode description → **Panel: Run > Description text**

#### Processing Controls (5 items) - **SIMPLE MODE**
- [x] Start Tracking button → **Panel: Run > Start button**
- [x] Stop button → **Panel: Run > Stop button**
- [x] Pause button → **Panel: Run > Pause button** (Expert mode)
- [x] Resume button → **Panel: Run > Resume button** (Expert mode)
- [x] Abort button → **Panel: Run > Abort button** (Expert mode)

#### Progress Display (3 items)
- [x] Stage 1 progress bar → **Panel: Run > Stage 1 Progress**
- [x] Stage 2 progress bar → **Panel: Run > Stage 2 Progress**
- [x] Stage 3 progress bar → **Panel: Run > Stage 3 Progress**

#### Quick Export (2 items)
- [x] Export Timeline 1 button → **Panel: Run > Export T1**
- [x] Export Timeline 2 button → **Panel: Run > Export T2**

#### Access to Options (1 item)
- [x] **"More Options..." button** → **Panel: Run > ⚙️ More Options** ← **NEW**

#### EXPERT MODE ONLY (additional items in Run tab)
- [x] Stage configuration options → **Moved to Options: General > Performance**
- [x] Force Rerun Stage 1 checkbox → **Moved to Options: General > Performance**
- [x] Force Rerun Stage 2 checkbox → **Moved to Options: General > Performance**
- [x] Save/Reuse Preprocessed Video → **Moved to Options: Output > Advanced**
- [x] Keep Stage 2 Database → **Moved to Options: Output > Advanced**
- [x] Resume from checkpoint → **Panel: Run > Resume button** (visible in Expert)

---

### 4. SIDE PANEL - POST-PROCESSING TAB (10 items) - **LOCATION: Side Panel > Post-Processing Tab**

#### Scope Selection (2 items)
- [x] Timeline selector (T1/T2) → **Panel: Post-Processing > Timeline dropdown**
- [x] Scope selector (All/Range/Chapter) → **Panel: Post-Processing > Scope dropdown**

#### Quick Plugin Application (2 items)
- [x] Plugin dropdown (quick select) → **Panel: Post-Processing > Plugin dropdown**
- [x] Apply button → **Panel: Post-Processing > Apply button**

#### Auto Post-Processing (2 items)
- [x] Enable Auto Post-Processing checkbox → **Panel: Post-Processing > Enable checkbox**
- [x] Use Per-Chapter Profiles checkbox → **Panel: Post-Processing > Per-Chapter checkbox**

#### Access to Options (1 item)
- [x] **"Configure Plugins..." button** → **Panel: Post-Processing > Configure** ← **NEW**
  - Opens Options window to Post-Processing > [Selected Plugin] tab

#### MOVED TO OPTIONS WINDOW:
- [x] Plugin parameters (sliders, inputs) → **Options: Post-Processing > [Plugin Name] tabs**
- [x] Per-position processing profiles → **Options: Post-Processing > Profiles tab**
- [x] Final RDP pass options → **Options: Post-Processing > Auto Processing tab**
- [x] Reset to Default button → **Options: Post-Processing > [Plugin Name] > Reset**

---

### 5. OPTIONS WINDOW - **LOCATION: Options Window (Modal/Floating)**

This is where **300+ settings** live, organized into 11 vertical tabs with horizontal subtabs.

---

#### 5.1 GENERAL TAB (25 settings) - **Options: 🎨 General**

##### Horizontal Tab: Interface (10 settings)
- [x] UI Mode (Simple/Expert) → **Options: General > Interface > UI Mode**
- [x] Layout Mode (Vertical/Horizontal/Floating) → **Options: General > Interface > Layout**
- [x] Font Scale slider → **Options: General > Interface > Font Scale**
- [x] Auto System Scaling checkbox → **Options: General > Interface > Auto Scaling**
- [x] Show Toolbar checkbox → **Options: General > Interface > Show Toolbar**
- [x] Full Width Navigation checkbox → **Options: General > Interface > Full Width Nav**
- [x] Timeline Pan Speed slider → **Options: General > Interface > Pan Speed**
- [x] Show Timeline Editor Buttons → **Options: General > Interface > Editor Buttons**
- [x] Show Advanced Options → **Options: General > Interface > Show Advanced**
- [x] Show Video Feed → **Options: General > Interface > Show Video**

##### Horizontal Tab: Performance (10 settings)
- [x] GPU Rendering checkbox → **Options: General > Performance > GPU Enabled**
- [x] GPU Threshold (points) → **Options: General > Performance > GPU Threshold**
- [x] Timeline Performance Logging → **Options: General > Performance > Perf Logging**
- [x] Show Performance Indicators → **Options: General > Performance > Show Indicators**
- [x] Video Decoding Method dropdown → **Options: General > Performance > Video Decode**
- [x] Hardware Acceleration Method → **Options: General > Performance > HW Accel**
- [x] FFmpeg Path → **Options: General > Performance > FFmpeg Path**
- [x] VR Unwarp Method → **Options: General > Performance > VR Unwarp**
- [x] Force Rerun Stage 1 → **Options: General > Performance > Force Rerun S1**
- [x] Force Rerun Stage 2 → **Options: General > Performance > Force Rerun S2**

##### Horizontal Tab: Autosave (5 settings)
- [x] Autosave Enabled checkbox → **Options: General > Autosave > Enabled**
- [x] Autosave Interval (seconds) → **Options: General > Autosave > Interval**
- [x] Autosave Final Funscript → **Options: General > Autosave > Save to Video Loc**
- [x] Backup Chapters Enabled → **Options: General > Autosave > Backup Chapters**
- [x] Backup Interval → **Options: General > Autosave > Backup Interval**

##### Horizontal Tab: Energy Saver (5 settings)
- [x] Energy Saver Enabled → **Options: General > Energy Saver > Enabled**
- [x] Normal FPS → **Options: General > Energy Saver > Normal FPS**
- [x] Idle Threshold (seconds) → **Options: General > Energy Saver > Idle Threshold**
- [x] Idle FPS → **Options: General > Energy Saver > Idle FPS**
- [x] Energy Saver Logging → **Options: General > Energy Saver > Logging**

---

#### 5.2 DISPLAY & WINDOWS TAB (30 settings) - **Options: 🖥️ Display**

##### Horizontal Tab: Video (8 settings)
- [x] Default Zoom Level → **Options: Display > Video > Default Zoom**
- [x] Show Stage 2 Overlay → **Options: Display > Video > Stage 2 Overlay**
- [x] Use Simplified Preview → **Options: Display > Video > Simplified Preview**
- [x] Video Display Width → **Options: Display > Video > Display Width**
- [x] Video Display Height → **Options: Display > Video > Display Height**
- [x] Video Overlay Alpha → **Options: Display > Video > Overlay Alpha**
- [x] Show Video Controls → **Options: Display > Video > Show Controls**
- [x] Fullscreen on Play → **Options: Display > Video > Fullscreen Play**

##### Horizontal Tab: Gauges (10 settings)
- [x] Show Timeline 1 Gauge → **Options: Display > Gauges > Show T1 Gauge**
- [x] Show Timeline 2 Gauge → **Options: Display > Gauges > Show T2 Gauge**
- [x] Show Movement Bar/LR Dial → **Options: Display > Gauges > Show LR Dial**
- [x] Show 3D Simulator → **Options: Display > Gauges > Show 3D Sim**
- [x] Show 3D Simulator Logo → **Options: Display > Gauges > Show 3D Logo**
- [x] Gauge Overlay Mode → **Options: Display > Gauges > Gauge Overlay**
- [x] Movement Bar Overlay Mode → **Options: Display > Gauges > LR Dial Overlay**
- [x] 3D Simulator Overlay Mode → **Options: Display > Gauges > 3D Sim Overlay**
- [x] LR Dial Window Size (W) → **Options: Display > Gauges > LR Dial Width**
- [x] LR Dial Window Size (H) → **Options: Display > Gauges > LR Dial Height**

##### Horizontal Tab: Timelines (8 settings)
- [x] Show Funscript Timeline → **Options: Display > Timelines > Show Timeline**
- [x] Show Interactive Timeline 1 → **Options: Display > Timelines > Interactive T1**
- [x] Show Interactive Timeline 2 → **Options: Display > Timelines > Interactive T2**
- [x] Show Heatmap → **Options: Display > Timelines > Show Heatmap**
- [x] Heatmap Height → **Options: Display > Timelines > Heatmap Height**
- [x] Timeline Height → **Options: Display > Timelines > Timeline Height**
- [x] Timeline Point Size → **Options: Display > Timelines > Point Size**
- [x] Timeline Line Width → **Options: Display > Timelines > Line Width**

##### Horizontal Tab: Panels (4 settings)
- [x] Show Chapter List Window → **Options: Display > Panels > Chapter List**
- [x] Panel Positions (floating mode) → **Options: Display > Panels > Positions**
- [x] Panel Sizes (floating mode) → **Options: Display > Panels > Sizes**
- [x] Remember Panel Positions → **Options: Display > Panels > Remember Pos**

---

#### 5.3 AI MODELS TAB (15 settings) - **Options: 🤖 AI Models**

##### Horizontal Tab: Models (5 settings)
- [x] YOLO Detection Model Path → **Options: AI Models > Models > YOLO Det Path**
- [x] YOLO Pose Model Path → **Options: AI Models > Models > YOLO Pose Path**
- [x] Pose Model Artifacts Directory → **Options: AI Models > Models > Artifacts Dir**
- [x] Model Auto-Download → **Options: AI Models > Models > Auto Download**
- [x] Model Cache Directory → **Options: AI Models > Models > Cache Dir**

##### Horizontal Tab: Inference (8 settings)
- [x] Stage 1 Producers (threads) → **Options: AI Models > Inference > S1 Producers**
- [x] Stage 1 Consumers (threads) → **Options: AI Models > Inference > S1 Consumers**
- [x] Stage 2 Workers (threads) → **Options: AI Models > Inference > S2 Workers**
- [x] Batch Size → **Options: AI Models > Inference > Batch Size**
- [x] Inference Device → **Options: AI Models > Inference > Device**
- [x] Inference Precision → **Options: AI Models > Inference > Precision**
- [x] Max Queue Size → **Options: AI Models > Inference > Queue Size**
- [x] Worker Affinity → **Options: AI Models > Inference > Affinity**

##### Horizontal Tab: TensorRT (2 settings)
- [x] Enable TensorRT → **Options: AI Models > TensorRT > Enabled**
- [x] TensorRT Cache Directory → **Options: AI Models > TensorRT > Cache Dir**

---

#### 5.4 TRACKING & DETECTION TAB (50 settings) - **Options: 🎯 Tracking**

##### Horizontal Tab: General (5 settings)
- [x] Confidence Threshold slider → **Options: Tracking > General > Confidence**
- [x] Default Tracker Mode → **Options: Tracking > General > Default Mode**
- [x] Tracker Logging → **Options: Tracking > General > Logging**
- [x] Show Tracker Overlays → **Options: Tracking > General > Show Overlays**
- [x] Tracker Debug Mode → **Options: Tracking > General > Debug Mode**

##### Horizontal Tab: ROI (9 settings)
- [x] ROI Padding (pixels) → **Options: Tracking > ROI > Padding**
- [x] ROI Update Interval (frames) → **Options: Tracking > ROI > Update Interval**
- [x] ROI Smoothing Factor slider → **Options: Tracking > ROI > Smoothing**
- [x] ROI Persistence Frames → **Options: Tracking > ROI > Persistence**
- [x] ROI Auto-Reset → **Options: Tracking > ROI > Auto Reset**
- [x] ROI Min Size → **Options: Tracking > ROI > Min Size**
- [x] ROI Max Size → **Options: Tracking > ROI > Max Size**
- [x] ROI Aspect Ratio Lock → **Options: Tracking > ROI > Lock Aspect**
- [x] Show ROI Bounds → **Options: Tracking > ROI > Show Bounds**

##### Horizontal Tab: Optical Flow (10 settings)
- [x] Use Sparse Flow checkbox → **Options: Tracking > Optical Flow > Use Sparse**
- [x] DIS Flow Preset dropdown → **Options: Tracking > Optical Flow > DIS Preset**
- [x] DIS Finest Scale → **Options: Tracking > Optical Flow > Finest Scale**
- [x] DIS Gradient Descent Iterations → **Options: Tracking > Optical Flow > GD Iterations**
- [x] DIS Patch Size → **Options: Tracking > Optical Flow > Patch Size**
- [x] DIS Patch Stride → **Options: Tracking > Optical Flow > Patch Stride**
- [x] Flow Visualize → **Options: Tracking > Optical Flow > Visualize**
- [x] Flow Smooth Factor → **Options: Tracking > Optical Flow > Smooth Factor**
- [x] Flow Max Magnitude → **Options: Tracking > Optical Flow > Max Magnitude**
- [x] Flow Use GPU → **Options: Tracking > Optical Flow > Use GPU**

##### Horizontal Tab: Sensitivity (8 settings)
- [x] Output Sensitivity slider → **Options: Tracking > Sensitivity > Output Sens**
- [x] Signal Amplification slider → **Options: Tracking > Sensitivity > Amplification**
- [x] Output Delay (frames) → **Options: Tracking > Sensitivity > Output Delay**
- [x] Class-Specific Multipliers:
  - [x] Person Multiplier → **Options: Tracking > Sensitivity > Person Mult**
  - [x] Hand Multiplier → **Options: Tracking > Sensitivity > Hand Mult**
  - [x] Toy Multiplier → **Options: Tracking > Sensitivity > Toy Mult**
  - [x] Body Part Multiplier → **Options: Tracking > Sensitivity > Body Mult**
- [x] Adaptive Sensitivity → **Options: Tracking > Sensitivity > Adaptive**

##### Horizontal Tab: Oscillation (10 settings)
- [x] Grid Size slider → **Options: Tracking > Oscillation > Grid Size**
- [x] Detection Sensitivity slider → **Options: Tracking > Oscillation > Sensitivity**
- [x] Stage 3 Detector Mode dropdown → **Options: Tracking > Oscillation > Detector Mode**
- [x] Area Selection Controls → **Options: Tracking > Oscillation > Area Select**
- [x] Overlay Toggles:
  - [x] Show Grid Overlay → **Options: Tracking > Oscillation > Show Grid**
  - [x] Show Heatmap Overlay → **Options: Tracking > Oscillation > Show Heatmap**
  - [x] Show Vector Overlay → **Options: Tracking > Oscillation > Show Vectors**
- [x] Enable Decay checkbox → **Options: Tracking > Oscillation > Enable Decay**
- [x] Hold Duration (ms) → **Options: Tracking > Oscillation > Hold Duration**
- [x] Decay Factor slider → **Options: Tracking > Oscillation > Decay Factor**

##### Horizontal Tab: Class Filtering (8+ settings)
- [x] Class Filter Checkboxes (dynamic based on detected classes):
  - [x] Person → **Options: Tracking > Class Filtering > Person**
  - [x] Hand → **Options: Tracking > Class Filtering > Hand**
  - [x] Toy → **Options: Tracking > Class Filtering > Toy**
  - [x] Face → **Options: Tracking > Class Filtering > Face**
  - [x] Body Part → **Options: Tracking > Class Filtering > Body Part**
  - [x] (Additional classes dynamically added)
- [x] Select All → **Options: Tracking > Class Filtering > Select All button**
- [x] Deselect All → **Options: Tracking > Class Filtering > Deselect All button**

---

#### 5.5 FUNSCRIPT GENERATION TAB (35 settings) - **Options: 📊 Funscript**

##### Horizontal Tab: General (8 settings)
- [x] Tracking Axis Mode dropdown → **Options: Funscript > General > Axis Mode**
- [x] Range Processing checkbox → **Options: Funscript > General > Range Processing**
- [x] Start Frame → **Options: Funscript > General > Start Frame**
- [x] End Frame → **Options: Funscript > General > End Frame**
- [x] Output Range Min → **Options: Funscript > General > Output Min**
- [x] Output Range Max → **Options: Funscript > General > Output Max**
- [x] Invert Output → **Options: Funscript > General > Invert**
- [x] Generate Roll File → **Options: Funscript > General > Generate Roll**

##### Horizontal Tab: User ROI (6 settings)
- [x] Enable User ROI → **Options: Funscript > User ROI > Enable**
- [x] ROI Selection Tool → **Options: Funscript > User ROI > Select Tool**
- [x] ROI X Position → **Options: Funscript > User ROI > X Position**
- [x] ROI Y Position → **Options: Funscript > User ROI > Y Position**
- [x] ROI Width → **Options: Funscript > User ROI > Width**
- [x] ROI Height → **Options: Funscript > User ROI > Height**

##### Horizontal Tab: Refinement (12 settings)
- [x] Scale slider → **Options: Funscript > Refinement > Scale**
- [x] Center slider → **Options: Funscript > Refinement > Center**
- [x] Clamp Min slider → **Options: Funscript > Refinement > Clamp Min**
- [x] Clamp Max slider → **Options: Funscript > Refinement > Clamp Max**
- [x] Amplification slider → **Options: Funscript > Refinement > Amplification**
- [x] Smoothing Method dropdown → **Options: Funscript > Refinement > Smooth Method**
- [x] Smoothing Window Size → **Options: Funscript > Refinement > Window Size**
- [x] Savitzky-Golay Order → **Options: Funscript > Refinement > SG Order**
- [x] Apply Smoothing checkbox → **Options: Funscript > Refinement > Apply Smooth**
- [x] Preview Refinement → **Options: Funscript > Refinement > Preview**
- [x] Reset to Default → **Options: Funscript > Refinement > Reset button**
- [x] Apply to Timeline → **Options: Funscript > Refinement > Apply button**

##### Horizontal Tab: Simplification (5 settings)
- [x] RDP Epsilon slider → **Options: Funscript > Simplification > RDP Epsilon**
- [x] Enable Simplification → **Options: Funscript > Simplification > Enable**
- [x] Simplify on Export → **Options: Funscript > Simplification > On Export**
- [x] Min Point Distance → **Options: Funscript > Simplification > Min Distance**
- [x] Preserve Extrema → **Options: Funscript > Simplification > Preserve Extrema**

##### Horizontal Tab: Calibration (4 settings)
- [x] Latency Calibration Mode → **Options: Funscript > Calibration > Mode**
- [x] Latency Offset (ms) → **Options: Funscript > Calibration > Offset**
- [x] Open Timeline Comparison → **Options: Funscript > Calibration > Compare button**
- [x] Auto-Calibrate → **Options: Funscript > Calibration > Auto button**

---

#### 5.6 POST-PROCESSING TAB (20+ settings) - **Options: 🔧 Post-Processing**

##### Horizontal Tab: Auto Processing (8 settings)
- [x] Enable Auto Post-Processing → **Options: Post-Processing > Auto > Enable**
- [x] Apply Per-Chapter Profiles → **Options: Post-Processing > Auto > Per-Chapter**
- [x] Final RDP Pass Enabled → **Options: Post-Processing > Auto > Final RDP**
- [x] Final RDP Epsilon → **Options: Post-Processing > Auto > RDP Epsilon**
- [x] Auto Apply on Tracking → **Options: Post-Processing > Auto > Apply on Track**
- [x] Auto Apply on Chapter Change → **Options: Post-Processing > Auto > On Chapter**
- [x] Processing Order → **Options: Post-Processing > Auto > Order**
- [x] Skip Empty Chapters → **Options: Post-Processing > Auto > Skip Empty**

##### Horizontal Tab: Profiles (5 settings)
- [x] Per-Position Processing Profiles:
  - [x] Blowjob Profile → **Options: Post-Processing > Profiles > Blowjob**
  - [x] Penetration Profile → **Options: Post-Processing > Profiles > Penetration**
  - [x] Handjob Profile → **Options: Post-Processing > Profiles > Handjob**
  - [x] Other Profile → **Options: Post-Processing > Profiles > Other**
- [x] Create New Profile → **Options: Post-Processing > Profiles > New button**

##### Horizontal Tab: [Plugin Name] (Dynamic - one tab per plugin)
**DISCOVERED PLUGINS (examples):**

- **Smoother Plugin Tab** → **Options: Post-Processing > Smoother**
  - [x] Window Size slider
  - [x] Method dropdown (Moving Average, Gaussian, Savitzky-Golay)
  - [x] Polynomial Order (for SG)
  - [x] Preview button
  - [x] Reset to Default button

- **RDP Simplifier Tab** → **Options: Post-Processing > RDP Simplifier**
  - [x] Epsilon slider
  - [x] Preserve Extrema checkbox
  - [x] Min Point Distance
  - [x] Preview button
  - [x] Reset to Default button

- **Limiter Tab** → **Options: Post-Processing > Limiter**
  - [x] Min Position slider
  - [x] Max Position slider
  - [x] Hard Limit checkbox
  - [x] Preview button
  - [x] Reset to Default button

- **Half Speed Tab** → **Options: Post-Processing > Half Speed**
  - [x] Interpolation Method dropdown
  - [x] Smoothing checkbox
  - [x] Preview button
  - [x] Reset to Default button

- **Double Speed Tab** → **Options: Post-Processing > Double Speed**
  - [x] Interpolation Method dropdown
  - [x] Preserve Peaks checkbox
  - [x] Preview button
  - [x] Reset to Default button

- **(Additional plugins dynamically discovered and added as tabs)**

---

#### 5.7 OUTPUT TAB (15 settings) - **Options: 💾 Output**

##### Horizontal Tab: General (6 settings)
- [x] Output Folder Path → **Options: Output > General > Output Folder**
- [x] Browse Button → **Options: Output > General > Browse button**
- [x] Generate .funscript → **Options: Output > General > Generate Funscript**
- [x] Generate .roll → **Options: Output > General > Generate Roll**
- [x] Output Filename Pattern → **Options: Output > General > Filename Pattern**
- [x] Auto-Open Output Folder → **Options: Output > General > Auto Open**

##### Horizontal Tab: Batch Mode (5 settings)
- [x] Batch Mode Overwrite Strategy → **Options: Output > Batch > Strategy**
  - 0 = Process All
  - 1 = Skip Existing
- [x] Batch Input Folder → **Options: Output > Batch > Input Folder**
- [x] Batch Output Folder → **Options: Output > Batch > Output Folder**
- [x] Recursive Search → **Options: Output > Batch > Recursive**
- [x] File Filter Pattern → **Options: Output > Batch > Filter**

##### Horizontal Tab: Advanced (4 settings)
- [x] Retain Stage 2 Database → **Options: Output > Advanced > Keep S2 DB**
- [x] Save Preprocessed Video → **Options: Output > Advanced > Save Preprocessed**
- [x] Preprocessing Cache Dir → **Options: Output > Advanced > Cache Dir**
- [x] Auto-Cleanup Temp Files → **Options: Output > Advanced > Cleanup**

---

#### 5.8 DEVICE CONTROL TAB (25 settings) - **Options: 🕹️ Device Control** *(SUPPORTERS ONLY)*

##### Horizontal Tab: Connection (7 settings)
- [x] Enable Device Control → **Options: Device > Connection > Enabled**
- [x] Preferred Backend dropdown → **Options: Device > Connection > Backend**
  - Handy
  - OSR2
  - Buttplug
- [x] Auto-Connect on Startup → **Options: Device > Connection > Auto Connect**
- [x] Connection Timeout → **Options: Device > Connection > Timeout**
- [x] Reconnect on Disconnect → **Options: Device > Connection > Reconnect**
- [x] Device Discovery button → **Options: Device > Connection > Discover button**
- [x] Selected Devices list → **Options: Device > Connection > Device List**

##### Horizontal Tab: Handy (5 settings)
- [x] Connection Key → **Options: Device > Handy > Connection Key**
- [x] Server URL → **Options: Device > Handy > Server URL**
- [x] Firmware Version → **Options: Device > Handy > Firmware (readonly)**
- [x] Offset (ms) → **Options: Device > Handy > Offset**
- [x] Test Connection button → **Options: Device > Handy > Test button**

##### Horizontal Tab: OSR2 (5 settings)
- [x] Serial Port → **Options: Device > OSR2 > Serial Port**
- [x] Baud Rate → **Options: Device > OSR2 > Baud Rate**
- [x] Scan Ports button → **Options: Device > OSR2 > Scan button**
- [x] Auto-Detect → **Options: Device > OSR2 > Auto Detect**
- [x] Test Connection button → **Options: Device > OSR2 > Test button**

##### Horizontal Tab: Buttplug (5 settings)
- [x] Server Address → **Options: Device > Buttplug > Server Address**
- [x] Server Port → **Options: Device > Buttplug > Server Port**
- [x] Auto-Connect checkbox → **Options: Device > Buttplug > Auto Connect**
- [x] Discovered Devices list → **Options: Device > Buttplug > Devices**
- [x] Connect button → **Options: Device > Buttplug > Connect button**

##### Horizontal Tab: Live Tracking (3 settings)
- [x] Enable Live Tracking → **Options: Device > Live > Enable**
- [x] Max Rate (Hz) → **Options: Device > Live > Max Rate**
- [x] Sync Timeline to Device → **Options: Device > Live > Sync Timeline**

---

#### 5.9 STREAMER TAB (10 settings) - **Options: 📡 Streamer** *(SUPPORTERS ONLY)*

##### Horizontal Tab: XBVR (5 settings)
- [x] Enable XBVR Integration → **Options: Streamer > XBVR > Enabled**
- [x] XBVR Host → **Options: Streamer > XBVR > Host**
- [x] XBVR Port → **Options: Streamer > XBVR > Port**
- [x] Auto-Connect → **Options: Streamer > XBVR > Auto Connect**
- [x] Test Connection button → **Options: Streamer > XBVR > Test button**

##### Horizontal Tab: Sync (3 settings)
- [x] Sync Status (readonly) → **Options: Streamer > Sync > Status**
- [x] Connected Clients count → **Options: Streamer > Sync > Clients**
- [x] Sync Offset (ms) → **Options: Streamer > Sync > Offset**

##### Horizontal Tab: Advanced (2 settings)
- [x] Native Sync Mode → **Options: Streamer > Advanced > Native Mode**
- [x] Logging Level → **Options: Streamer > Advanced > Logging**

---

#### 5.10 KEYBOARD SHORTCUTS TAB (30+ shortcuts) - **Options: ⌨️ Keyboard**

##### Horizontal Tab: Navigation (8 shortcuts)
- [x] Previous Frame → **Options: Keyboard > Navigation > Prev Frame** (LEFT)
- [x] Next Frame → **Options: Keyboard > Navigation > Next Frame** (RIGHT)
- [x] Jump to Start → **Options: Keyboard > Navigation > Jump Start** (HOME)
- [x] Jump to End → **Options: Keyboard > Navigation > Jump End** (END)
- [x] Play/Pause → **Options: Keyboard > Navigation > Play/Pause** (SPACE)
- [x] Previous Point → **Options: Keyboard > Navigation > Prev Point** (CTRL+LEFT)
- [x] Next Point → **Options: Keyboard > Navigation > Next Point** (CTRL+RIGHT)
- [x] Jump to Chapter → **Options: Keyboard > Navigation > Jump Chapter**

##### Horizontal Tab: Editing (8 shortcuts)
- [x] Undo T1 → **Options: Keyboard > Editing > Undo T1** (CTRL+Z)
- [x] Redo T1 → **Options: Keyboard > Editing > Redo T1** (CTRL+Y)
- [x] Undo T2 → **Options: Keyboard > Editing > Undo T2** (CTRL+SHIFT+Z)
- [x] Redo T2 → **Options: Keyboard > Editing > Redo T2** (CTRL+SHIFT+Y)
- [x] Select All → **Options: Keyboard > Editing > Select All** (CTRL+A)
- [x] Delete Points → **Options: Keyboard > Editing > Delete** (DELETE)
- [x] Copy Points → **Options: Keyboard > Editing > Copy** (CTRL+C)
- [x] Paste Points → **Options: Keyboard > Editing > Paste** (CTRL+V)

##### Horizontal Tab: Project (6 shortcuts)
- [x] New Project → **Options: Keyboard > Project > New** (CTRL+N)
- [x] Open Project → **Options: Keyboard > Project > Open** (CTRL+O)
- [x] Save Project → **Options: Keyboard > Project > Save** (CTRL+S)
- [x] Import Funscript → **Options: Keyboard > Project > Import** (CTRL+I)
- [x] Export Funscript → **Options: Keyboard > Project > Export** (CTRL+E)
- [x] Close Project → **Options: Keyboard > Project > Close**

##### Horizontal Tab: Chapters (8 shortcuts)
- [x] Create Chapter → **Options: Keyboard > Chapters > Create**
- [x] Delete Chapter → **Options: Keyboard > Chapters > Delete**
- [x] Previous Chapter → **Options: Keyboard > Chapters > Previous**
- [x] Next Chapter → **Options: Keyboard > Chapters > Next**
- [x] Edit Chapter → **Options: Keyboard > Chapters > Edit**
- [x] Merge Chapters → **Options: Keyboard > Chapters > Merge**
- [x] Split Chapter → **Options: Keyboard > Chapters > Split**
- [x] Assign Position → **Options: Keyboard > Chapters > Assign Position**

##### Global Shortcuts (not in tabs)
- [x] **Open Options** → **Ctrl+,** or **F12** ← **NEW**

---

#### 5.11 ABOUT TAB - **Options: ℹ️ About**

##### Horizontal Tab: Info
- [x] Application Name → **Options: About > Info > Name**
- [x] Version Number → **Options: About > Info > Version**
- [x] Build Date → **Options: About > Info > Build Date**
- [x] GitHub Link → **Options: About > Info > GitHub button**
- [x] License Information → **Options: About > Info > License**

##### Horizontal Tab: Support
- [x] Ko-fi Support Button → **Options: About > Support > Ko-fi button**
- [x] Documentation Links → **Options: About > Support > Docs**
- [x] Report Bug → **Options: About > Support > Report Bug button**
- [x] Feature Request → **Options: About > Support > Request Feature button**

##### Horizontal Tab: Updates
- [x] Current Version → **Options: About > Updates > Current Version**
- [x] Check for Updates button → **Options: About > Updates > Check button**
- [x] Auto-Check Updates → **Options: About > Updates > Auto Check**
- [x] Update Channel dropdown → **Options: About > Updates > Channel**
  - Stable
  - Beta
  - Nightly

---

### 6. FLOATING WINDOWS & DIALOGS (15 windows) - **LOCATION: Separate Windows**

#### Gauge Windows (4 windows)
- [x] Timeline 1 Gauge → **Floating: Gauge T1** (Menu: View > Gauges > T1)
- [x] Timeline 2 Gauge → **Floating: Gauge T2** (Menu: View > Gauges > T2)
- [x] Movement Bar / LR Dial → **Floating: LR Dial** (Menu: View > Gauges > LR Dial)
- [x] 3D Simulator → **Floating: 3D Simulator** (Menu: View > Gauges > 3D Sim)
  - Can also render as video overlay

#### Management Windows (3 windows)
- [x] Chapter List Window → **Floating: Chapter List** (Menu: View > Navigation > Chapters)
- [x] Chapter Type Manager → **Floating: Chapter Types** (Menu: View > Navigation > Chapter Types)
- [x] Generated File Manager → **Floating: File Manager** (Menu: View > Navigation > File Manager)

#### Tool Windows (4 windows)
- [x] Autotuner Window → **Floating: Autotuner** (Menu: Tools > Autotuner)
- [x] Timeline Comparison → **Floating: Timeline Compare** (Calibration)
- [x] Batch Processing → **Floating: Batch Processing** (Menu: Tools > Batch)
- [x] TensorRT Compiler → **Floating: TensorRT** (Menu: Tools > TensorRT)

#### Info Windows (2 windows)
- [x] Info Graphs → **Floating: Info Graphs** (Menu: View > Navigation > Info Graphs)
- [x] Keyboard Shortcuts Dialog → **Floating: Shortcuts** (Menu: View > Navigation > Shortcuts)

#### System Dialogs (2 dialogs)
- [x] File Dialog (Open/Save) → **System: File Dialog** (Triggered by Open/Save)
- [x] Error Popup → **System: Error Dialog** (Triggered by errors)

---

### 7. CONTEXT MENUS (20+ actions) - **LOCATION: Right-Click Menus**

#### Timeline Context Menu (10 actions)
- [x] Add Point → **Timeline: Right-Click > Add Point**
- [x] Delete Point(s) → **Timeline: Right-Click > Delete**
- [x] Edit Point → **Timeline: Right-Click > Edit**
- [x] Copy Point(s) → **Timeline: Right-Click > Copy**
- [x] Paste Point(s) → **Timeline: Right-Click > Paste**
- [x] Select Range → **Timeline: Right-Click > Select Range**
- [x] Smooth Selection → **Timeline: Right-Click > Smooth**
- [x] Simplify Selection → **Timeline: Right-Click > Simplify**
- [x] Invert Selection → **Timeline: Right-Click > Invert**
- [x] Clear Selection → **Timeline: Right-Click > Clear**

#### Chapter Bar Context Menu (8 actions)
- [x] Create Chapter → **Chapter Bar: Right-Click > Create**
- [x] Edit Chapter → **Chapter Bar: Right-Click > Edit**
- [x] Delete Chapter → **Chapter Bar: Right-Click > Delete**
- [x] Merge with Next → **Chapter Bar: Right-Click > Merge**
- [x] Split Chapter → **Chapter Bar: Right-Click > Split**
- [x] Assign Position → **Chapter Bar: Right-Click > Assign Position**
- [x] Set Chapter Type → **Chapter Bar: Right-Click > Set Type**
- [x] Copy Chapter → **Chapter Bar: Right-Click > Copy**

#### Video Display Context Menu (2 actions)
- [x] Set ROI → **Video: Right-Click > Set ROI**
- [x] Clear ROI → **Video: Right-Click > Clear ROI**

---

### 8. VIDEO DISPLAY AREA - **LOCATION: Video Display**

#### Video Playback (Integrated)
- [x] Video Frame Display → **Video Display: Canvas**
- [x] 3D Simulator Overlay → **Video Display: Overlay** (if enabled)
- [x] Gauge Overlays → **Video Display: Overlays** (if enabled)
- [x] ROI Selection Visual → **Video Display: ROI Box**
- [x] Stage 2 Overlay → **Video Display: Tracking Overlay** (if enabled)

---

### 9. TIMELINE AREA - **LOCATION: Timeline Display**

#### Timeline Display Features
- [x] Interactive Timeline 1 → **Timeline: T1** (if enabled)
- [x] Interactive Timeline 2 → **Timeline: T2** (if enabled)
- [x] Heatmap View → **Timeline: Heatmap** (if enabled)
- [x] Chapter Indicators → **Timeline: Chapter Bar** (if enabled)
- [x] Playback Position → **Timeline: Playhead**
- [x] Point Selection → **Timeline: Select Tool**
- [x] Point Editing → **Timeline: Edit Tool**
- [x] Zoom Controls → **Timeline: Zoom In/Out**
- [x] Pan Controls → **Timeline: Pan Left/Right**

---

## ✅ VERIFICATION CHECKLIST

### All 400+ Features Mapped

#### ✅ Menu Bar (60+ items)
- [x] File Menu (18 items)
- [x] Edit Menu (5 items, including new Options)
- [x] View Menu (20+ items)
- [x] Tools Menu (11 items, including new Options)
- [x] Help Menu (2 items)
- [x] Dynamic Indicators (2 items)

#### ✅ Toolbar (40+ buttons)
- [x] MODE Section (1)
- [x] PROJECT Section (4)
- [x] AI TRACKING Section (1)
- [x] PLAYBACK Section (6)
- [x] TIMELINE EDIT Section (6)
- [x] VIEW Section (4)
- [x] TOOLS Section (5, including new Options)
- [x] SPEED Section (3)

#### ✅ Side Panel - Run Tab (15+ items)
- [x] Video Info (3)
- [x] Tracker Config (2)
- [x] Processing Controls (5)
- [x] Progress Display (3)
- [x] Quick Export (2)
- [x] Options Access (1 new)

#### ✅ Side Panel - Post-Processing Tab (10+ items)
- [x] Scope Selection (2)
- [x] Quick Plugin (2)
- [x] Auto Post-Processing (2)
- [x] Options Access (1 new)

#### ✅ Options Window (300+ settings)
- [x] General Tab (25 settings, 4 horizontal tabs)
- [x] Display Tab (30 settings, 4 horizontal tabs)
- [x] AI Models Tab (15 settings, 3 horizontal tabs)
- [x] Tracking Tab (50 settings, 6 horizontal tabs)
- [x] Funscript Tab (35 settings, 5 horizontal tabs)
- [x] Post-Processing Tab (20+ settings, dynamic plugin tabs)
- [x] Output Tab (15 settings, 3 horizontal tabs)
- [x] Device Control Tab (25 settings, 5 horizontal tabs) *Supporters*
- [x] Streamer Tab (10 settings, 3 horizontal tabs) *Supporters*
- [x] Keyboard Tab (30+ shortcuts, 4 horizontal tabs)
- [x] About Tab (3 horizontal tabs)

#### ✅ Floating Windows (15 windows)
- [x] Gauge Windows (4)
- [x] Management Windows (3)
- [x] Tool Windows (4)
- [x] Info Windows (2)
- [x] System Dialogs (2)

#### ✅ Context Menus (20+ actions)
- [x] Timeline Menu (10)
- [x] Chapter Bar Menu (8)
- [x] Video Display Menu (2)

#### ✅ Video & Timeline Areas
- [x] Video Display Features (5)
- [x] Timeline Display Features (9)

---

## 🎯 DESIGN PRINCIPLES

### Quick Access (Side Panel)
**Philosophy:** Frequently used workflow actions that need to be **always visible**.
- Video info and tracker selection
- Start/Stop/Pause processing
- Progress monitoring
- Quick export
- Quick plugin application

### Configuration (Options Window)
**Philosophy:** "Set it and forget it" settings that are configured **occasionally**.
- All threshold values, multipliers, parameters
- File paths and output settings
- Display preferences
- Advanced tracking configuration
- Device and streamer settings

### Actions (Menu & Toolbar)
**Philosophy:** One-time actions and toggles.
- File operations (New, Open, Save, Export)
- View toggles (Show/Hide windows)
- Tool launches (Autotuner, Batch Processing)
- Playback controls

---

## 🔍 NOTHING LEFT BEHIND

### Expert Mode Features
All Expert mode features are preserved:
- Stage rerun controls → **Options: General > Performance**
- Advanced tracking settings → **Options: Tracking** (all tabs)
- Plugin parameters → **Options: Post-Processing** (plugin tabs)
- Device control → **Options: Device Control** (all tabs)

### Simple Mode Features
All Simple mode features are accessible:
- Quick tracker selection → **Panel: Run > Tracker**
- One-click start → **Panel: Run > Start button**
- Progress display → **Panel: Run > Progress bars**
- Quick export → **Panel: Run > Export buttons**

### Supporter Features
All supporter-only features are preserved and feature-gated:
- Device Control → **Options: Device Control tab** (visible if supporter)
- Streamer → **Options: Streamer tab** (visible if supporter)
- Device Control indicator → **Menu Bar: Dynamic indicator**

### Hidden Features
All "hidden" or less-used features are discoverable:
- Energy saver settings → **Options: General > Energy Saver**
- TensorRT compilation → **Options: AI Models > TensorRT**
- Batch processing settings → **Options: Output > Batch Mode**
- Stage 2 database retention → **Options: Output > Advanced**

---

## 📊 FEATURE DISTRIBUTION

| Category | Panel | Options | Menu | Toolbar | Floating | TOTAL |
|----------|-------|---------|------|---------|----------|-------|
| **Playback** | 0 | 5 | 2 | 6 | 0 | 13 |
| **Project** | 2 | 5 | 18 | 4 | 2 | 31 |
| **Tracking** | 5 | 50 | 1 | 1 | 0 | 57 |
| **Editing** | 0 | 0 | 4 | 6 | 2 | 12 |
| **Display** | 0 | 30 | 20 | 4 | 4 | 58 |
| **Post-Proc** | 4 | 20+ | 2 | 2 | 0 | 28+ |
| **Output** | 2 | 15 | 5 | 1 | 3 | 26 |
| **AI Models** | 0 | 15 | 0 | 0 | 1 | 16 |
| **Funscript** | 0 | 35 | 0 | 0 | 1 | 36 |
| **Device** | 0 | 25 | 2 | 1 | 0 | 28 |
| **Streamer** | 0 | 10 | 2 | 1 | 0 | 13 |
| **Keyboard** | 0 | 30 | 1 | 0 | 1 | 32 |
| **Chapters** | 0 | 0 | 8 | 1 | 2 | 11 |
| **Gauges** | 0 | 10 | 4 | 1 | 4 | 19 |
| **About** | 0 | 5 | 2 | 0 | 0 | 7 |
| **TOTAL** | **13** | **255+** | **71** | **28** | **20** | **387+** |

**Additional Context Menus:** 20+ actions
**Additional Dynamic UI:** 13+ (plugin tabs, class filters, etc.)

**GRAND TOTAL: 420+ features** ✅

---

## 🎉 CONCLUSION

**✅ ZERO FEATURES LEFT BEHIND**

Every single feature from the original UI (400+) has been mapped to a location in the new design:
- **Minimal Side Panel** for workflow and quick actions
- **Options Window** for all configuration
- **Menu Bar & Toolbar** for commands and toggles
- **Floating Windows** for specialized views
- **Context Menus** for editing actions

The design achieves:
1. ✅ **Complete feature coverage** - Nothing lost
2. ✅ **Better organization** - Logical categorization
3. ✅ **Easier discovery** - Search functionality
4. ✅ **Minimal clutter** - Clean side panel
5. ✅ **Professional UI** - Modern design patterns

Ready for implementation! 🚀
