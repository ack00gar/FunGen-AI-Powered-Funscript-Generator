"""Typed protocols for the ApplicationLogic interface.

These protocols document the attributes/methods that GUI components actually
use from ApplicationLogic.  They can be adopted incrementally — a component
that only needs logging + settings can type-hint its ``app`` parameter as
``LoggingProvider & SettingsProvider`` (via intersection or a combined
protocol) instead of the full ApplicationLogic.

Usage example (in a GUI component):
    from application.logic.app_protocols import SettingsProvider

    class SomePanel:
        def __init__(self, app: SettingsProvider):
            self.app = app

This is purely for documentation and static analysis — no runtime enforcement.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

# ---------------------------------------------------------------------------
# Cluster 1: Logging & Status
# Used by: virtually every component (153+ refs)
# ---------------------------------------------------------------------------

@runtime_checkable
class LoggingProvider(Protocol):
    logger: logging.Logger

    def set_status_message(self, message: str) -> None: ...


# ---------------------------------------------------------------------------
# Cluster 2: Persistent Settings
# Used by: 35+ components (90+ refs to app_settings)
# ---------------------------------------------------------------------------

@runtime_checkable
class SettingsProvider(Protocol):
    @property
    def app_settings(self) -> Any:
        """Returns the AppSettings instance (get/set interface)."""
        ...


# ---------------------------------------------------------------------------
# Cluster 3: Transient UI State
# Used by: 35+ components (60+ refs to app_state_ui)
# ---------------------------------------------------------------------------

@runtime_checkable
class UIStateProvider(Protocol):
    @property
    def app_state_ui(self) -> Any:
        """Returns the AppStateUI instance."""
        ...


# ---------------------------------------------------------------------------
# Cluster 4: Video Processing Pipeline
# Used by: 25+ components (74 refs to processor alone)
# ---------------------------------------------------------------------------

@runtime_checkable
class VideoProcessingProvider(Protocol):
    @property
    def processor(self) -> Any:
        """Returns the VideoProcessor instance."""
        ...

    @property
    def funscript_processor(self) -> Any:
        """Returns the AppFunscriptProcessor instance."""
        ...

    @property
    def stage_processor(self) -> Any:
        """Returns the AppStageProcessor instance."""
        ...

    @property
    def file_manager(self) -> Any:
        """Returns the AppFileManager instance."""
        ...


# ---------------------------------------------------------------------------
# Cluster 5: Project Management
# Used by: 15+ components (42+ refs)
# ---------------------------------------------------------------------------

@runtime_checkable
class ProjectProvider(Protocol):
    @property
    def project_manager(self) -> Any: ...


# ---------------------------------------------------------------------------
# Cluster 6: Tracking & Detection
# Used by: 12+ components
# ---------------------------------------------------------------------------

@runtime_checkable
class TrackerProvider(Protocol):
    @property
    def tracker(self) -> Any: ...

    tracking_axis_mode: str
    single_axis_output_target: str


# ---------------------------------------------------------------------------
# Cluster 7: Energy / Activity Management
# Used by: 6 components (27 refs)
# ---------------------------------------------------------------------------

@runtime_checkable
class EnergyProvider(Protocol):
    @property
    def energy_saver(self) -> Any: ...


# ---------------------------------------------------------------------------
# Cluster 8: Batch Processing
# Used by: cp_execution_ui (primary)
# ---------------------------------------------------------------------------

@runtime_checkable
class BatchProvider(Protocol):
    is_batch_processing_active: bool
    stop_batch_event: threading.Event
    pause_batch_event: threading.Event
    current_batch_video_index: int
    batch_video_paths: List[Any]

    @property
    def batch_processor(self) -> Any: ...


# ---------------------------------------------------------------------------
# Composite protocols for common groupings
# ---------------------------------------------------------------------------

@runtime_checkable
class CoreAppProvider(LoggingProvider, SettingsProvider, UIStateProvider, EnergyProvider, Protocol):
    """The minimal interface most GUI components need."""
    ...


@runtime_checkable
class FullAppProvider(
    CoreAppProvider,
    VideoProcessingProvider,
    ProjectProvider,
    TrackerProvider,
    BatchProvider,
    Protocol,
):
    """The full interface — equivalent to ApplicationLogic today."""
    gui_instance: Any
