import orjson
import os
import time
import datetime
import numpy as np
from typing import TYPE_CHECKING, Optional, Dict, Tuple

from config.constants import AUTOSAVE_FILE, PROJECT_FILE_EXTENSION, APP_VERSION
from application.utils.write_access import check_write_access

if TYPE_CHECKING:
    from application.logic.app_logic import ApplicationLogic

# Add a handler to convert NumPy types to standard Python types for JSON serialization
def numpy_default_handler(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


class ProjectManager:
    def __init__(self, app: 'ApplicationLogic'):
        self.app = app  # Reference to the main application logic instance
        self._project_file_path: Optional[str] = None
        self._project_dirty: bool = False
        self.last_autosave_time: float = time.time()

    @property
    def project_file_path(self) -> Optional[str]:
        return self._project_file_path

    @project_file_path.setter
    def project_file_path(self, value: Optional[str]):
        self._project_file_path = value

    @property
    def project_dirty(self) -> bool:
        return self._project_dirty

    @project_dirty.setter
    def project_dirty(self, value: bool):
        if self._project_dirty != value:
            self._project_dirty = value

    def new_project(self):
        if self.project_dirty:
            self.app.logger.warning("Unsaved changes in current project. Consider saving first.")

        # Delegate to ApplicationLogic's comprehensive reset method
        self.app.reset_project_state(for_new_project=True)

        # ProjectManager specific resets
        self.project_file_path = None
        self.project_dirty = False  # A new project starts clean
        self.last_autosave_time = time.time()

    def get_suggested_save_path_and_dir(self, save_as: bool) -> Optional[Tuple[str, str]]:
        if self.app.file_manager.video_path:
            # Suggest saving the project inside the video's dedicated output folder
            suggested_path = self.app.file_manager.get_output_path_for_file(
                self.app.file_manager.video_path,
                PROJECT_FILE_EXTENSION
            )
            return os.path.basename(suggested_path), os.path.dirname(suggested_path)
        return None

    def open_project_dialog(self):  # Called by MainMenu/AppLogic
        if hasattr(self.app, 'show_file_dialog_for_project_open'):
            self.app.show_file_dialog_for_project_open(self.load_project)
        elif hasattr(self.app, 'file_dialog_bridge'):  # Example of a bridge
            self.app.file_dialog_bridge.show_open_project_dialog(self.load_project)
        elif hasattr(self.app, 'gui_instance') and hasattr(self.app.gui_instance, 'file_dialog'):  # If GUI instance is on app

            suggested_path_info = self.get_suggested_save_path_and_dir(save_as=False)
            initial_dir = None
            if suggested_path_info:
                _, initial_dir = suggested_path_info

            self.app.gui_instance.file_dialog.show(
                title="Open Project",
                is_save=False,
                callback=self.load_project,
                extension_filter=f"FunGen Projects (*{PROJECT_FILE_EXTENSION}),*{PROJECT_FILE_EXTENSION}|Autosave States (*{AUTOSAVE_FILE.split('.')[-1]}),*{AUTOSAVE_FILE.split('.')[-1]}|All files (*.*),*.*", initial_path=initial_dir)
        else:
            self.app.logger.error("File dialog cannot be shown from ProjectManager. GUI bridge missing.")

    def _add_to_recent_projects(self, filepath: str):
        """Adds a project path to the top of the recent projects list."""
        if not filepath:
            return

        # Ensure we're working with an absolute path for consistency
        abs_filepath = os.path.abspath(filepath)

        recent_list = self.app.app_settings.get("recent_projects", [])

        # Remove any existing instance of this path to avoid duplicates and move it to the top
        if abs_filepath in recent_list:
            recent_list.remove(abs_filepath)

        # Add the new path to the front of the list
        recent_list.insert(0, abs_filepath)

        # Trim the list to a maximum of 10 recent files
        max_recent_files = 10
        trimmed_list = recent_list[:max_recent_files]

        # Persist the changes to settings
        self.app.app_settings.set("recent_projects", trimmed_list)

    def load_project(self, filepath: str, is_autosave: bool = False):  # Added is_autosave
        if not is_autosave and self.project_dirty:
            self.app.logger.warning("WARNING: Unsaved changes in current project. Loading new project will discard them.")

        try:
            with open(filepath, 'rb') as f:
                project_data = orjson.loads(f.read())

            # Reset application state before loading new project data
            self.app.reset_project_state(for_new_project=False)  # False indicates it's for loading

            self._apply_project_state_from_dict(project_data)

            self.project_file_path = filepath
            self.project_dirty = is_autosave

            # Handle video loading AFTER paths are set by _apply_project_state_from_dict
            if self.app.file_manager.video_path and os.path.exists(self.app.file_manager.video_path):
                self.app.file_manager.handle_video_file_load(self.app.file_manager.video_path, is_project_load=True)
            else:
                if self.app.file_manager.video_path:  # Path was set but file not found
                    self.app.logger.warning(
                        f"Video file specified in project not found: {self.app.file_manager.video_path}")
                self.app.file_manager.video_path = ""  # Ensure video_path is cleared if not valid

            # On successful load, update the recent projects list
            self._add_to_recent_projects(filepath)

            self.project_file_path = filepath

            # Final UI updates after everything is loaded
            self.app.app_state_ui.heatmap_dirty = True
            self.app.app_state_ui.funscript_preview_dirty = True
            self.app.app_state_ui.force_timeline_pan_to_current_frame = True

            if is_autosave:
                self.app.logger.info(f"State restored from autosave: {os.path.basename(filepath)}", extra={'status_message': True})
            else:
                self.app.logger.info(f"Project '{os.path.basename(filepath)}' loaded.", extra={'status_message': True})

        except Exception as e:
            self.app.logger.error(f"Error loading project '{os.path.basename(filepath)}': {e}", exc_info=True, extra={'status_message': True})
            # If loading fails, remove the bad path from the recent list
            recent_list = self.app.app_settings.get("recent_projects", [])
            abs_filepath = os.path.abspath(filepath)
            if abs_filepath in recent_list:
                recent_list.remove(abs_filepath)
                self.app.app_settings.set("recent_projects", recent_list)
            if is_autosave:
                self.app.logger.error(f"Autosave restoration from '{os.path.basename(filepath)}' failed critically.")

    def save_project_dialog(self, save_as: bool = False):
        if not self.project_file_path or save_as:
            suggested = self.get_suggested_save_path_and_dir(save_as) # No need to call the function twice
            suggested_filename, initial_dir_save = suggested if suggested else ("", None)

            if hasattr(self.app, 'show_file_dialog_for_project_save'):
                self.app.show_file_dialog_for_project_save(self.save_project, suggested_filename, initial_dir_save)
            elif hasattr(self.app, 'gui_instance') and hasattr(self.app.gui_instance, 'file_dialog'):
                self.app.gui_instance.file_dialog.show(
                    title="Save Project As" if save_as or not self.project_file_path else "Save Project",
                    is_save=True,
                    callback=self.save_project,
                    extension_filter=f"FunGen Projects (*{PROJECT_FILE_EXTENSION}),*{PROJECT_FILE_EXTENSION}|All files (*.*),*.*",
                    initial_filename=suggested_filename,
                    initial_path=initial_dir_save
                )
            else:
                self.app.logger.error("File dialog for save cannot be shown. GUI bridge missing.")
        else:
            self.save_project(self.project_file_path)

    def save_project(self, filepath: str):
        check_write_access(filepath)
        project_data = self._get_project_state_as_dict()
        project_data["version"] = APP_VERSION

        try:
            with open(filepath, 'wb') as f:
                f.write(orjson.dumps(project_data, default=numpy_default_handler))
            self.project_file_path = filepath
            self.project_dirty = False

            # On successful save, update the recent projects list
            self._add_to_recent_projects(filepath)

            self.app.logger.info(f"Project saved to '{os.path.basename(filepath)}'.", extra={'status_message': True})
            self.app.energy_saver.reset_activity_timer()
        except Exception as e:
            self.app.logger.error(f"Error saving project to '{filepath}': {e}", exc_info=True, extra={'status_message': True})

    def perform_autosave(self, is_exit_save: bool = False):
        """
        Performs an autosave of the current project file if autosave is enabled
        and the project state is dirty. The save is video-specific.
        """
        if not self.app.app_settings.get("autosave_enabled", True):
            if is_exit_save:
                self.app.logger.info("Autosave on exit skipped: Autosave is disabled in settings.")
            return

        # The primary condition for autosave is that the project has unsaved changes.
        if not self.project_dirty:
            self.last_autosave_time = time.time()  # Update timestamp to avoid rapid re-checking
            return

        # A video must be loaded to save a video-specific project file.
        video_path = self.app.file_manager.video_path
        if not video_path:
            self.app.logger.debug("Autosave skipped: No video loaded to associate the project with.")
            self.last_autosave_time = time.time()
            return

        # Determine the standard project file path for the current video.
        project_filepath = self.app.file_manager.get_output_path_for_file(video_path, PROJECT_FILE_EXTENSION)

        log_message_prefix = "Performing final project autosave on exit" if is_exit_save else "Autosaving project"
        self.app.logger.info(
            f"{log_message_prefix} for '{os.path.basename(video_path)}'...",
            extra={'status_message': not is_exit_save}
        )

        # Use the existing save_project method, which handles state gathering,
        # writing the file, and resetting the dirty flag.
        self.save_project(project_filepath)

        # The save_project method resets the dirty flag, which is what we want.
        # It also logs its own success message.
        self.last_autosave_time = time.time()


    def _get_project_state_as_dict(self) -> Dict:
        """Gathers all necessary data from app logic sub-modules for saving."""
        # Data from FunscriptProcessor
        fs_proc_data = self.app.funscript_processor.get_project_save_data()
        # Data from StageProcessor
        stage_proc_data = self.app.stage_processor.get_project_save_data()

        primary_actions, secondary_actions = [], []
        if self.app.processor and self.app.processor.tracker and self.app.processor.tracker.funscript:
            primary_actions = self.app.funscript_processor.get_actions('primary')  # Get copies
            secondary_actions = self.app.funscript_processor.get_actions('secondary')

        project_data = {
            # File Manager Data
            "video_path": self.app.file_manager.video_path,
            "funscript_path": self.app.file_manager.funscript_path,  # Project-associated funscript
            "loaded_funscript_path_timeline1": self.app.file_manager.loaded_funscript_path,  # T1 specific
            "stage1_output_msgpack_path": stage_proc_data.get("stage1_output_msgpack_path"),
            "stage2_overlay_msgpack_path": stage_proc_data.get("stage2_overlay_msgpack_path"),

            # Funscript Data
            "funscript_actions_timeline1": primary_actions,
            "funscript_actions_timeline2": secondary_actions,
            "video_chapters": fs_proc_data.get("video_chapters", []),
            "scripting_range_active": fs_proc_data.get("scripting_range_active", False),
            "scripting_start_frame": fs_proc_data.get("scripting_start_frame", 0),
            "scripting_end_frame": fs_proc_data.get("scripting_end_frame", -1),
            "selected_chapter_for_scripting_id": fs_proc_data.get("selected_chapter_for_scripting_id"),

            # AppStateUI Data
            "timeline_pan_offset_ms": self.app.app_state_ui.timeline_pan_offset_ms,
            "timeline_zoom_factor_ms_per_px": self.app.app_state_ui.timeline_zoom_factor_ms_per_px,
            "show_funscript_interactive_timeline": self.app.app_state_ui.show_funscript_interactive_timeline,
            "show_funscript_interactive_timeline2": self.app.app_state_ui.show_funscript_interactive_timeline2,
            "show_lr_dial_graph": self.app.app_state_ui.show_lr_dial_graph,
            "show_heatmap": self.app.app_state_ui.show_heatmap,
            "show_gauge_window_timeline1": getattr(self.app.app_state_ui, 'show_gauge_window_timeline1', True),
            "show_gauge_window_timeline2": getattr(self.app.app_state_ui, 'show_gauge_window_timeline2', False),

            "show_stage2_overlay": self.app.app_state_ui.show_stage2_overlay,
            "show_audio_waveform": self.app.app_state_ui.show_audio_waveform,
            # Note: Window positions/sizes are saved in app_settings, not project file.

            # ApplicationLogic direct settings (or could be AppSettings if purely user preference)
            "yolo_detection_model_path_setting": self.app.yolo_detection_model_path_setting,
            "yolo_pose_model_path_setting": self.app.yolo_pose_model_path_setting,
            # Consider if funscript_output_delay_frames (from calibration) should be project specific
            "funscript_output_delay_frames": self.app.calibration.funscript_output_delay_frames,

            # StageProcessor Data (mostly status, as progress is transient)
            "stage2_status_text": stage_proc_data.get("stage2_status_text", "Not run."),
        }
        if self.app.audio_waveform_data is not None:
            project_data["audio_waveform_data"] = self.app.audio_waveform_data
        return project_data

    def _apply_project_state_from_dict(self, project_data: Dict):
        """Applies loaded project data to the relevant app logic sub-modules."""
        # Data for AppLogic itself (or to be passed to AppSettings if they become project-specific)
        self.app.yolo_detection_model_path_setting = project_data.get("yolo_detection_model_path_setting", self.app.app_settings.get("yolo_det_model_path"))
        self.app.yolo_det_model_path = self.app.yolo_detection_model_path_setting
        self.app.yolo_pose_model_path_setting = project_data.get("yolo_pose_model_path_setting", self.app.app_settings.get("yolo_pose_model_path"))
        self.app.yolo_pose_model_path = self.app.yolo_pose_model_path_setting
        if self.app.tracker:  # Update tracker if it exists
            self.app.tracker.det_model_path = self.app.yolo_det_model_path
            self.app.tracker.pose_model_path = self.app.yolo_pose_model_path

        self.app.calibration.funscript_output_delay_frames = project_data.get("funscript_output_delay_frames", self.app.app_settings.get("funscript_output_delay_frames", 0))
        self.app.calibration.update_tracker_delay_params()  # Apply to tracker

        # Data for FileManager
        fm = self.app.file_manager
        fm.video_path = project_data.get("video_path", "")
        fm.funscript_path = project_data.get("funscript_path", "")
        fm.loaded_funscript_path = project_data.get("loaded_funscript_path_timeline1", fm.funscript_path)
        fm.stage1_output_msgpack_path = project_data.get("stage1_output_msgpack_path")
        fm.stage2_output_msgpack_path = project_data.get("stage2_overlay_msgpack_path")  # Path to overlay

        # Data for FunscriptProcessor
        fs_proc = self.app.funscript_processor
        t1_actions = project_data.get("funscript_actions_timeline1", [])
        fs_proc.clear_timeline_history_and_set_new_baseline(1, t1_actions, "Project Loaded (T1)")
        t2_actions = project_data.get("funscript_actions_timeline2", [])
        fs_proc.clear_timeline_history_and_set_new_baseline(2, t2_actions, "Project Loaded (T2)")

        fs_proc.update_project_specific_settings(project_data)  # Handles chapters, scripting range

        # Data for AppStateUI
        app_state = self.app.app_state_ui
        app_state.timeline_pan_offset_ms = project_data.get("timeline_pan_offset_ms",
    self.app.app_settings.get("timeline_pan_offset_ms", 0.0))
        app_state.timeline_zoom_factor_ms_per_px = project_data.get("timeline_zoom_factor_ms_per_px", self.app.app_settings.get("timeline_zoom_factor_ms_per_px", 20.0))
        app_state.show_funscript_interactive_timeline = project_data.get("show_funscript_interactive_timeline", self.app.app_settings.get("show_funscript_interactive_timeline",True))
        app_state.show_funscript_interactive_timeline2 = project_data.get("show_funscript_interactive_timeline2", self.app.app_settings.get("show_funscript_interactive_timeline2", False))
        app_state.show_lr_dial_graph = project_data.get("show_lr_dial_graph", self.app.app_settings.get("show_lr_dial_graph", True))
        app_state.show_heatmap = project_data.get("show_heatmap", self.app.app_settings.get("show_heatmap", True))
        app_state.show_gauge_window_timeline1 = project_data.get("show_gauge_window_timeline1", self.app.app_settings.get("show_gauge_window_timeline1", True))
        app_state.show_gauge_window_timeline2 = project_data.get("show_gauge_window_timeline2", self.app.app_settings.get("show_gauge_window_timeline2", False))
        app_state.show_stage2_overlay = project_data.get("show_stage2_overlay", self.app.app_settings.get("show_stage2_overlay", True))
        app_state.show_audio_waveform = project_data.get("show_audio_waveform", self.app.app_settings.get("show_audio_waveform", True))
        # Data for Audio Waveform
        loaded_waveform_list = project_data.get("audio_waveform_data")
        if loaded_waveform_list is not None and isinstance(loaded_waveform_list, list):
            # Waveform data is saved as a list, convert back to NumPy array on load
            self.app.audio_waveform_data = np.array(loaded_waveform_list, dtype=np.float32)
            self.app.logger.info(
                f"Loaded audio waveform data ({len(self.app.audio_waveform_data)} samples) from project.")
        else:
            self.app.audio_waveform_data = None

        # Data for StageProcessor
        stage_proc = self.app.stage_processor
        stage_proc.update_project_specific_settings(project_data)  # Handles stage status texts
        if fm.stage1_output_msgpack_path and os.path.exists(fm.stage1_output_msgpack_path):
            stage_proc.stage1_status_text = f"From Project: {os.path.basename(fm.stage1_output_msgpack_path)}"
            stage_proc.stage1_progress_value = 1.0
            stage_proc.stage1_progress_label = "Loaded from project"
        else:  # No valid S1 path in project or file missing
            stage_proc.reset_stage_status(stages=("stage1",))

        # Load S2 overlay data if path exists from project
        if fm.stage2_output_msgpack_path and os.path.exists(fm.stage2_output_msgpack_path):
            fm.load_stage2_overlay_data(fm.stage2_output_msgpack_path)


