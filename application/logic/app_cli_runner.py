"""CLI runner functionality for ApplicationLogic."""
import time
import sys
import os
import math
import logging


# ---------------------------------------------------------------------------
# Module-level functions (standalone, no class dependency)
# ---------------------------------------------------------------------------

def cli_live_video_progress_callback(current_frame, total_frames, start_time):
    """A simpler progress callback for frame-by-frame video processing."""
    if total_frames <= 0 or current_frame < 0:
        return

    progress = float(current_frame + 1) / total_frames
    bar = _create_cli_progress_bar(progress)

    time_elapsed = time.time() - start_time
    fps = (current_frame + 1) / time_elapsed if time_elapsed > 0 else 0
    eta_seconds = ((total_frames - current_frame - 1) / fps) if fps > 0 else 0
    eta_str = f"{int(eta_seconds // 60):02d}:{int(eta_seconds % 60):02d}" if eta_seconds > 0 else "..."

    status_line = f"\rProcessing Video: {bar} | {int(fps):>3} FPS | ETA: {eta_str}  "
    sys.stdout.write(status_line)
    sys.stdout.flush()
    if current_frame + 1 == total_frames:
        sys.stdout.write("\n")

def _create_cli_progress_bar(percentage: float, width: int = 40) -> str:
    """Helper to create a text-based progress bar string."""
    filled_width = int(percentage * width)
    bar = '\u2588' * filled_width + '-' * (width - filled_width)
    return f"|{bar}| {percentage * 100:6.2f}%"


def cli_stage1_progress_callback(current, total, message, time_elapsed, avg_fps, instant_fps, eta_seconds, timing=None):
    if total <= 0: return
    progress = float(current) / total
    bar = _create_cli_progress_bar(progress)
    eta_str = f"{int(eta_seconds // 3600):02d}:{int((eta_seconds % 3600) // 60):02d}:{int(eta_seconds % 60):02d}" if eta_seconds > 0 else "..."

    timing_str = ""
    if timing:
        parts = [f"Dec:{timing.get('decode_ms', 0):.0f}ms"]
        if timing.get('unwarp_ms', 0) > 0:
            parts.append(f"Unw:{timing['unwarp_ms']:.0f}ms")
        parts.append(f"Det:{timing.get('yolo_det_ms', 0):.0f}ms")
        if timing.get('yolo_pose_ms', 0) > 0:
            parts.append(f"Pose:{timing['yolo_pose_ms']:.0f}ms")
        timing_str = f" | {' '.join(parts)}"

    status_line = f"\rStage 1: {bar} | {int(avg_fps):>3} FPS | ETA: {eta_str}{timing_str}   "
    sys.stdout.write(status_line)
    sys.stdout.flush()
    if current == total:
        sys.stdout.write("\n")


def cli_stage2_progress_callback(main_info, sub_info, force_update=False):
    main_current, total_main, main_name = main_info
    main_progress = float(main_current) / total_main if total_main > 0 else 0

    sub_progress = 0.0
    if isinstance(sub_info, dict):
        sub_current = sub_info.get("current", 0)
        sub_total = sub_info.get("total", 1)
        sub_progress = float(sub_current) / sub_total if sub_total > 0 else 0
    elif isinstance(sub_info, tuple) and len(sub_info) == 3:
        sub_current, sub_total, _ = sub_info
        sub_progress = float(sub_current) / sub_total if sub_total > 0 else 0

    main_bar = _create_cli_progress_bar(main_progress)
    status_line = f"\rStage 2: {main_name} ({main_current}/{total_main}) {main_bar} | Sub-task: {int(sub_progress * 100):>3}%  "
    sys.stdout.write(status_line)
    sys.stdout.flush()
    if main_current == total_main:
        sys.stdout.write("\n")

def cli_stage3_progress_callback(current_chapter_idx, total_chapters, chapter_name, current_chunk_idx, total_chunks, total_frames_processed_overall, total_frames_to_process_overall, processing_fps, time_elapsed, eta_seconds):
    if total_frames_to_process_overall <= 0: return
    overall_progress = float(total_frames_processed_overall) / total_frames_to_process_overall
    bar = _create_cli_progress_bar(overall_progress)

    eta_str = "..."
    if not (math.isnan(eta_seconds) or math.isinf(eta_seconds)):
        if eta_seconds > 1:
            eta_str = f"{int(eta_seconds // 3600):02d}:{int((eta_seconds % 3600) // 60):02d}:{int(eta_seconds % 60):02d}"

    status_line = f"\rStage 3: {bar} | Chapter {current_chapter_idx}/{total_chapters} ({chapter_name}) | {int(processing_fps):>3} FPS | ETA: {eta_str}   "
    sys.stdout.write(status_line)
    sys.stdout.flush()
    if total_frames_processed_overall >= total_frames_to_process_overall:
        sys.stdout.write("\n")


# ---------------------------------------------------------------------------
# Composition class -- delegates to ApplicationLogic via self.app
# ---------------------------------------------------------------------------

class AppCLIRunner:
    """Handles CLI mode operations."""

    def __init__(self, app_logic):
        self.app = app_logic

    # -- public -------------------------------------------------------------

    def run_cli(self, args):
        """
        Handles the application's command-line interface logic.
        """
        console_handler = None
        original_log_level = None
        for handler in self.app.logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                console_handler = handler
                original_log_level = handler.level
                break

        if console_handler:
            # Temporarily set the console to only show WARNINGs and above
            console_handler.setLevel(logging.WARNING)

        try:
            self.app.logger.info("Running in Command-Line Interface (CLI) mode.")

            # Check if we're in funscript processing mode
            if hasattr(args, 'funscript_mode') and args.funscript_mode:
                self._run_funscript_cli_mode(args)
                return

            # 1. Resolve input path and find video files
            input_path = os.path.abspath(args.input_path)
            if not os.path.exists(input_path):
                self.app.logger.error(f"Input path does not exist: {input_path}")
                return

            video_paths = []
            if os.path.isfile(input_path):
                video_paths.append(input_path)
            elif os.path.isdir(input_path):
                self.app.logger.info(f"Scanning folder for videos: {input_path} (Recursive: {args.recursive})")
                valid_extensions = {".mp4", ".mkv", ".mov", ".avi", ".webm"}
                if args.recursive:
                    for root, _, files in os.walk(input_path):
                        for file in files:
                            if os.path.splitext(file)[1].lower() in valid_extensions:
                                video_paths.append(os.path.join(root, file))
                else:
                    for file in os.listdir(input_path):
                        if os.path.splitext(file)[1].lower() in valid_extensions:
                            video_paths.append(os.path.join(input_path, file))

            if not video_paths:
                self.app.logger.error("No video files found at the specified path.")
                return

            self.app.logger.info(f"Found {len(video_paths)} video(s) to process.")

            self.app.logger.info("Redirecting progress callbacks to CLI output.")
            self.app.stage_processor.on_stage1_progress = cli_stage1_progress_callback
            self.app.stage_processor.on_stage2_progress = cli_stage2_progress_callback
            self.app.stage_processor.on_stage3_progress = cli_stage3_progress_callback

            # 2. Configure batch processing from CLI args using dynamic discovery
            from config.tracker_discovery import get_tracker_discovery
            discovery = get_tracker_discovery()

            # Resolve CLI mode to tracker info
            tracker_info = discovery.get_tracker_info(args.mode)
            if not tracker_info:
                self.app.logger.error(f"Unknown processing mode: {args.mode}")
                self.app.logger.error(f"Available modes: {discovery.get_supported_cli_modes()}")
                return

            if not tracker_info.supports_batch:
                self.app.logger.error(f"Mode '{args.mode}' does not support batch processing")
                self.app.logger.error(f"Batch-compatible modes: {[info.cli_aliases[0] for info in discovery.get_batch_compatible_trackers() if info.cli_aliases]}")
                return

            # Store the tracker name directly for batch processing
            self.app.batch_tracker_name = tracker_info.internal_name
            self.app.logger.info(f"Processing Mode: {args.mode} -> {tracker_info.display_name}")

            # Set oscillation detector mode for Stage 3 if provided
            if hasattr(args, 'od_mode') and args.od_mode:
                self.app.app_settings.set("stage3_oscillation_detector_mode", args.od_mode)
                self.app.logger.info(f"Stage 3 Oscillation Detector Mode: {args.od_mode}")

            # Overwrite mode: 2 for overwrite, 1 for skip if missing (default), 0 process all except own matching.
            self.app.batch_overwrite_mode = 2 if args.overwrite else 1
            self.app.batch_apply_ultimate_autotune = args.autotune
            self.app.batch_copy_funscript_to_video_location = args.copy

            # Post-processing and Ultimate Autotune are mutually exclusive to avoid double simplification
            # Priority: Ultimate Autotune > Auto Post-processing
            if args.autotune:
                # When Ultimate Autotune is enabled, disable post-processing to avoid double simplification
                self.app.batch_apply_post_processing = False
                self.app.logger.info("Ultimate Autotune enabled - auto post-processing disabled to prevent double simplification")
            else:
                # When Ultimate Autotune is disabled, allow post-processing based on settings
                self.app.batch_apply_post_processing = self.app.app_settings.get("enable_auto_post_processing", False)
                if self.app.batch_apply_post_processing:
                    self.app.logger.info("Ultimate Autotune disabled - auto post-processing enabled from settings")
                else:
                    self.app.logger.info("Both Ultimate Autotune and auto post-processing disabled")
            # Roll file generation: --generate-roll forces on, --no-generate-roll forces off, default (None) = auto
            if args.generate_roll is True:
                self.app.batch_generate_roll_file = True
            elif args.generate_roll is False:
                self.app.batch_generate_roll_file = False
            else:
                # Default behavior: enable for 3-stage modes or dual-axis trackers
                self.app.batch_generate_roll_file = (args.mode in ['3-stage', '3-stage-mixed']) or (tracker_info and tracker_info.supports_dual_axis)

            # Preprocessed video saving (off by default in batch/CLI to save disk)
            self.app.batch_save_preprocessed_video = getattr(args, 'save_preprocessed', False)

            self.app.logger.info(f"Settings -> Overwrite: {args.overwrite}, Autotune: {args.autotune}, Copy to video location: {args.copy}, Save preprocessed: {self.app.batch_save_preprocessed_video}")

            # 3. Set up and run the batch processing
            self.app.batch_video_paths = [
                {"path": path, "override_format": "Auto (Heuristic)"} for path in video_paths
            ]
            self.app.is_batch_processing_active = True
            self.app.current_batch_video_index = -1
            self.app.stop_batch_event.clear()

            # For CLI, we run the batch process in the main thread.
            self.app._run_batch_processing_thread()

            self.app.logger.info("CLI processing has finished.")

        finally:
            if console_handler and original_log_level is not None:
                # Restore the original logging level to the console
                console_handler.setLevel(original_log_level)

    # -- private ------------------------------------------------------------

    def _run_funscript_cli_mode(self, args):
        """
        Handles CLI funscript processing mode - applies filters to existing funscripts.
        """
        self.app.logger.info("Running in funscript processing mode")

        # 1. Find funscript files
        input_path = os.path.abspath(args.input_path)
        if not os.path.exists(input_path):
            self.app.logger.error(f"Input path does not exist: {input_path}")
            return

        funscript_paths = []
        if os.path.isfile(input_path):
            if input_path.lower().endswith('.funscript'):
                funscript_paths.append(input_path)
            else:
                self.app.logger.error(f"File is not a funscript: {input_path}")
                return
        elif os.path.isdir(input_path):
            self.app.logger.info(f"Scanning folder for funscripts: {input_path} (Recursive: {args.recursive})")
            if args.recursive:
                for root, _, files in os.walk(input_path):
                    for file in files:
                        if file.lower().endswith('.funscript'):
                            funscript_paths.append(os.path.join(root, file))
            else:
                for file in os.listdir(input_path):
                    if file.lower().endswith('.funscript'):
                        funscript_paths.append(os.path.join(input_path, file))

        if not funscript_paths:
            self.app.logger.error("No funscript files found at the specified path.")
            return

        self.app.logger.info(f"Found {len(funscript_paths)} funscript(s) to process with filter: {args.filter}")

        # 2. Load plugin system
        try:
            from funscript.plugins.base_plugin import plugin_registry
            # Import all plugins to ensure they're registered
            from funscript.plugins import (
                ultimate_autotune_plugin, rdp_simplify_plugin, savgol_filter_plugin,
                speed_limiter_plugin, anti_jerk_plugin, amplify_plugin, clamp_plugin,
                invert_plugin, keyframe_plugin
            )

            # Manually register plugins that don't auto-register
            from funscript.plugins.rdp_simplify_plugin import RdpSimplifyPlugin
            from funscript.plugins.amplify_plugin import AmplifyPlugin
            from funscript.plugins.clamp_plugin import ValueClampPlugin
            from funscript.plugins.invert_plugin import InvertPlugin
            from funscript.plugins.savgol_filter_plugin import SavgolFilterPlugin
            from funscript.plugins.speed_limiter_plugin import SpeedLimiterPlugin
            from funscript.plugins.anti_jerk_plugin import AntiJerkPlugin
            from funscript.plugins.keyframe_plugin import KeyframePlugin

            # Register plugins that aren't auto-registering
            plugins_to_register = [
                RdpSimplifyPlugin(), AmplifyPlugin(), ValueClampPlugin(), InvertPlugin(),
                SavgolFilterPlugin(), SpeedLimiterPlugin(), AntiJerkPlugin(), KeyframePlugin()
            ]

            for plugin in plugins_to_register:
                try:
                    plugin_registry.register(plugin)
                except Exception:
                    pass  # May already be registered

        except ImportError as e:
            self.app.logger.error(f"Failed to import plugin system: {e}")
            return

        # 3. Get the specified plugin
        plugin_map = {
            'ultimate-autotune': 'Ultimate Autotune',
            'rdp-simplify': 'Simplify (RDP)',
            'savgol-filter': 'Smooth (SG)',
            'speed-limiter': 'Speed Limiter',
            'anti-jerk': 'Anti-Jerk',
            'amplify': 'Amplify',
            'clamp': 'Clamp',
            'invert': 'Invert',
            'keyframe': 'Keyframes'
        }

        plugin_name = plugin_map.get(args.filter)
        if not plugin_name:
            self.app.logger.error(f"Unknown filter: {args.filter}")
            return

        plugin = plugin_registry.get_plugin(plugin_name)
        if not plugin:
            self.app.logger.error(f"Plugin not found: {plugin_name}")
            return

        self.app.logger.info(f"Using plugin: {plugin_name}")

        # 4. Process each funscript
        success_count = 0
        for i, funscript_path in enumerate(funscript_paths):
            try:
                self.app.logger.info(f"Processing {i+1}/{len(funscript_paths)}: {os.path.basename(funscript_path)}")

                # Load the funscript using existing parsing logic
                from funscript import MultiAxisFunscript
                actions, error_msg, _, _ = self.app.file_manager._parse_funscript_file(funscript_path)

                if error_msg:
                    self.app.logger.error(f"Failed to parse funscript {funscript_path}: {error_msg}")
                    continue

                if not actions:
                    self.app.logger.warning(f"Skipping empty funscript: {funscript_path}")
                    continue

                # Create funscript object and set actions
                funscript = MultiAxisFunscript()
                funscript.primary_actions = actions

                # Get default parameters for the plugin
                if hasattr(plugin, 'get_default_params'):
                    params = plugin.get_default_params()
                else:
                    params = {}

                # Apply the filter
                self.app.logger.info(f"Applying {plugin_name} filter...")
                result = plugin.transform(funscript, 'primary', **params)

                # Some plugins return the funscript object, others modify in-place and return None
                # We'll treat any non-exception result as success and use the original funscript
                # which should now be modified by the plugin
                output_path = self._generate_filtered_funscript_path(funscript_path, args.filter, args.overwrite)

                # Save the filtered funscript using existing file manager
                # Use the modified funscript (plugins modify in-place)
                self.app.file_manager._save_funscript_file(output_path, funscript.primary_actions)
                self.app.logger.info(f"Saved filtered funscript: {output_path}")
                success_count += 1

            except Exception as e:
                self.app.logger.error(f"Error processing {funscript_path}: {e}")
                continue

        self.app.logger.info(f"Funscript processing complete. Successfully processed {success_count}/{len(funscript_paths)} files.")

    def _generate_filtered_funscript_path(self, original_path, filter_name, overwrite):
        """Generate output path for filtered funscript."""
        if overwrite:
            return original_path

        # Insert filter name before .funscript extension
        base, ext = os.path.splitext(original_path)
        return f"{base}.{filter_name}{ext}"
