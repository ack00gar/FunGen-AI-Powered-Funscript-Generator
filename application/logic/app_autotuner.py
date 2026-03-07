"""Autotuner functionality for ApplicationLogic."""

import os
import threading
from typing import Optional, Dict


class AppAutotuner:
    """Handles autotuner operations."""

    def __init__(self, app_logic):
        self.app = app_logic

    def start_autotuner(self, force_hwaccel: Optional[str] = None):
        """Initiates the autotuning process in a background thread."""
        if self.app.is_autotuning_active:
            self.app.logger.warning("Autotuner is already running.")
            return
        if not self.app.processor or not self.app.processor.is_video_open():
            self.app.logger.error("Cannot start autotuner: No video loaded.", extra={'status_message': True})
            return

        self.app.autotuner_forced_hwaccel = force_hwaccel
        self.app.is_autotuning_active = True
        self.app.autotuner_thread = threading.Thread(target=self._run_autotuner_thread, daemon=True, name="AutotunerThread")
        self.app.autotuner_thread.start()

    def _run_autotuner_thread(self):
        """The actual logic for the autotuning process."""
        self.app.logger.info("Starting Stage 1 performance autotuner thread.")
        with self.app._autotuner_lock:
            self.app.autotuner_results = {}
            self.app.autotuner_best_combination = None
            self.app.autotuner_best_fps = 0.0

        def run_single_test(p: int, c: int, accel: str) -> Optional[float]:
            """Helper to run one analysis and return its FPS."""
            with self.app._autotuner_lock:
                self.app.autotuner_status_message = f"Running test: {p}P / {c}C (HW Accel: {accel})..."
            self.app.logger.info(self.app.autotuner_status_message)

            completion_event = threading.Event()
            # Set the flag as an attribute on the stage processor instance
            self.app.stage_processor.force_rerun_stage1 = True

            original_hw_method = self.app.hardware_acceleration_method
            try:
                self.app.hardware_acceleration_method = accel

                total_frames = self.app.processor.total_frames
                start_frame = min(1000, total_frames // 4)
                end_frame = min(start_frame + 1000, total_frames - 1)
                autotune_frame_range = (start_frame, end_frame)

                self.app.stage_processor.start_full_analysis(
                    processing_mode="OFFLINE_GUIDED_FLOW",
                    override_producers=p,
                    override_consumers=c,
                    completion_event=completion_event,
                    frame_range_override=autotune_frame_range,
                    is_autotune_run=True
                )
                completion_event.wait()

            finally:
                self.app.hardware_acceleration_method = original_hw_method

            if self.app.stage_processor.stage1_final_fps_str and "FPS" in self.app.stage_processor.stage1_final_fps_str:
                try:
                    fps_str = self.app.stage_processor.stage1_final_fps_str.replace(" FPS", "").strip()
                    fps = float(fps_str)
                    self.app.logger.info(f"Test finished for {p}P / {c}C ({accel}). Result: {fps:.2f} FPS")
                    return fps
                except (ValueError, TypeError):
                    self.app.logger.error(f"Could not parse FPS string: '{self.app.stage_processor.stage1_final_fps_str}'")
                    return None
            else:
                self.app.logger.error(f"Test failed for {p}P / {c}C ({accel}). No final FPS reported.")
                return None

        def get_perf(p, c, accel):
            with self.app._autotuner_lock:
                if (p, c, accel) in self.app.autotuner_results:
                    return self.app.autotuner_results[(p, c, accel)][0]

            fps = run_single_test(p, c, accel)
            with self.app._autotuner_lock:
                if fps is None:
                    self.app.autotuner_results[(p, c, accel)] = (0.0, "Failed")
                    return 0.0

                self.app.autotuner_results[(p, c, accel)] = (fps, "")

                if fps > self.app.autotuner_best_fps:
                    self.app.autotuner_best_fps = fps
                    self.app.autotuner_best_combination = (p, c, accel)
            return fps

        def find_best_consumer_for_producer(p, accel, max_cores):
            self.app.logger.info(f"Starting search for best consumer count for P={p}, Accel={accel}...")
            low = 2
            high = max(2, max_cores - p)

            while high - low >= 3:
                if self.app.stop_batch_event.is_set(): return
                m1 = low + (high - low) // 3
                m2 = high - (high - low) // 3

                perf_m1 = get_perf(p, m1, accel)
                if self.app.stop_batch_event.is_set(): return

                perf_m2 = get_perf(p, m2, accel)
                if self.app.stop_batch_event.is_set(): return

                if perf_m1 < perf_m2:
                    low = m1
                else:
                    high = m2

            self.app.logger.info(f"Narrowed search for P={p}, Accel={accel} to range [{low}, {high}]. Finalizing...")
            for c in range(low, high + 1):
                if self.app.stop_batch_event.is_set(): return
                get_perf(p, c, accel)

        try:
            accel_methods_to_test = []
            if self.app.autotuner_forced_hwaccel:
                self.app.logger.info(f"Autotuner forced to test only HW Accel: {self.app.autotuner_forced_hwaccel}")
                accel_methods_to_test.append(self.app.autotuner_forced_hwaccel)
            else:
                self.app.logger.info("Autotuner running in default mode (testing CPU and best GPU).")
                best_hw_accel = 'none'
                available_hw = self.app.available_ffmpeg_hwaccels
                if 'cuda' in available_hw or 'nvdec' in available_hw:
                    best_hw_accel = 'cuda'
                elif 'qsv' in available_hw:
                    best_hw_accel = 'qsv'
                elif 'videotoolbox' in available_hw:
                    best_hw_accel = 'videotoolbox'

                accel_methods_to_test.append('none')
                if best_hw_accel != 'none':
                    accel_methods_to_test.append(best_hw_accel)

            max_cores = os.cpu_count() or 4
            PRODUCER_RANGE = range(1, 3)

            for accel in accel_methods_to_test:
                for p in PRODUCER_RANGE:
                    if self.app.stop_batch_event.is_set():
                        raise InterruptedError("Autotuner aborted by user.")
                    find_best_consumer_for_producer(p, accel, max_cores)

            with self.app._autotuner_lock:
                if self.app.autotuner_best_combination:
                    p_final, c_final, accel_final = self.app.autotuner_best_combination
                    self.app.autotuner_status_message = f"Finished! Best: {p_final}P/{c_final}C, Accel: {accel_final} at {self.app.autotuner_best_fps:.2f} FPS"
                    self.app.logger.info(f"Autotuner finished. Best combination: {self.app.autotuner_best_combination} with {self.app.autotuner_best_fps:.2f} FPS.")
                else:
                    self.app.autotuner_status_message = "Finished, but no successful runs were completed."
                    self.app.logger.warning("Autotuner finished without any successful test runs.")

        except InterruptedError as e:
            with self.app._autotuner_lock:
                self.app.autotuner_status_message = "Aborted by user."
            self.app.logger.info(str(e))
        except Exception as e:
            with self.app._autotuner_lock:
                self.app.autotuner_status_message = f"An error occurred: {e}"
            self.app.logger.error(f"Autotuner thread failed: {e}", exc_info=True)
        finally:
            self.app.is_autotuning_active = False
            self.app.stage_processor.force_rerun_stage1 = False

    def get_autotuner_snapshot(self) -> Dict:
        """Thread-safe snapshot of autotuner state for GUI rendering."""
        with self.app._autotuner_lock:
            return {
                "status_message": self.app.autotuner_status_message,
                "results": dict(self.app.autotuner_results),
                "best_combination": self.app.autotuner_best_combination,
                "best_fps": self.app.autotuner_best_fps,
            }

    def trigger_ultimate_autotune_with_defaults(self, timeline_num: int):
        """
        Non-interactively runs the Ultimate Autotune pipeline with default settings.
        This is called automatically in 'Simple Mode' after an analysis completes.
        """
        self.app.logger.info(f"Triggering default Ultimate Autotune for Timeline {timeline_num}...")
        fs_proc = self.app.funscript_processor
        funscript_instance, axis_name = fs_proc._get_target_funscript_object_and_axis(timeline_num)

        if not funscript_instance or not axis_name:
            self.app.logger.error(f"Ultimate Autotune (auto): Could not find target funscript for T{timeline_num}.")
            return

        # Get default parameters from the funscript processor helper
        params = fs_proc.get_default_ultimate_autotune_params()
        op_desc = "Auto-Applied Ultimate Autotune (Simple Mode)"

        # 1. Record state for Undo
        fs_proc._record_timeline_action(timeline_num, op_desc)

        # 2. Apply Ultimate Autotune using the plugin system
        try:
            from funscript.plugins.base_plugin import plugin_registry
            # Import the plugin to ensure it's registered
            from funscript.plugins import ultimate_autotune_plugin
            ultimate_plugin = plugin_registry.get_plugin('Ultimate Autotune')

            if ultimate_plugin:
                result = ultimate_plugin.transform(funscript_instance, axis_name, **params)

                if result:
                    fs_proc._finalize_action_and_update_ui(timeline_num, op_desc)
                    self.app.logger.info("Default Ultimate Autotune applied successfully.",
                                         extra={'status_message': True, 'duration': 5.0})
                else:
                    self.app.logger.warning("Default Ultimate Autotune failed to produce a result.",
                                          extra={'status_message': True})
            else:
                self.app.logger.error("Ultimate Autotune plugin not available.",
                                    extra={'status_message': True})
        except Exception as e:
            self.app.logger.error(f"Error applying Ultimate Autotune: {e}",
                                extra={'status_message': True})
