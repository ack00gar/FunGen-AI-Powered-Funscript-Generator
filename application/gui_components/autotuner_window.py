import imgui
from config.constants import DEFAULT_S1_NUM_PRODUCERS, DEFAULT_S1_NUM_CONSUMERS

class AutotunerWindow:
    def __init__(self, app_logic):
        self.app = app_logic

    def render(self):
        app_state = self.app.app_state_ui
        if not getattr(app_state, 'show_autotuner_window', False):
            return

        is_open, new_visibility = imgui.begin("Stage 1 Performance Autotuner", closable=True, flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)
        try:
            if new_visibility != app_state.show_autotuner_window:
                app_state.show_autotuner_window = new_visibility
                if self.app.autotuner_thread and self.app.autotuner_thread.is_alive():
                    self.app.logger.warning("Autotuner window closed, but process is still running in the background.")

            # Only render the content if the window is visible/open
            if is_open:
                imgui.text_wrapped("This tool will run Stage 1 analysis multiple times to find the best-performing combination of Producer and Consumer threads for your system.")
                imgui.text_wrapped("A video must be loaded to run the test. The process may take a long time.")
                imgui.separator()

                is_ready = self.app.processor and self.app.processor.is_video_open()
                is_running = self.app.is_autotuning_active

                if not is_ready:
                    imgui.text_colored("Please load a video first.", 1.0, 0.5, 0.5, 1.0)

                # --- Start Button ---
                if is_running:
                    imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True)
                    imgui.push_style_var(imgui.STYLE_ALPHA, imgui.get_style().alpha * 0.5)

                if imgui.button("Start Autotune", width=-1):
                    if is_ready:
                        self.app.start_autotuner()

                if is_running:
                    imgui.pop_style_var()
                    imgui.internal.pop_item_flag()

                imgui.separator()

                # --- Status & Progress ---
                imgui.text("Status:")
                imgui.same_line()
                imgui.text_colored(self.app.autotuner_status_message, 0.2, 0.8, 1.0, 1.0)

                if is_running:
                    stage_proc = self.app.stage_processor
                    if stage_proc.full_analysis_active and stage_proc.current_analysis_stage == 1:
                        imgui.progress_bar(stage_proc.stage1_progress_value, size=(-1, 0),
                                           overlay=f"{stage_proc.stage1_progress_value * 100:.0f}%")

                imgui.separator()

                # --- Results Table ---
                imgui.text("Results (Frames Per Second):")
                table_flags = imgui.TABLE_BORDERS
                if imgui.begin_table("AutotuneResults", 5, flags=table_flags):
                    imgui.table_setup_column("HW Accel")
                    imgui.table_setup_column("Producers")
                    imgui.table_setup_column("Consumers")
                    imgui.table_setup_column("Peak FPS")
                    imgui.table_setup_column("Notes")
                    imgui.table_headers_row()

                    sorted_results = sorted(self.app.autotuner_results.items(), key=lambda item: (item[0][2], item[0][0], item[0][1]))

                    for (p, c, accel), (fps, note) in sorted_results:
                        imgui.table_next_row()
                        is_best = (p, c, accel) == self.app.autotuner_best_combination

                        # If this is the best row, push a new text color
                        if is_best:
                            # Push a bright green color for the text
                            imgui.push_style_color(imgui.COLOR_TEXT, 0.4, 1.0, 0.4, 1.0)

                        imgui.table_set_column_index(0)
                        imgui.text(accel)
                        imgui.table_set_column_index(1)
                        imgui.text(str(p))
                        imgui.table_set_column_index(2)
                        imgui.text(str(c))
                        imgui.table_set_column_index(3)
                        imgui.text(f"{fps:.2f}")
                        imgui.table_set_column_index(4)
                        imgui.text(note)

                        # If we pushed a color, we must pop it to restore the style
                        if is_best:
                            imgui.pop_style_color()

                    imgui.end_table()

                imgui.separator()

                # --- Recommendation and Apply Button ---
                if self.app.autotuner_best_combination:
                    p_best, c_best, accel_best = self.app.autotuner_best_combination
                    fps_best = self.app.autotuner_results.get((p_best, c_best, accel_best), (0.0, ""))[0]
                    imgui.text(f"Recommendation: {p_best}P/{c_best}C with HW Accel '{accel_best}' ({fps_best:.2f} FPS)")

                    if imgui.button("Apply Recommended Settings"):
                        self.app.stage_processor.num_producers_stage1 = p_best
                        self.app.stage_processor.num_consumers_stage1 = c_best
                        self.app.hardware_acceleration_method = accel_best
                        self.app.app_settings.set("num_producers_stage1", p_best)
                        self.app.app_settings.set("num_consumers_stage1", c_best)
                        self.app.app_settings.set("hardware_acceleration_method", accel_best)
                        self.app.logger.info(f"Autotuner settings applied: P={p_best}, C={c_best}, HW Accel={accel_best}", extra={'status_message': True})
        finally:
            imgui.end()
