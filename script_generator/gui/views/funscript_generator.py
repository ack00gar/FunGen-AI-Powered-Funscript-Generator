import tkinter as tk
from tkinter import ttk, messagebox

from script_generator.debug.debug_data import get_metrics_file_info
from script_generator.debug.logger import log
from script_generator.funscript.debug.heatmap import generate_heatmap
from script_generator.funscript.debug.report import create_funscript_report
from script_generator.gui.controller.debug_video import debug_video
from script_generator.gui.controller.generate_funscript import generate_funscript
from script_generator.gui.controller.regenerate_funscript import regenerate_funscript
from script_generator.gui.controller.stop_processing import stop_processing
from script_generator.gui.messages.messages import UIMessage, ProgressMessage, UpdateGUIState
from script_generator.gui.utils.utils import enable_widgets, disable_widgets, set_progressbars_done, reset_progressbars
from script_generator.gui.utils.widgets import Widgets, LABEL_WIDTH
from script_generator.gui.views.popups.create_debug_video import render_debug_video_popup
from script_generator.gui.views.popups.edit_video_info import render_video_edit_popup
from script_generator.object_detection.util.data import get_raw_yolo_file_info
from script_generator.state.app_state import AppState


class FunscriptGeneratorPage(tk.Frame):
    def __init__(self, parent, controller):
        # region SETUP
        super().__init__(parent)
        self.controller = controller
        self.state: AppState = controller.state
        state: AppState = controller.state

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        wrapper = Widgets.frame(self, title=None, sticky="nsew")
        # endregion

        # region VIDEO SELECTION
        video_selection = Widgets.frame(wrapper, title="Video", main_section=True)


        def update_video_path():
            state.set_video_info()
            update_ui_for_state()
            fov = f", FOV {state.video_info.fov}" if state.video_info and state.video_info.is_vr else ""
            new_label = f"VR: {state.video_info.is_vr}, Fisheye: {state.video_info.is_fisheye}{fov}" if state.video_info else ""
            video_label.config(text=new_label)
            video_info_btn.show(state.video_info)
            ref_path.set(state.reference_script if state.reference_script else "")

        _, fs_entry, fs_button, _ = Widgets.file_selection(
            attr="video_path",
            parent=video_selection,
            label_text="Video",
            button_text="Browse",
            file_selector_title="Choose a File",
            file_types=[("Text Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")],
            state=state,
            tooltip_text="The video to generate a funscript for. For proper detection of fisheye keep the suffix like _FISHEYE190, _MKX200, etc. in the filename\n\nIn the future we'll add the option to override this in the UI.",
            command=lambda val: update_video_path(),
            pady=(0, 0)
        )

        video_info_container = Widgets.frame(video_selection, row=1, padx=(0, 0), pady=(0, 5), min_height=35)
        video_label = Widgets.label(video_info_container, "", None, align="left", padx=(LABEL_WIDTH + 17, 0), sticky="w", row=0)
        video_info_btn = Widgets.button(video_info_container, "Edit", command=lambda: Widgets.create_popup(
            title="Edit video settings", master=controller, width=350, height=120, content_builder=lambda window, user_action: render_video_edit_popup(window, state, update_video_path)),
                                        tooltip_text="Override the default detected values", visible=False, default_style=True, padx=(0, 10), column=90, row=0)
        # endregion

        # # region OPTIONAL SETTINGS
        # optional_settings = Widgets.frame(wrapper, title="Optional settings", main_section=True, row=2)
        #
        # _, start_f_widgets, _ = Widgets.frames_input(optional_settings, "Start", state=state, attr="frame_start", tooltip_text="Where to start processing the video. If you have a 60fps video starting at 10s would mean frame 600")
        # _, end_f_widgets, _ = Widgets.frames_input(optional_settings, "End", state=state, attr="frame_end", tooltip_text="Where to end processing the video. If you have a 60fps video stopping  at 10s would mean frame 600", row=1)
        # # endregion

        # region PROCESSING
        processing = Widgets.frame(wrapper, title="Processing", main_section=True, row=3)
        yolo_p_container, yolo_p, yolo_p_label, yolo_p_perc = Widgets.labeled_progress(processing, "Object Detection", row=0)
        track_p_container, track_p, track_p_label, track_p_perc = Widgets.labeled_progress(processing, "Tracking Analysis", row=2)

        def handle_process_btn():
            if not state.is_processing:
                if state.analyze_task:
                    state.analyze_task.is_stopped = False
                state.is_processing = True
                state.has_raw_yolo = False
                state.has_tracking_data = False
                reset_progressbars([(yolo_p, yolo_p_perc), (track_p, track_p_perc)])
                generate_funscript(state, controller)
                update_ui_for_state()
            else:
                # TODO add proper stop for analysis
                if state.has_raw_yolo:
                    messagebox.showwarning("WIP", f"Only the Object Detection process can be stopped for now")
                    return

                disable_widgets([processing_btn])
                stop_processing(state)
                state.is_processing = False
                state.video_info.has_raw_yolo, _, _ = get_raw_yolo_file_info(state)
                state.video_info.has_tracking_data, _, _ = get_metrics_file_info(state)
                state.analyze_task = None
                self.controller.after(500, update_ui_for_state)

        processing_btn = Widgets.button(processing, "Start processing", handle_process_btn, row=3)
        # endregion

        # region FUNSCRIPT TWEAKING
        tweaking = Widgets.frame(wrapper, title="Funscript", main_section=True, row=4)
        tweaking_container = ttk.Frame(tweaking)
        tweaking_container.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky="nsew")

        boost_frame = ttk.LabelFrame(tweaking_container, text="Boost Settings", padding=(10, 5))
        boost_frame.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

        _, boost_check, _ = Widgets.checkbox(boost_frame, "Boost Settings", state, "boost_enabled", False)
        b_d_1 = Widgets.range_selector(
            parent=boost_frame,
            label_text="Boost Up %",
            state=self.state,
            attr='boost_up_percent',
            values=[str(i) for i in range(0, 21)],
            row=1,
            tooltip_text="Boost up peaks by x%"
        )
        b_d_2 = Widgets.range_selector(
            parent=boost_frame,
            label_text="Reduce Down %",
            state=self.state,
            attr='boost_down_percent',
            values=[str(i) for i in range(0, 21)],
            row=2,
            tooltip_text="Reduce down peaks by x%"
        )

        # Threshold Settings
        threshold_frame = ttk.LabelFrame(tweaking_container, text="Threshold Settings", padding=(10, 5))
        threshold_frame.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        _, tres_check, _ = Widgets.checkbox(threshold_frame, "Enable Threshold", state, "threshold_enabled", False)
        t_d_1 = Widgets.range_selector(
            parent=threshold_frame,
            label_text="0 Threshold",
            state=self.state,
            attr='threshold_low',
            values=[str(i) for i in range(0, 16)],
            row=1,
            tooltip_text="Threshold under which values are mapped to 0"
        )
        t_d_2 = Widgets.range_selector(
            parent=threshold_frame,
            label_text="100 Threshold",
            state=self.state,
            attr='threshold_high',
            values=[str(i) for i in range(80, 101)],
            row=2,
            tooltip_text="Threshold above which values are mapped to 100"
        )

        # Simplification Settings
        vw_frame = ttk.LabelFrame(tweaking_container, text="Simplification", padding=(10, 5))
        vw_frame.grid(row=1, column=3, padx=5, pady=5, sticky="ew")

        _, simp_check, _ = Widgets.checkbox(vw_frame, "Enable Simplification", state, "vw_simplification_enabled", False)
        s_d_1 = Widgets.range_selector(
            parent=vw_frame,
            label_text="VW Factor",
            state=self.state,
            attr='vw_factor',
            values=[i / 5 for i in range(10, 51)],
            row=1,
            tooltip_text="Rationalize the number of points\nMakes the script end-user friendly\nDeactivating this will make the script unusable as is\nExpert mode, for post-editing"
        )
        s_d_2 = Widgets.range_selector(
            parent=vw_frame,
            label_text="Rounding",
            state=self.state,
            attr='rounding',
            values=[5, 10],
            row=2,
            tooltip_text="Rounding factor for values"
        )

        # Regenerate Funscript Button
        regenerate_btn = Widgets.button(tweaking_container, "Regenerate Funscript", lambda: regenerate_funscript(self.state), row=2)

        # endregion

        # region DEBUGGING
        debugging = Widgets.frame(wrapper, title="Debugging", main_section=True, row=5)
        general = Widgets.frame(debugging, title="General", row=0)
        _, _, live_preview_checked = Widgets.checkbox(
            general, "Live preview",
            state,
            "live_preview_mode",
            tooltip_text="Will show a live preview of what is being generated.\n\nFor debugging only this will reduce your performance.\nStage 1: Show bounding boxes during object detection.\nStage 2: Show funscript and metric overlay while the funscript is being processed.",
            row=0
        )
        _, metrics_check, _ = Widgets.checkbox(
            general,
            "Save debug metrics",
            state, "save_debug_file",
            tooltip_text="Saves a debug file to disk with all collected metrics.\nThis is a prerequisite for playing back the debug video with the debug statistics overlay.\nThis also contains all generated metrics.",
            row=1
        )
        _, max_fps_e, max_fps = Widgets.input(
            general,
            "Max fps",
            state=state,
            attr="max_preview_fps",
            row=1,
            column=28,
            label_width_px=45,
            width=55,
            tooltip_text="The maximum FPS for the debug video"
        )
        _, _, debug_mode_dropdown, _ = Widgets.dropdown(
            attr="debug_mode",
            parent=general,
            label_text="Mode",
            options=["funscript", "detection"],
            default_value=state.debug_mode,
            tooltip_text="Change the debug metrics\nfunscript: Script overlay and funscript metrics\ndetection: Shows object detection boxes, confidence score and tracking id",
            state=state,
            label_width_px=33,
            dropdown_width_px=80,
            row=1,
            column=29
        )
        play_btn = Widgets.button(
            general,
            "Play debug video",
            lambda: debug_video(state),
            row=1,
            column=30,
            tooltip_text="Opens a debug video player overlaid with debugging metrics (Press space to pause).\n\nThis overlay shows object detection boxes and a live funscript overlay,\namong other useful debugging information.\nCan only be triggered after the funscript generation process has completed.\nNeeds the 'Save debug metrics' option activated during processing."
        )
        script_compare = Widgets.frame(debugging, title="Script compare", row=1)
        _, ref_entry, ref_button, ref_path = Widgets.file_selection(
            attr="reference_script",
            parent=script_compare,
            label_text="Reference Script",
            button_text="Browse",
            file_selector_title="Choose a File",
            file_types=[("Funscript Files", "*.funscript"), ("All Files", "*.*")],
            tooltip_text="If provided the reference script will be compared in the\nfunscript report that is generated on completion and be shown\nin the live display funscript overlay when enabled.",
            state=state,
            row=0
        )
        debug_video_section = Widgets.frame(debugging, title="Share debug files", row=2)
        gen_video_btn = Widgets.button(
            debug_video_section,
            "Generate sharable debug video",
            lambda: Widgets.create_popup(title="Generate debug video", master=controller, width=650, height=205, content_builder=lambda window, user_action: render_debug_video_popup(window, state)),
            column=0,
            tooltip_text="Render a debug video once funscript processing is complete that can be\neasily shared on Discord for showcasing issues or areas of improvement.",
            sticky="ew"
        )
        gen_report_btn = Widgets.button(
            debug_video_section,
            "Generate report",
            lambda: create_funscript_report(state),
            column=1,
            tooltip_text="Generates a heatmap of your funscript",
            sticky="ew"
        )
        gen_heatmap_btn = Widgets.button(
            debug_video_section,
            "Generate heatmap",
            lambda: generate_heatmap(state),
            column=2,
            tooltip_text="Generates a heatmap of your funscript",
            sticky="ew"
        )
        # endregion

        # region FOOTER
        Widgets.disclaimer(wrapper)
        # endregion

        def update_ui_for_state():
            if not state.is_processing:
                processing_btn.config(text="Start processing")

            if state.has_raw_yolo and state.has_tracking_data:
                enable_widgets([play_btn, regenerate_btn, gen_video_btn, max_fps_e, debug_mode_dropdown, gen_report_btn, gen_heatmap_btn])
                set_progressbars_done([(yolo_p, yolo_p_perc), (track_p, track_p_perc)])
                processing_btn.config(text="Re-run object detection and or tracking")
            elif state.has_raw_yolo and not state.has_tracking_data:
                disable_widgets([play_btn, regenerate_btn, gen_video_btn, max_fps_e, debug_mode_dropdown, gen_report_btn, gen_heatmap_btn])
                set_progressbars_done([(yolo_p, yolo_p_perc)])
                reset_progressbars([(track_p, track_p_perc)])
                processing_btn.config(text="Re-run object detection and or start tracking")
            else:
                disable_widgets([play_btn, regenerate_btn, gen_video_btn, max_fps_e, debug_mode_dropdown, gen_report_btn, gen_heatmap_btn])
                reset_progressbars([(yolo_p, yolo_p_perc), (track_p, track_p_perc)])
                processing_btn.config(text="Start processing")

            if state.video_path:
                enable_widgets([processing_btn])
            else:
                disable_widgets([processing_btn])

            proc_widgets = [fs_entry, fs_button, ref_entry, ref_button, metrics_check, # , reader_dropdown, *start_f_widgets, *end_f_widgets
                            boost_check, simp_check, tres_check, t_d_1, t_d_2, s_d_1, s_d_2, b_d_1, b_d_2]
            if state.is_processing:
                disable_widgets([*proc_widgets])
                processing_btn.config(text="Stop processing")
            else:
                enable_widgets(proc_widgets)

            max_fps.set(str(state.max_preview_fps))

        # region UI UPDATE CALLBACK
        def update_ui(msg: UIMessage):
            """Handle UI updates using a switch-like statement."""

            def process_update():
                handlers = {
                    ProgressMessage: handle_progress_message,
                    UpdateGUIState: handle_state_update
                }

                handler = handlers.get(type(msg))
                if handler:
                    handler(msg)
                else:
                    log.info(f"Unhandled message type: {type(msg)}")

            def handle_state_update(state_msg: UpdateGUIState):
                setattr(state, state_msg.attr, state_msg.value)
                if state_msg.attr == "live_preview_mode":
                    live_preview_checked.set(state_msg.value)

                update_ui_for_state()

            def handle_progress_message(progress_msg: ProgressMessage):
                progress_mapping = {
                    "OBJECT_DETECTION": (yolo_p, yolo_p_perc, "has_raw_yolo"),
                    "TRACKING_ANALYSIS": (track_p, track_p_perc, "has_tracking_data"),
                }

                if progress_msg.process in progress_mapping:
                    progress_bar, progress_label, state_attr = progress_mapping[progress_msg.process]
                    progress_bar["value"] = progress_msg.frames_processed
                    progress_bar["maximum"] = progress_msg.total_frames
                    percentage = (progress_msg.frames_processed / progress_msg.total_frames) * 100 if progress_msg.total_frames > 0 else 0
                    is_done = progress_msg.frames_processed >= progress_msg.total_frames > 0
                    progress_label.config(text="Done" if is_done else f"{percentage:.0f}% - ETA: {progress_msg.eta}")

                    if is_done or progress_msg.frames_processed == 0:
                        update_ui_for_state()

            # Schedule the update on the main thread
            self.controller.after(0, process_update)

        self.state.update_ui = update_ui
        update_ui_for_state()
        # endregion
