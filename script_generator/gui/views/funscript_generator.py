import tkinter as tk
from tkinter import ttk

from script_generator.gui.controller.process_video import video_analysis
from script_generator.gui.controller.regenerate_funscript import regenerate_funscript
from script_generator.gui.controller.debug_video import debug_video
from script_generator.gui.messages.messages import UIMessage, ProgressMessage, UpdateGUIState
from script_generator.gui.utils.widgets import Widgets
from script_generator.gui.views.popups.create_debug_video import render_debug_video_popup
from script_generator.state.app_state import AppState
from script_generator.utils.helpers import is_mac
from script_generator.debug.logger import logger


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
        video_selection = Widgets.frame(wrapper, title="Video Selection", main_section=True)

        Widgets.file_selection(
            attr="video_path",
            parent=video_selection,
            label_text="Video",
            button_text="Browse",
            file_selector_title="Choose a File",
            file_types=[("Text Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")],
            state=state,
            tooltip_text="The video to generate a funscript for. For proper detection of fisheye keep the suffix like _FISHEYE190, _MKX200, etc. in the filename\n\nIn the future we'll add the option to override this in the UI.",
            command=lambda val: state.set_video_info()
        )

        Widgets.dropdown(
            attr="video_reader",
            parent=video_selection,
            label_text="Video Reader",
            options=["FFmpeg", *([] if is_mac() else ["FFmpeg + OpenGL (Windows)"])],
            default_value=state.video_reader,
            tooltip_text=("On Mac only FFmpeg is supported" if is_mac() else "FFmpeg + OpenGL is usually about 30% faster on a good GPU."),
            state=state,
            row=1
        )
        # endregion

        # region OPTIONAL SETTINGS
        optional_settings = Widgets.frame(wrapper, title="Optional settings", main_section=True, row=2)

        Widgets.frames_input(optional_settings, "Start", state=state, attr="frame_start", tooltip_text="Where to start processing the video. If you have a 60fps video starting at 10s would mean frame 600")
        Widgets.frames_input(optional_settings, "End", state=state, attr="frame_end", tooltip_text="Where to end processing the video. If you have a 60fps video stopping  at 10s would mean frame 600", row=1)
        # endregion

        # region PROCESSING
        processing = Widgets.frame(wrapper, title="Processing", main_section=True, row=3)
        yolo_p_container, yolo_p, yolo_p_label, yolo_p_perc = Widgets.labeled_progress(processing, "YOLO Detection", row=0)
        # scene_p_container, scene_p, scene_p_label, scene_p_perc = Widgets.labeled_progress(processing, "Scene detection", row=1)
        track_p_container, track_p, track_p_label, track_p_perc = Widgets.labeled_progress(processing, "Tracking Analysis", row=2)

        def start_processing():
            # TODO reset the progress bars
            video_analysis(state, controller)

        Widgets.button(processing, "Start processing", start_processing, row=3)
        # endregion

        # region FUNSCRIPT TWEAKING
        tweaking = Widgets.frame(wrapper, title="Funscript", main_section=True, row=4)
        # tweaking.grid_rowconfigure(1, weight=10)
        tweaking_container = ttk.Frame(tweaking)
        tweaking_container.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky="nsew")

        boost_frame = ttk.LabelFrame(tweaking_container, text="Boost Settings", padding=(10, 5))
        boost_frame.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

        Widgets.checkbox(boost_frame, "Boost Settings", state, "boost_enabled", False)
        Widgets.range_selector(
            parent=boost_frame,
            label_text="Boost Up %",
            state=self.state,
            attr='boost_up_percent',
            values=[str(i) for i in range(0, 21)],
            row=1,
            tooltip_text="Boost up peaks by x%"
        )
        Widgets.range_selector(
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

        Widgets.checkbox(threshold_frame, "Enable Threshold", state, "threshold_enabled", False)
        Widgets.range_selector(
            parent=threshold_frame,
            label_text="0 Threshold",
            state=self.state,
            attr='threshold_low',
            values=[str(i) for i in range(0, 16)],
            row=1,
            tooltip_text="Threshold under which values are mapped to 0"
        )
        Widgets.range_selector(
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

        Widgets.checkbox(vw_frame, "Enable Simplification", state, "vw_simplification_enabled", False)
        Widgets.range_selector(
            parent=vw_frame,
            label_text="VW Factor",
            state=self.state,
            attr='vw_factor',
            values=[i / 5 for i in range(10, 51)],
            row=1,
            tooltip_text="Rationalize the number of points\nMakes the script end-user friendly\nDeactivating this will make the script unusable as is\nExpert mode, for post-editing"
        )
        Widgets.range_selector(
            parent=vw_frame,
            label_text="Rounding",
            state=self.state,
            attr='rounding',
            values=[5, 10],
            row=2,
            tooltip_text="Rounding factor for values"
        )

        # Regenerate Funscript Button
        Widgets.button(tweaking_container, "Regenerate Funscript", lambda: regenerate_funscript(self.state), row=2)

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
        Widgets.checkbox(
            general,
            "Save debug metrics",
            state, "save_debug_file",
            tooltip_text="Saves a debug file to disk with all collected metrics.\nThis is a prerequisite for playing back the debug video with the debug statistics overlay.\nThis also contains all generated metrics.",
            row=1
        )
        Widgets.button(
            general,
            "Play debug video",
            lambda: debug_video(state),
            row=1,
            column=30,
            tooltip_text="Opens a debug video player overlaid with debugging information (Press space to pause).\n\nThis overlay shows object detection boxes and a live funscript overlay,\namong other useful debugging information.\nCan only be triggered after the funscript generation process has completed.\nNeeds the 'Save debug information' option activated during processing."
        )
        script_compare = Widgets.frame(debugging, title="Script compare", row=1)
        Widgets.file_selection(
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
        debug_video_section = Widgets.frame(debugging, title="Share debug video", row=1)
        Widgets.button(
            debug_video_section,
            "Generate sharable debug video",
            lambda: Widgets.create_popup(title="Generate debug video", master=controller, width=650, height=205, content_builder=lambda window, user_action: render_debug_video_popup(window, state)),
            row=1,
            column=0,
            tooltip_text="Render a debug video once funscript processing is complete that can be\neasily shared on Discord for showcasing issues or areas of improvement."
        )
        # endregion

        # region FOOTER
        Widgets.disclaimer(wrapper)
        # endregion

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
                    logger.info(f"Unhandled message type: {type(msg)}")

            def handle_state_update(state_msg: UpdateGUIState):
                if state_msg.attr == "live_preview_mode":
                    live_preview_checked.set(state_msg.value)

            def handle_progress_message(progress_msg: ProgressMessage):
                progress_mapping = {
                    "OBJECT_DETECTION": (yolo_p, yolo_p_perc),
                    # "SCENE_DETECTION": (scene_p, scene_p_perc),
                    "TRACKING_ANALYSIS": (track_p, track_p_perc),
                }

                if progress_msg.process in progress_mapping:
                    progress_bar, progress_label = progress_mapping[progress_msg.process]
                    progress_bar["value"] = progress_msg.frames_processed
                    progress_bar["maximum"] = progress_msg.total_frames
                    percentage = (progress_msg.frames_processed / progress_msg.total_frames) * 100 if progress_msg.total_frames > 0 else 0
                    progress_label.config(text=f"{percentage:.0f}% - ETA: {progress_msg.eta}" if progress_msg.frames_processed < progress_msg.total_frames else "Done")

            # Schedule the update on the main thread
            self.controller.after(0, process_update)

        self.state.update_ui = update_ui
        # endregion
