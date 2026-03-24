"""AI Model Settings mixin for ControlPanelUI.

Provides YOLO detection/pose model path management and download UI.
Called from gui_dialog_renderer.py (AI Models dialog).
"""
import imgui
import os
import config
from application.utils import get_icon_texture_manager, destructive_button_style, primary_button_style
from application.utils.imgui_helpers import DisabledScope as _DisabledScope, tooltip_if_hovered as _tooltip_if_hovered


def _readonly_input(label_id, value, width=-1):
    if width >= 0:
        imgui.push_item_width(width)
    imgui.input_text(label_id, value or "Not set", 256, flags=imgui.INPUT_TEXT_READ_ONLY)
    if width >= 0:
        imgui.pop_item_width()


class AIModelsMixin:
    """Mixin providing AI model settings rendering methods."""

    # ------- Model path updates -------

    def _update_model_path(self, path, kind):
        """Update a model path (kind='detection' or 'pose') and reload."""
        app = self.app
        tracker = app.tracker
        if kind == "detection":
            tracker_attr, setting_key = "det_model_path", "yolo_det_model_path"
            app_setting_attr, app_path_attr = "yolo_detection_model_path_setting", "yolo_det_model_path"
        else:
            tracker_attr, setting_key = "pose_model_path", "yolo_pose_model_path"
            app_setting_attr, app_path_attr = "yolo_pose_model_path_setting", "yolo_pose_model_path"

        if not path or (tracker and path == getattr(tracker, tracker_attr, None)):
            return
        app.cached_class_names = None
        setattr(app, app_setting_attr, path)
        app.app_settings.set(setting_key, path)
        setattr(app, app_path_attr, path)
        app.project_manager.project_dirty = True
        app.logger.info("%s model path updated to: %s. Reloading models." % (kind.capitalize(), path))
        if tracker:
            setattr(tracker, tracker_attr, path)
            tracker._load_models()

    def _update_detection_model_path(self, path):
        self._update_model_path(path, "detection")

    def _update_pose_model_path(self, path):
        self._update_model_path(path, "pose")

    # ------- AI model settings -------

    def _render_ai_model_settings(self):
        app = self.app
        stage_proc = app.stage_processor
        settings = app.app_settings
        style = imgui.get_style()

        is_batch_mode = app.is_batch_processing_active
        is_analysis_running = stage_proc.full_analysis_active
        is_live_tracking_running = (app.processor and
                                    app.processor.is_processing and
                                    app.processor.enable_tracker_processing)
        is_setting_roi = app.is_setting_user_roi_mode
        is_any_process_active = is_batch_mode or is_analysis_running or is_live_tracking_running or is_setting_roi

        with _DisabledScope(is_any_process_active):
            # Precompute widths and shared icon
            tp = style.frame_padding.x * 2
            browse_w = imgui.calc_text_size("Browse").x + tp
            unload_w = imgui.calc_text_size("Unload").x + tp
            total_btn_w = browse_w + unload_w + style.item_spacing.x
            avail_w = imgui.get_content_region_available_width()
            input_w = avail_w - total_btn_w - style.item_spacing.x
            icon_mgr = get_icon_texture_manager()
            folder_tex, _, _ = icon_mgr.get_icon_texture('folder-open.png')
            btn_size = imgui.get_frame_height()

            def _browse_button(imgui_id, on_click):
                """Render a browse button (icon or text fallback). Call on_click if pressed."""
                imgui.push_id(imgui_id)
                clicked = False
                if folder_tex:
                    clicked = imgui.image_button(folder_tex, btn_size, btn_size)
                else:
                    clicked = imgui.button("Browse")
                if imgui.is_item_hovered():
                    imgui.set_tooltip("Browse...")
                imgui.pop_id()
                if clicked:
                    on_click()

            def _show_model_dialog(title, current_path, callback):
                gi = getattr(app, "gui_instance", None)
                if not gi:
                    return
                gi.file_dialog.show(
                    title=title, is_save=False, callback=callback,
                    extension_filter=self.AI_modelExtensionsFilter,
                    initial_path=os.path.dirname(current_path) if current_path else None,
                )

            # Detection model
            imgui.text("Detection Model")
            _readonly_input("##S1YOLOPath", app.yolo_detection_model_path_setting, input_w)
            imgui.same_line()
            _browse_button("S1YOLOBrowse", lambda: _show_model_dialog(
                "Select YOLO Detection Model", app.yolo_detection_model_path_setting, self._update_detection_model_path))
            imgui.same_line()
            with destructive_button_style():
                if imgui.button("Unload##S1YOLOUnload"):
                    app.unload_model("detection")
            _tooltip_if_hovered("Path to the YOLO object detection model file (%s)." % self.AI_modelTooltipExtensions)

            # Pose model
            imgui.text("Pose Model")
            _readonly_input("##PoseYOLOPath", app.yolo_pose_model_path_setting, input_w)
            imgui.same_line()
            _browse_button("PoseYOLOBrowse", lambda: _show_model_dialog(
                "Select YOLO Pose Model", app.yolo_pose_model_path_setting, self._update_pose_model_path))
            imgui.same_line()
            with destructive_button_style():
                if imgui.button("Unload##PoseYOLOUnload"):
                    app.unload_model("pose")
            _tooltip_if_hovered("Path to the YOLO pose estimation model file (%s). This model is optional." % self.AI_modelTooltipExtensions)

            # Download models button
            imgui.spacing()
            is_downloading = app.first_run_thread and app.first_run_thread.is_alive() if hasattr(app, 'first_run_thread') else False
            with _DisabledScope(is_downloading):
                with primary_button_style():
                    if imgui.button("Download / Update Models##DownloadModels", width=-1):
                        app.trigger_first_run_setup()
            if is_downloading:
                progress = getattr(app, 'first_run_progress', 0) / 100.0
                status = getattr(app, 'first_run_status_message', 'Downloading...')
                imgui.text(status)
                imgui.progress_bar(progress, size=(-1, 0), overlay=f"{progress * 100:.0f}%")
            else:
                _tooltip_if_hovered("Re-download default AI models from GitHub")

            mode = app.app_state_ui.selected_tracker_name
            if self._is_offline_tracker(mode):
                imgui.text("Stage 1 Inference Workers:")
                imgui.push_item_width(100)
                is_save_pre = getattr(stage_proc, "save_preprocessed_video", False)
                with _DisabledScope(is_save_pre):
                    ch_p, n_p = imgui.input_int(
                        "Producers##S1Producers", stage_proc.num_producers_stage1
                    )
                    if ch_p and not is_save_pre:
                        v = max(1, n_p)
                        if v != stage_proc.num_producers_stage1:
                            stage_proc.num_producers_stage1 = v
                            settings.set("num_producers_stage1", v)
                if is_save_pre:
                    _tooltip_if_hovered("Producers are forced to 1 when 'Save/Reuse Preprocessed Video' is enabled.")
                else:
                    _tooltip_if_hovered("Number of threads for video decoding & preprocessing.")

                imgui.same_line()
                ch_c, n_c = imgui.input_int("Consumers##S1Consumers", stage_proc.num_consumers_stage1)
                if ch_c:
                    v = max(1, n_c)
                    if v != stage_proc.num_consumers_stage1:
                        stage_proc.num_consumers_stage1 = v
                        settings.set("num_consumers_stage1", v)
                _tooltip_if_hovered("Number of threads for AI model inference. Match to available cores for best performance.")
                imgui.pop_item_width()

                imgui.text("Stage 2 OF Workers")
                imgui.same_line()
                imgui.push_item_width(120)
                cur_s2 = settings.get("num_workers_stage2_of", config.constants.DEFAULT_S2_OF_WORKERS)
                ch, new_s2 = imgui.input_int("##S2OFWorkers", cur_s2)
                if ch:
                    v = max(1, new_s2)
                    if v != cur_s2:
                        settings.set("num_workers_stage2_of", v)
                imgui.pop_item_width()
                _tooltip_if_hovered(
                    "Number of processes for Stage 2 Optical Flow gap recovery.\n"
                    "More may be faster on high-core CPUs."
                )
