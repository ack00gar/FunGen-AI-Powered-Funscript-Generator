"""Post-Processing UI mixin for ControlPanelUI.

Plugin application is handled directly from the timeline's Plugins dropdown menu,
which provides live preview. This mixin retains only the range selection and
utility methods used by the Run tab.
"""
import imgui
from application.utils.imgui_helpers import DisabledScope as _DisabledScope, tooltip_if_hovered as _tooltip_if_hovered


class PostProcessingMixin:
    """Mixin providing range selection and post-processing utility methods."""

    # ------- Range selection -------

    def _render_range_selection(self, stage_proc, fs_proc, event_handlers):
        app = self.app
        disabled = stage_proc.full_analysis_active or (app.processor and app.processor.is_processing) or app.is_setting_user_roi_mode

        with _DisabledScope(disabled):
            ch, new_active = imgui.checkbox("Enable Range Processing", fs_proc.scripting_range_active)
            if ch:
                event_handlers.handle_scripting_range_active_toggle(new_active)
            _tooltip_if_hovered(
                "Restrict processing to a specific frame range or chapter.\n"
                "Enable the checkbox and set frames, or select a chapter."
            )

            if fs_proc.scripting_range_active:
                imgui.push_item_width(120)
                ch, nv = imgui.input_int(
                    "Start Frame##SR_InputStart",
                    fs_proc.scripting_start_frame,
                    flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE,
                )
                if ch:
                    event_handlers.handle_scripting_start_frame_input(nv)
                ch, nv = imgui.input_int(
                    "End Frame (-1 = end)##SR_InputEnd",
                    fs_proc.scripting_end_frame,
                    flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE,
                )
                if ch:
                    event_handlers.handle_scripting_end_frame_input(nv)
                imgui.pop_item_width()

                start_disp, end_disp = fs_proc.get_scripting_range_display_text()
                imgui.text("Active Range: Frames: %s to %s" % (start_disp, end_disp))
                sel_ch = fs_proc.selected_chapter_for_scripting
                if sel_ch:
                    imgui.text("Chapter: %s (%s)" % (sel_ch.class_name, sel_ch.segment_type))
                if imgui.button("Clear Range Selection##ClearRangeButton"):
                    event_handlers.clear_scripting_range_selection()
                _tooltip_if_hovered("Reset frame range and deselect chapter.")
        if disabled and imgui.is_item_hovered():
            imgui.set_tooltip("Disabled while another process is active.")
