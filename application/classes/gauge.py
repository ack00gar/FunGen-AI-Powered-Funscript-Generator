import imgui
import numpy as np
from config.element_group_colors import GaugeColors

class GaugeWindow:
    def __init__(self, app_instance, timeline_num: int):
        self.app = app_instance
        self.timeline_num = timeline_num
        self.window_title = f"Gauge T{self.timeline_num}##GaugeWindowT{self.timeline_num}"

    def render(self):
        app_state = self.app.app_state_ui  # Cache for convenience

        # Determine which state variables to use based on timeline_num
        show_window_attr = f'show_gauge_window_timeline{self.timeline_num}'
        gauge_value_attr = f'gauge_value_t{self.timeline_num}'
        window_pos_attr = f'gauge_window_pos_t{self.timeline_num}'
        window_size_attr = f'gauge_window_size_t{self.timeline_num}'

        show_window = getattr(app_state, show_window_attr)
        if not show_window:
            return

        window_pos = getattr(app_state, window_pos_attr)
        window_size = getattr(app_state, window_size_attr)

        imgui.set_next_window_size(*window_size, condition=imgui.ONCE)
        imgui.set_next_window_position(*window_pos, condition=imgui.ONCE)

        window_flags = imgui.WINDOW_NO_SCROLLBAR

        opened, new_show_state = imgui.begin(
            self.window_title,
            closable=True,
            flags=window_flags
        )

        # Update the app state if the window was closed using the 'X'
        if show_window != new_show_state:
            setattr(app_state, show_window_attr, new_show_state)
            self.app.project_manager.project_dirty = True  # Closing/opening can be a preference to save

        if not opened:
            imgui.end()
            return

        # Update app_state with current window position and size if changed by user
        current_pos = imgui.get_window_position()
        current_size = imgui.get_window_size()
        current_pos_int = (int(current_pos[0]), int(current_pos[1]))
        current_size_int = (int(current_size[0]), int(current_size[1]))

        # Get stored pos/size from app_state for comparison
        stored_pos_int = (int(window_pos[0]), int(window_pos[1]))
        stored_size_int = (int(window_size[0]), int(window_size[1]))

        if current_pos_int != stored_pos_int or current_size_int != stored_size_int:
            setattr(app_state, window_pos_attr, current_pos_int)
            setattr(app_state, window_size_attr, current_size_int)

            self.app.project_manager.project_dirty = True  # Window move/resize makes project dirty

        # --- Gauge Drawing ---
        draw_list = imgui.get_window_draw_list()
        content_start_pos = imgui.get_cursor_screen_pos()
        content_avail_size = imgui.get_content_region_available()

        padding = 20  # Padding around the gauge drawing area
        gauge_area_x = content_start_pos[0] + padding
        gauge_area_y = content_start_pos[1] + padding
        gauge_area_width = content_avail_size[0] - 2 * padding

        # Calculate height for the value text display below the gauge
        value_text_height_approx = imgui.get_text_line_height_with_spacing() + (padding / 2)
        gauge_area_height = content_avail_size[1] - 2 * padding - value_text_height_approx

        if gauge_area_width < 20 or gauge_area_height < 50:  # Minimum viable drawing area
            imgui.text("Too small")
            imgui.end()
            return

        # Colors
        bg_color = imgui.get_color_u32_rgba(*GaugeColors.BACKGROUND)
        border_color = imgui.get_color_u32_rgba(*GaugeColors.BORDER)
        # Change bar color for the second timeline's gauge
        bar_color_fill = imgui.get_color_u32_rgba(*GaugeColors.BAR_RED) if self.timeline_num == 2 else imgui.get_color_u32_rgba(*GaugeColors.BAR_GREEN)
        text_color = imgui.get_color_u32_rgba(*GaugeColors.TEXT)

        # Draw gauge background and border
        draw_list.add_rect_filled(gauge_area_x, gauge_area_y,
                                  gauge_area_x + gauge_area_width, gauge_area_y + gauge_area_height,
                                  bg_color, rounding=3.0)
        draw_list.add_rect(gauge_area_x, gauge_area_y,
                           gauge_area_x + gauge_area_width, gauge_area_y + gauge_area_height,
                           border_color, rounding=3.0, thickness=1.0)

        # Draw gauge fill based on the correct gauge_value
        gauge_value = getattr(app_state, gauge_value_attr)
        normalized_value = np.clip(gauge_value / 100.0, 0.0, 1.0)

        bar_fill_height = normalized_value * gauge_area_height
        bar_fill_top_y = gauge_area_y + gauge_area_height - bar_fill_height

        draw_list.add_rect_filled(gauge_area_x, bar_fill_top_y,
                                  gauge_area_x + gauge_area_width, gauge_area_y + gauge_area_height,
                                  bar_color_fill, rounding=3.0)

        # Draw scale labels if space permits
        if gauge_area_height > 60:  # Enough vertical space for labels
            label_0_pos_y = gauge_area_y + gauge_area_height - imgui.get_text_line_height() - 2
            label_100_pos_y = gauge_area_y + 2
            draw_list.add_text(gauge_area_x + gauge_area_width + 3, label_0_pos_y, text_color, "0")
            draw_list.add_text(gauge_area_x + gauge_area_width + 3, label_100_pos_y, text_color, "100")
            if gauge_area_height > 80:  # Even more space for a middle label
                label_50_pos_y = gauge_area_y + gauge_area_height / 2 - imgui.get_text_line_height() / 2
                draw_list.add_text(gauge_area_x + gauge_area_width + 3, label_50_pos_y, text_color, "50")

        # Display the numerical value below the gauge
        value_text = f"{int(gauge_value)}"
        text_size_val = imgui.calc_text_size(value_text)

        # Position for the value text
        text_pos_x_val = content_start_pos[0] + (content_avail_size[0] - text_size_val[0]) / 2
        text_pos_y_val = gauge_area_y + gauge_area_height + (padding / 2 if gauge_area_height > 0 else padding)

        imgui.set_cursor_pos((text_pos_x_val - content_start_pos[0], text_pos_y_val - content_start_pos[1]))
        imgui.text_colored(value_text, *GaugeColors.VALUE_TEXT)  # Yellowish text for value

        imgui.end()
