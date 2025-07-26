import imgui
import numpy as np
from config.element_group_colors import LRDialColors


class LRDialWindow:
    def __init__(self, app_instance):
        self.app = app_instance
        self.lr_dial_pos_initialized = False  # Local flag for one-time position adjustment

    def render(self):
        app_state = self.app.app_state_ui  # Cache for convenience

        if not app_state.show_lr_dial_graph:
            return

        # One-time Y position adjustment after main menu bar height is known
        if not self.lr_dial_pos_initialized and hasattr(app_state, 'main_menu_bar_height') and app_state.main_menu_bar_height > 0:

            default_uninitialized_y_placeholder = self.app.app_settings.get("lr_dial_window_pos_y", 35)  # Get what default Y might have been

            if app_state.lr_dial_window_pos[1] == default_uninitialized_y_placeholder or \
                    app_state.lr_dial_window_pos[1] < app_state.main_menu_bar_height:
                # Recalculate default X based on current window width and gauge width
                gauge_w = app_state.gauge_window_size[0]
                dial_w = app_state.lr_dial_window_size[0]
                new_dial_x = app_state.window_width - gauge_w - dial_w - 30

                app_state.lr_dial_window_pos = (new_dial_x, app_state.main_menu_bar_height + 10)
            self.lr_dial_pos_initialized = True

        imgui.set_next_window_size(*app_state.lr_dial_window_size, condition=imgui.ONCE)
        imgui.set_next_window_position(*app_state.lr_dial_window_pos, condition=imgui.ONCE)

        window_flags = imgui.WINDOW_NO_SCROLLBAR

        opened_state, new_show_state = imgui.begin(
            "L/R Dial##LRDialWindow",  # Unique ID for the window
            closable=True,
            flags=window_flags
        )

        # Update the app state if the window was closed using the 'X'
        if app_state.show_lr_dial_graph != new_show_state:
            app_state.show_lr_dial_graph = new_show_state
            self.app.project_manager.project_dirty = True

        if not opened_state:  # If window is not visible
            imgui.end()
            return

        # Update app_state with current window position and size if changed by user
        current_pos = imgui.get_window_position()
        current_size = imgui.get_window_size()
        current_pos_int = (int(current_pos[0]), int(current_pos[1]))
        current_size_int = (int(current_size[0]), int(current_size[1]))

        stored_pos_int = (int(app_state.lr_dial_window_pos[0]), int(app_state.lr_dial_window_pos[1]))
        stored_size_int = (int(app_state.lr_dial_window_size[0]), int(app_state.lr_dial_window_size[1]))

        if current_pos_int != stored_pos_int or current_size_int != stored_size_int:
            app_state.lr_dial_window_pos = current_pos_int
            app_state.lr_dial_window_size = current_size_int
            self.app.project_manager.project_dirty = True  # Window move/resize makes project dirty

        # --- L/R Dial Drawing ---
        draw_list = imgui.get_window_draw_list()
        content_start_pos = imgui.get_cursor_screen_pos()
        content_avail_w, content_avail_h = imgui.get_content_region_available()

        padding = 10  # Padding around the dial drawing area
        canvas_origin_x = content_start_pos[0] + padding
        canvas_origin_y = content_start_pos[1] + padding
        drawable_width = content_avail_w - 2 * padding

        # Reserve space for the value text display below the dial
        text_height_reserve = imgui.get_text_line_height_with_spacing() + 5
        drawable_height = content_avail_h - 2 * padding - text_height_reserve

        if drawable_width < 40 or drawable_height < 40:  # Minimum viable drawing area
            imgui.text("Too small")
            imgui.end()
            return

        # Dial properties
        center_x_dial = canvas_origin_x + drawable_width / 2
        center_y_dial = canvas_origin_y + drawable_height / 2
        radius = min(drawable_width, drawable_height) * 0.40  # Adjust radius factor as needed

        # Draw dial background and border
        draw_list.add_circle_filled(center_x_dial, center_y_dial, radius,
                                    imgui.get_color_u32_rgba(*LRDialColors.BACKGROUND), num_segments=32)
        draw_list.add_circle(center_x_dial, center_y_dial, radius, imgui.get_color_u32_rgba(*LRDialColors.BORDER), num_segments=32, thickness=1)

        # Draw L and R labels
        l_text_size = imgui.calc_text_size("L")
        r_text_size = imgui.calc_text_size("R")
        draw_list.add_text(center_x_dial - radius - l_text_size[0] - 5, center_y_dial - l_text_size[1] / 2, imgui.get_color_u32_rgba(*LRDialColors.LEFT_LABEL), "L")  # Red for Left
        draw_list.add_text(center_x_dial + radius + 5, center_y_dial - r_text_size[1] / 2, imgui.get_color_u32_rgba(*LRDialColors.RIGHT_LABEL), "R")  # Blue for Right

        # Get dial value from app_state
        current_dial_value = int(app_state.lr_dial_value)  # Value 0-100

        # Calculate angle for the indicator
        # 0 = Left (-90 deg), 50 = Center (0 deg), 100 = Right (+90 deg)
        # Angle range is -90 to +90 degrees relative to upward vertical.
        angle_deg_relative_to_vertical = ((current_dial_value / 100.0) - 0.5) * 180.0
        # Convert to standard math angle (0 deg = right, positive CCW) then adjust for drawing
        angle_rad_for_drawing = np.radians(angle_deg_relative_to_vertical - 90)  # -90 makes 0 deg point up

        indicator_len = radius * 0.9
        indicator_tip_x = center_x_dial + indicator_len * np.cos(angle_rad_for_drawing)
        indicator_tip_y = center_y_dial + indicator_len * np.sin(angle_rad_for_drawing)

        # Draw indicator line and tip
        draw_list.add_line(center_x_dial, center_y_dial, indicator_tip_x, indicator_tip_y, imgui.get_color_u32_rgba(*LRDialColors.INDICATOR_LINE), thickness=3)
        draw_list.add_circle_filled(indicator_tip_x, indicator_tip_y, 4, imgui.get_color_u32_rgba(*LRDialColors.INDICATOR_TIP))  # Yellow tip

        # Display the numerical value below the dial
        l_r_value_text = f"{current_dial_value:02d}"  # e.g., "05", "50", "95"
        l_r_text_size_calc = imgui.calc_text_size(l_r_value_text)

        # Position for the value text
        text_pos_x = canvas_origin_x + (drawable_width - l_r_text_size_calc[0]) / 2
        text_pos_y = canvas_origin_y + drawable_height + 5  # Below the dial drawing area

        imgui.set_cursor_screen_pos((text_pos_x, text_pos_y))
        imgui.text_colored(l_r_value_text, *LRDialColors.VALUE_TEXT)  # Light text for value

        imgui.end()
