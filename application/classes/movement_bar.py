import imgui
import numpy as np
from config.element_group_colors import LRDialColors


class MovementBarWindow:
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
            "Movement Bar##LRDialWindow",  # Rotating bar with up/down fill and roll angle
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

        # --- Insertion Bar with Roll Angle ---
        draw_list = imgui.get_window_draw_list()
        content_start_pos = imgui.get_cursor_screen_pos()
        content_avail_w, content_avail_h = imgui.get_content_region_available()

        padding = 15  # Padding around the bar
        canvas_origin_x = content_start_pos[0] + padding
        canvas_origin_y = content_start_pos[1] + padding
        drawable_width = content_avail_w - 2 * padding

        # No text display needed - use full height
        drawable_height = content_avail_h - 2 * padding

        if drawable_width < 80 or drawable_height < 120:  # Minimum for bar display
            imgui.text("Too small")
            imgui.end()
            return

        # Get current funscript values from app_state (same as gauge windows)
        up_down_position = getattr(app_state, 'gauge_value_t1', 50)  # Primary axis (0=down, 100=up)
        roll_angle = getattr(app_state, 'lr_dial_value', 50)         # Secondary axis (0=left, 50=center, 100=right)

        # Calculate center point for rotation
        center_x = canvas_origin_x + drawable_width / 2
        center_y = canvas_origin_y + drawable_height / 2
        
        # Convert roll_angle (0-100) to rotation angle (-30° to +30°) - INVERTED for device signal
        roll_degrees = -((roll_angle / 100.0) - 0.5) * 60.0  # -30° to +30° (inverted to match device)
        roll_radians = np.radians(roll_degrees)
        cos_r, sin_r = np.cos(roll_radians), np.sin(roll_radians)
        
        # Bar dimensions (will be rotated as a whole)
        bar_width = min(50, drawable_width * 0.25)  # Slightly narrower
        bar_height = drawable_height * 0.75        # Bar height
        
        # Create the rotated bar rectangle corners (centered at origin, then transformed)
        half_w, half_h = bar_width / 2, bar_height / 2
        bar_corners = [
            (-half_w, -half_h), (half_w, -half_h),  # Top edge  
            (half_w, half_h), (-half_w, half_h)     # Bottom edge
        ]
        
        # Transform corners: rotate then translate to center
        rotated_bar_corners = []
        for x, y in bar_corners:
            rx = x * cos_r - y * sin_r + center_x
            ry = x * sin_r + y * cos_r + center_y
            rotated_bar_corners.append((rx, ry))
        
        # Draw rotated bar background
        p1, p2, p3, p4 = rotated_bar_corners
        bar_bg_color = imgui.get_color_u32_rgba(0.2, 0.2, 0.2, 0.9)
        draw_list.add_triangle_filled(p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], bar_bg_color)
        draw_list.add_triangle_filled(p1[0], p1[1], p3[0], p3[1], p4[0], p4[1], bar_bg_color)
        
        # Draw rotated bar border
        bar_border_color = imgui.get_color_u32_rgba(0.5, 0.5, 0.5, 1.0)
        draw_list.add_line(p1[0], p1[1], p2[0], p2[1], bar_border_color, thickness=2)  # Top
        draw_list.add_line(p2[0], p2[1], p3[0], p3[1], bar_border_color, thickness=2)  # Right
        draw_list.add_line(p3[0], p3[1], p4[0], p4[1], bar_border_color, thickness=2)  # Bottom
        draw_list.add_line(p4[0], p4[1], p1[0], p1[1], bar_border_color, thickness=2)  # Left
        
        # Calculate fill level based on up_down_position (0=bottom, 100=top)
        fill_ratio = up_down_position / 100.0
        fill_height = bar_height * fill_ratio
        
        # Create fill rectangle corners (from bottom up to fill_height)
        fill_y_start = half_h - fill_height  # Start from bottom
        fill_corners = [
            (-half_w + 2, fill_y_start), (half_w - 2, fill_y_start),  # Top of fill
            (half_w - 2, half_h - 2), (-half_w + 2, half_h - 2)      # Bottom of fill
        ]
        
        # Only draw fill if there's something to fill
        if fill_height > 0:
            # Transform fill corners: rotate then translate
            rotated_fill_corners = []
            for x, y in fill_corners:
                rx = x * cos_r - y * sin_r + center_x
                ry = x * sin_r + y * cos_r + center_y
                rotated_fill_corners.append((rx, ry))
            
            # Color based on position: red (down) to green (up)  
            if fill_ratio < 0.5:
                # Bottom half: red to yellow
                t = fill_ratio * 2  # 0 to 1 
                r, g, b = 0.8, 0.2 + t * 0.6, 0.0
            else:
                # Top half: yellow to green
                t = (fill_ratio - 0.5) * 2  # 0 to 1
                r, g, b = 0.8 - t * 0.6, 0.8, t * 0.2
                
            fill_color = imgui.get_color_u32_rgba(r, g, b, 0.8)
            
            # Draw rotated fill
            f1, f2, f3, f4 = rotated_fill_corners
            draw_list.add_triangle_filled(f1[0], f1[1], f2[0], f2[1], f3[0], f3[1], fill_color)
            draw_list.add_triangle_filled(f1[0], f1[1], f3[0], f3[1], f4[0], f4[1], fill_color)
        
        # Draw scale markers on the left side (not rotated - fixed reference)
        for i in range(5):  # 0%, 25%, 50%, 75%, 100%
            marker_y = center_y + half_h - (i / 4.0) * bar_height  # Fixed vertical positions
            marker_x1 = center_x - half_w - 15
            marker_x2 = center_x - half_w - 10
            marker_color = imgui.get_color_u32_rgba(0.6, 0.6, 0.6, 0.7)
            draw_list.add_line(marker_x1, marker_y, marker_x2, marker_y, marker_color, thickness=1)
        
        # Clean interface - no text needed

        imgui.end()
