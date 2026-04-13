import imgui
from typing import Optional

from application.utils import _format_time, VideoSegment, get_icon_texture_manager, primary_button_style, destructive_button_style
from config.constants import POSITION_INFO_MAPPING, DEFAULT_CHAPTER_FPS
from config.element_group_colors import VideoNavigationColors
from config.constants_colors import CurrentTheme



class ChapterPluginsMixin:
    """Mixin fragment for VideoNavigationUI."""

    def _render_chapter_plugin_menu(self, target_timeline: str):
        """Render plugin selection menu for chapter-based plugin application.
        
        Args:
            target_timeline: 'primary' or 'secondary' - which timeline to target
        """
        if not self.context_selected_chapters:
            imgui.text_disabled("No chapters selected")
            return
            
        # Get the appropriate timeline instance
        timeline_instance = None
        timeline_num = None
        
        if target_timeline == 'primary':
            timeline_instance = self.gui_instance.timeline_editor1
            timeline_num = 1
        elif target_timeline == 'secondary':
            timeline_instance = self.gui_instance.timeline_editor2
            timeline_num = 2
        else:
            imgui.text_disabled("Invalid timeline")
            return
            
        # Get funscript and axis using the same method as timeline
        target_funscript, axis_name = self.app.funscript_processor._get_target_funscript_object_and_axis(timeline_num)
            
        if not timeline_instance or not target_funscript:
            imgui.text_disabled("Timeline not available")
            return
            
        # Get available plugins from the timeline's plugin renderer
        if not hasattr(timeline_instance, 'plugin_renderer') or not timeline_instance.plugin_renderer:
            imgui.text_disabled("Plugin system not available")
            return
            
        plugin_manager = timeline_instance.plugin_renderer.plugin_manager
        available_plugins = plugin_manager.get_available_plugins()
        
        if not available_plugins:
            imgui.text_disabled("No plugins available")
            return
            
        chapter_count = len(self.context_selected_chapters)
        chapter_text = "chapter" if chapter_count == 1 else f"{chapter_count} chapters"
        
        # Render plugin menu items
        for plugin_name in sorted(available_plugins):
            ui_data = plugin_manager.get_plugin_ui_data(plugin_name)
            if not ui_data or not ui_data['available']:
                continue
                
            display_name = ui_data['display_name']
            if imgui.menu_item(f"{display_name}")[0]:
                # Apply plugin to chapter(s) using the same logic as timeline selection
                self._apply_plugin_to_chapters(
                    plugin_name, 
                    target_timeline, 
                    timeline_instance,
                    plugin_manager
                )
                imgui.close_current_popup()
                

    def _apply_plugin_to_chapters(self, plugin_name: str, target_timeline: str, 
                                 timeline_instance, plugin_manager):
        """Apply a plugin to all points within selected chapter(s).
        
        This method:
        1. Selects all points within the chapter time ranges on the target timeline
        2. Uses the exact same PluginUIRenderer code paths as timeline selection menu
        """
        try:
            # Step 1: Select all points in the chapters on the target timeline
            if hasattr(self.app.funscript_processor, 'select_points_in_chapters'):
                # Use the existing method to select points in chapters
                self.app.funscript_processor.select_points_in_chapters(
                    self.context_selected_chapters, 
                    target_timeline=target_timeline
                )
            else:
                self.app.logger.error("select_points_in_chapters method not available")
                return
                
            # Step 2: Get the selected indices from the timeline
            selected_indices = timeline_instance._resolve_selected_indices() if timeline_instance.multi_selected_action_indices else []
            
            if len(selected_indices) < 2:
                chapter_names = [ch.position_short_name for ch in self.context_selected_chapters]
                self.app.logger.warning(f"No points found in chapter(s) {chapter_names} on {target_timeline} timeline")
                return
                
            # Step 3: Apply plugin using EXACT same logic as timeline selection menu
            ui_data = plugin_manager.get_plugin_ui_data(plugin_name)
            
            if ui_data and timeline_instance.plugin_renderer._should_apply_directly(plugin_name, ui_data):
                # Direct application (like Invert, Ultimate Autotune)
                context = plugin_manager.plugin_contexts.get(plugin_name)
                if context:
                    context.apply_requested = True
                    # Force apply_to_selection for chapter context  
                    if hasattr(context, 'apply_to_selection'):
                        context.apply_to_selection = True
                    
                    chapter_names = [ch.position_short_name for ch in self.context_selected_chapters]
                    chapter_text = chapter_names[0] if len(chapter_names) == 1 else f"{len(chapter_names)} chapters"
                    self.app.logger.info(
                        f"Applied {ui_data['display_name']} to {len(selected_indices)} points in {chapter_text} on {target_timeline} timeline",
                        extra={"status_message": True}
                    )
            else:
                # Open configuration window (like other filters)
                from application.classes.plugin_ui_manager import PluginUIState
                plugin_manager.set_plugin_state(plugin_name, PluginUIState.OPEN)
                
                # Force apply_to_selection for chapter context
                context = plugin_manager.plugin_contexts.get(plugin_name)
                if context and hasattr(context, 'apply_to_selection'):
                    context.apply_to_selection = True
                    
                chapter_names = [ch.position_short_name for ch in self.context_selected_chapters]
                chapter_text = chapter_names[0] if len(chapter_names) == 1 else f"{len(chapter_names)} chapters"
                self.app.logger.info(
                    f"Opened {ui_data['display_name']} configuration for {len(selected_indices)} points in {chapter_text} on {target_timeline} timeline",
                    extra={"status_message": True}
                )
            
            # Clear chapter selection
            self.context_selected_chapters.clear()
            
        except Exception as e:
            self.app.logger.error(f"Error applying plugin {plugin_name} to chapters: {e}")
            import traceback
            traceback.print_exc()


    def _render_dynamic_chapter_analysis_menu(self):
        """Render dynamic chapter analysis menu with all available trackers."""
        if not self.context_selected_chapters:
            imgui.text_disabled("No chapter selected")
            return
            
        selected_chapter = self.context_selected_chapters[0]
        
        try:
            from config.tracker_discovery import get_tracker_discovery, TrackerCategory
            discovery = get_tracker_discovery()
            
            # Get all tracker categories
            live_trackers = discovery.get_trackers_by_category(TrackerCategory.LIVE)
            live_intervention_trackers = discovery.get_trackers_by_category(TrackerCategory.LIVE_INTERVENTION)
            offline_trackers = discovery.get_trackers_by_category(TrackerCategory.OFFLINE)
            
            # Group trackers by category for better organization
            tracker_groups = [
                ("Live Trackers", live_trackers),
                ("Live Intervention Trackers", live_intervention_trackers), 
                ("Offline Trackers", offline_trackers)
            ]
            
            tracker_found = False
            
            for group_name, trackers in tracker_groups:
                if not trackers:
                    continue
                    
                if tracker_found:  # Add separator between groups
                    imgui.separator()
                    
                # Render group header (non-clickable)
                imgui.text_colored(group_name, 0.7, 0.7, 0.7, 1.0)
                
                for tracker in trackers:
                    display_name = getattr(tracker, 'display_name', tracker.internal_name)
                    
                    # Only show the tracker name, not the description
                    menu_text = display_name
                    
                    if imgui.menu_item(menu_text)[0]:
                        self._apply_tracker_to_chapter(tracker, selected_chapter)
                        imgui.close_current_popup()
                
                tracker_found = True
            
            if not tracker_found:
                imgui.text_disabled("No trackers available")
                
        except Exception as e:
            self.app.logger.error(f"Error loading trackers for chapter analysis: {e}")
            imgui.text_colored("Error loading trackers", 1.0, 0.3, 0.3, 1.0)
            

    def _apply_tracker_to_chapter(self, tracker, chapter):
        """Apply a specific tracker to a chapter."""
        try:
            # Set the selected tracker
            self.app.app_state_ui.selected_tracker_name = tracker.internal_name
            
            # Set scripting range to the selected chapter
            if hasattr(self.app.funscript_processor, 'set_scripting_range_from_chapter'):
                self.app.funscript_processor.set_scripting_range_from_chapter(chapter)
                
                # Start tracking with descriptive message
                display_name = getattr(tracker, 'display_name', tracker.internal_name)
                self._start_live_tracking(
                    success_info=f"Started {display_name} analysis for chapter: {chapter.position_short_name}"
                )
                
                self.context_selected_chapters.clear()
            else:
                self.app.logger.error("set_scripting_range_from_chapter not found in funscript_processor.")
                
        except Exception as e:
            self.app.logger.error(f"Error applying tracker {tracker.internal_name} to chapter: {e}")


    def _render_resize_preview_tooltip(self, frame_num: int, edge: str):
        """Render preview tooltip when dragging chapter boundaries."""
        imgui.begin_tooltip()

        try:
            # Show edge being adjusted and frame number
            edge_name = "Start" if edge == 'left' else "End"
            imgui.text(f"Chapter {edge_name}: Frame {frame_num}")

            # Show timestamp if processor available
            if self.app.processor and self.app.processor.fps > 0:
                time_s = frame_num / self.app.processor.fps
                imgui.text(f"Time: {_format_time(self.app, time_s)}")

            imgui.separator()

            # Show video frame preview if available
            if self.resize_preview_data and self.resize_preview_data.get('frame') == frame_num:
                frame_data = self.resize_preview_data.get('frame_data')
                if frame_data is not None and frame_data.size > 0:
                    # Use GUI instance's enhanced preview texture
                    if hasattr(self.gui_instance, 'enhanced_preview_texture_id') and self.gui_instance.enhanced_preview_texture_id:
                        # Update texture with frame data
                        self.gui_instance.update_texture(self.gui_instance.enhanced_preview_texture_id, frame_data)

                        # Calculate display dimensions
                        frame_height, frame_width = frame_data.shape[:2]
                        max_width = 300
                        if frame_width > max_width:
                            scale = max_width / frame_width
                            display_width = max_width
                            display_height = int(frame_height * scale)
                        else:
                            display_width = frame_width
                            display_height = frame_height

                        # Display frame
                        imgui.image(self.gui_instance.enhanced_preview_texture_id, display_width, display_height)
                else:
                    imgui.text("Loading frame...")
            else:
                imgui.text("Loading frame...")

        except Exception as e:
            imgui.text(f"Preview error: {e}")

        imgui.end_tooltip()


