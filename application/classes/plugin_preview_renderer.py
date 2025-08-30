"""
Plugin Preview Renderer - Modular overlay system for visualizing plugin effects.

This module provides a plugin-agnostic preview system that renders overlays on the
timeline to show what changes a plugin will make before applying them.
"""

import imgui
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
import logging
from dataclasses import dataclass
import time


@dataclass
class PreviewPoint:
    """A single preview point to render."""
    at: int  # Timestamp in milliseconds
    pos: int  # Position 0-100
    is_new: bool = False  # True if this is a new point being added
    is_modified: bool = False  # True if this is an existing point being modified
    original_pos: Optional[int] = None  # Original position if modified
    is_selected: bool = True  # True if this point is part of the selection (default True for backward compatibility)


@dataclass
class PreviewSegment:
    """A continuous segment of preview points."""
    points: List[PreviewPoint]
    style: str = 'default'  # 'default', 'filled_gap', 'smoothed', 'amplified', etc.
    color: Optional[Tuple[float, float, float, float]] = None  # RGBA override


class PluginPreviewRenderer:
    """
    Renders preview overlays for plugin transformations on the timeline.
    
    This class is designed to be plugin-agnostic and work with any plugin that
    can provide preview data in the expected format.
    """
    
    # Default preview colors (RGBA) - All use consistent orange
    PREVIEW_COLORS = {
        'default': (1.0, 0.65, 0.0, 0.8),  # Orange
        'filled_gap': (1.0, 0.65, 0.0, 0.8),  # Orange (same as default)
        'smoothed': (1.0, 0.65, 0.0, 0.8),  # Orange (same as default) 
        'amplified': (1.0, 0.65, 0.0, 0.8),  # Orange (same as default)
        'reduced': (1.0, 0.65, 0.0, 0.8),  # Orange (same as default)
        'new_point': (1.0, 0.65, 0.0, 0.9),  # Orange (slightly more opaque)
        'modified_point': (1.0, 0.65, 0.0, 0.9),  # Orange (slightly more opaque)
        'unselected': (0.5, 0.5, 0.5, 0.4),  # Gray (dimmed for unselected points)
        'unselected_point': (0.5, 0.5, 0.5, 0.5),  # Gray (dimmed for unselected individual points)
    }
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the preview renderer."""
        self.logger = logger or logging.getLogger('PluginPreviewRenderer')
        self.active_previews: Dict[str, List[PreviewSegment]] = {}
        self.preview_cache: Dict[str, Any] = {}
        self.last_preview_update: Dict[str, float] = {}
        self.preview_update_throttle = 0.1  # Minimum seconds between updates
        
    def set_preview_data(self, plugin_name: str, preview_data: Any):
        """
        Set preview data for a plugin.
        
        Args:
            plugin_name: Name of the plugin
            preview_data: Preview data from the plugin
        """
        # Throttle updates to prevent excessive recomputation
        current_time = time.time()
        last_update = self.last_preview_update.get(plugin_name, 0)
        
        if current_time - last_update < self.preview_update_throttle:
            return
            
        self.last_preview_update[plugin_name] = current_time
        
        # Convert plugin preview data to our internal format
        segments = self._convert_preview_data(plugin_name, preview_data)
        if segments:
            self.active_previews[plugin_name] = segments
            self.preview_cache[plugin_name] = preview_data
        
    def clear_preview(self, plugin_name: str):
        """Clear preview data for a plugin."""
        if plugin_name in self.active_previews:
            del self.active_previews[plugin_name]
        if plugin_name in self.preview_cache:
            del self.preview_cache[plugin_name]
        if plugin_name in self.last_preview_update:
            del self.last_preview_update[plugin_name]
    
    def clear_all_previews(self):
        """Clear all active previews."""
        self.active_previews.clear()
        self.preview_cache.clear()
        self.last_preview_update.clear()
    
    def has_preview(self, plugin_name: str) -> bool:
        """Check if a plugin has active preview data."""
        return plugin_name in self.active_previews
    
    def render_preview_overlay(self, 
                              draw_list,
                              timeline_x: float,
                              timeline_y: float,
                              timeline_width: float,
                              timeline_height: float,
                              visible_start_ms: int,
                              visible_end_ms: int,
                              plugin_name: Optional[str] = None):
        """
        Render preview overlay on the timeline.
        
        Args:
            draw_list: ImGui draw list
            timeline_x: X coordinate of timeline
            timeline_y: Y coordinate of timeline
            timeline_width: Width of timeline
            timeline_height: Height of timeline
            visible_start_ms: Start of visible time range in milliseconds
            visible_end_ms: End of visible time range in milliseconds
            plugin_name: Optional specific plugin to render (None = render all)
        """
        if plugin_name:
            # Render specific plugin preview
            if plugin_name in self.active_previews:
                self._render_plugin_preview(
                    draw_list, plugin_name, self.active_previews[plugin_name],
                    timeline_x, timeline_y, timeline_width, timeline_height,
                    visible_start_ms, visible_end_ms
                )
        else:
            # Render all active previews
            for name, segments in self.active_previews.items():
                self._render_plugin_preview(
                    draw_list, name, segments,
                    timeline_x, timeline_y, timeline_width, timeline_height,
                    visible_start_ms, visible_end_ms
                )
    
    def _convert_preview_data(self, plugin_name: str, preview_data: Any) -> List[PreviewSegment]:
        """
        Convert plugin-specific preview data to our internal format.
        
        This method handles various preview data formats from different plugins.
        """
        segments = []
        
        try:
            # Handle dictionary format with preview points
            if isinstance(preview_data, dict):
                if 'preview_points' in preview_data:
                    # Direct preview points format
                    points = []
                    for point_data in preview_data['preview_points']:
                        point = PreviewPoint(
                            at=point_data['at'],
                            pos=point_data['pos'],
                            is_new=point_data.get('is_new', False),
                            is_modified=point_data.get('is_modified', False),
                            original_pos=point_data.get('original_pos'),
                            is_selected=point_data.get('is_selected', True)
                        )
                        points.append(point)
                    
                    if points:
                        style = preview_data.get('style', 'default')
                        color = preview_data.get('color')
                        segments.append(PreviewSegment(points=points, style=style, color=color))
                
                elif 'gaps_to_fill' in preview_data:
                    # Gap filling format - gaps with interpolated points
                    for gap in preview_data['gaps_to_fill']:
                        points = []
                        for point_data in gap.get('interpolated_points', []):
                            point = PreviewPoint(
                                at=point_data['at'],
                                pos=point_data['pos'],
                                is_new=True
                            )
                            points.append(point)
                        
                        if points:
                            segments.append(PreviewSegment(
                                points=points,
                                style='default'  # Use consistent style
                            ))
                
                elif 'modified_actions' in preview_data:
                    # Generic modification format
                    points = []
                    for action in preview_data['modified_actions']:
                        point = PreviewPoint(
                            at=action['at'],
                            pos=action['new_pos'],
                            is_modified=True,
                            original_pos=action.get('original_pos')
                        )
                        points.append(point)
                    
                    if points:
                        style = preview_data.get('filter_type', 'default').lower()
                        segments.append(PreviewSegment(points=points, style=style))
            
            # Handle list format (direct list of preview segments)
            elif isinstance(preview_data, list):
                for segment_data in preview_data:
                    if isinstance(segment_data, PreviewSegment):
                        segments.append(segment_data)
                    elif isinstance(segment_data, dict) and 'points' in segment_data:
                        points = []
                        for p in segment_data['points']:
                            point = PreviewPoint(
                                at=p['at'],
                                pos=p['pos'],
                                is_new=p.get('is_new', False),
                                is_modified=p.get('is_modified', False),
                                original_pos=p.get('original_pos'),
                                is_selected=p.get('is_selected', True)
                            )
                            points.append(point)
                        
                        if points:
                            segments.append(PreviewSegment(
                                points=points,
                                style=segment_data.get('style', 'default'),
                                color=segment_data.get('color')
                            ))
            
        except Exception as e:
            self.logger.warning(f"Failed to convert preview data for {plugin_name}: {e}")
        
        return segments
    
    def _render_plugin_preview(self,
                              draw_list,
                              plugin_name: str,
                              segments: List[PreviewSegment],
                              timeline_x: float,
                              timeline_y: float,
                              timeline_width: float,
                              timeline_height: float,
                              visible_start_ms: int,
                              visible_end_ms: int):
        """Render preview segments for a specific plugin."""
        
        visible_duration = visible_end_ms - visible_start_ms
        if visible_duration <= 0:
            return
        
        # Render each segment
        for segment in segments:
            if not segment.points:
                continue
            
            # Determine color for this segment
            if segment.color:
                color = imgui.get_color_u32_rgba(*segment.color)
            else:
                color_tuple = self.PREVIEW_COLORS.get(segment.style, self.PREVIEW_COLORS['default'])
                color = imgui.get_color_u32_rgba(*color_tuple)
            
            # Render as connected line for continuous segments
            if len(segment.points) > 1:
                points_to_draw = []  # [(x, y, is_selected), ...]
                
                for point in segment.points:
                    # Skip points outside visible range
                    if point.at < visible_start_ms or point.at > visible_end_ms:
                        continue
                    
                    # Calculate screen position
                    x = timeline_x + ((point.at - visible_start_ms) / visible_duration) * timeline_width
                    y = timeline_y + timeline_height - (point.pos / 100.0) * timeline_height
                    
                    points_to_draw.append((x, y, point.is_selected))
                
                # Draw connected line with appropriate colors
                if len(points_to_draw) > 1:
                    for i in range(len(points_to_draw) - 1):
                        # Use dimmed color if both points are unselected, otherwise use bright color
                        point1_selected = points_to_draw[i][2]
                        point2_selected = points_to_draw[i+1][2]
                        
                        if not point1_selected and not point2_selected:
                            line_color = imgui.get_color_u32_rgba(*self.PREVIEW_COLORS['unselected'])
                        else:
                            line_color = color  # Use the bright color if at least one point is selected
                        
                        draw_list.add_line(
                            points_to_draw[i][0], points_to_draw[i][1],
                            points_to_draw[i+1][0], points_to_draw[i+1][1],
                            line_color, 2.0
                        )
            
            # Render individual points (especially for new/modified points)
            for point in segment.points:
                # Skip points outside visible range
                if point.at < visible_start_ms or point.at > visible_end_ms:
                    continue
                
                # Calculate screen position
                x = timeline_x + ((point.at - visible_start_ms) / visible_duration) * timeline_width
                y = timeline_y + timeline_height - (point.pos / 100.0) * timeline_height
                
                # Choose color based on selection status
                if not point.is_selected:
                    # Unselected points are dimmed
                    point_color = imgui.get_color_u32_rgba(*self.PREVIEW_COLORS['unselected_point'])
                    # Render as small dimmed circle
                    draw_list.add_circle_filled(x, y, 2.0, point_color)
                    
                elif point.is_new:
                    # New points as circles (bright orange)
                    point_color = imgui.get_color_u32_rgba(*self.PREVIEW_COLORS['new_point'])
                    draw_list.add_circle_filled(x, y, 4.0, point_color)
                    draw_list.add_circle(x, y, 4.0, color, 1.0)
                    
                elif point.is_modified:
                    # Modified points as diamonds (bright orange)
                    point_color = imgui.get_color_u32_rgba(*self.PREVIEW_COLORS['modified_point'])
                    draw_list.add_rect_filled(x - 3, y - 3, x + 3, y + 3, point_color)
                    
                    # Draw line from original to new position if available
                    if point.original_pos is not None:
                        orig_y = timeline_y + timeline_height - (point.original_pos / 100.0) * timeline_height
                        draw_list.add_line(x, orig_y, x, y, color, 1.0)
                else:
                    # Regular selected preview points as small circles (bright orange)
                    draw_list.add_circle_filled(x, y, 2.0, color)
    
    def generate_preview_for_plugin(self, plugin_instance, funscript, axis: str, parameters: Dict[str, Any]) -> Optional[Any]:
        """
        Generate preview data by calling the plugin's preview method.
        
        Args:
            plugin_instance: The plugin instance
            funscript: The funscript to preview on
            axis: Which axis to preview
            parameters: Plugin parameters
            
        Returns:
            Preview data from the plugin, or None if preview generation fails
        """
        try:
            # Check if plugin supports preview generation
            if hasattr(plugin_instance, 'generate_preview'):
                return plugin_instance.generate_preview(funscript, axis, **parameters)
            elif hasattr(plugin_instance, 'get_preview'):
                return plugin_instance.get_preview(funscript, axis, **parameters)
            else:
                self.logger.warning(f"Plugin {plugin_instance.name} does not support preview generation")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to generate preview for plugin {plugin_instance.name}: {e}")
            return None