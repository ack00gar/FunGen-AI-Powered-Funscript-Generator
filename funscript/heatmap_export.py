"""Heatmap PNG export for funscript files.

Generates horizontal heatmap images showing speed distribution over time,
using the OFS-standard color gradient.
"""
import numpy as np
from typing import List, Dict, Optional

from application.utils.heatmap_utils import HeatmapColorMapper


class HeatmapExporter:
    """Generates and exports heatmap images from funscript actions."""

    def __init__(self, max_speed: float = 400.0):
        self._mapper = HeatmapColorMapper(max_speed=max_speed)

    def generate_heatmap_image(
        self,
        actions: List[Dict],
        duration_ms: float,
        width: int = 2000,
        height: int = 50,
    ) -> np.ndarray:
        """Generate a heatmap RGBA image as a numpy array.

        Args:
            actions: Funscript actions list [{'at': ms, 'pos': 0-100}, ...]
            duration_ms: Total video/script duration in milliseconds
            width: Output image width in pixels
            height: Output image height in pixels

        Returns:
            numpy array of shape (height, width, 4) with uint8 RGBA values
        """
        if not actions or len(actions) < 2 or duration_ms <= 0:
            # Return black image
            return np.zeros((height, width, 4), dtype=np.uint8)

        # Compute per-segment speeds
        speeds = HeatmapColorMapper.compute_segment_speeds(actions)

        # Build a 1D speed profile across the image width
        speed_profile = np.zeros(width, dtype=np.float32)
        for i in range(len(speeds)):
            t_start = actions[i]['at']
            t_end = actions[i + 1]['at']

            # Map time range to pixel columns
            px_start = int((t_start / duration_ms) * width)
            px_end = int((t_end / duration_ms) * width)
            px_start = max(0, min(width - 1, px_start))
            px_end = max(px_start + 1, min(width, px_end))

            speed_profile[px_start:px_end] = speeds[i]

        # Convert speeds to colors
        colors_rgba = self._mapper.speeds_to_colors_rgba(speed_profile)  # (width, 4) float

        # Scale to uint8
        colors_u8 = (colors_rgba * 255).astype(np.uint8)  # (width, 4)

        # Tile vertically to create the full image
        image = np.tile(colors_u8[np.newaxis, :, :], (height, 1, 1))

        return image

    def export_png(
        self,
        filepath: str,
        actions: List[Dict],
        duration_ms: float,
        width: int = 2000,
        height: int = 50,
    ):
        """Export heatmap as a PNG file.

        Args:
            filepath: Output file path (should end in .png)
            actions: Funscript actions list
            duration_ms: Total duration in ms
            width: Image width
            height: Image height
        """
        import cv2

        image_rgba = self.generate_heatmap_image(actions, duration_ms, width, height)

        # cv2 expects BGR(A), convert from RGBA
        image_bgra = np.zeros_like(image_rgba)
        image_bgra[:, :, 0] = image_rgba[:, :, 2]  # B
        image_bgra[:, :, 1] = image_rgba[:, :, 1]  # G
        image_bgra[:, :, 2] = image_rgba[:, :, 0]  # R
        image_bgra[:, :, 3] = image_rgba[:, :, 3]  # A

        cv2.imwrite(filepath, image_bgra)
