"""FFmpeg hardware-acceleration probing.

Extracted from ApplicationLogic. Runs off the main thread at startup,
caches the result to settings so the next launch validates immediately.
"""
from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from application.logic.app_logic import ApplicationLogic


# Hwaccels that ffmpeg may report but that don't work as a generic decode
# path through our analysis filter chain (crop, v360, scale). They're either
# encoder-side APIs or codec-specific (AV1-only etc) and break frame fetch
# silently. Hide them from the user-facing selector.
_DECODE_BLOCKLIST = {"amf"}


class HardwareAccelController:
    """Owns the ffmpeg hwaccel probe + validation of the configured method."""

    __slots__ = ("app",)

    def __init__(self, app: "ApplicationLogic") -> None:
        self.app = app

    def query_background(self) -> None:
        """Background thread body: query ffmpeg, validate configured method."""
        app = self.app
        queried = self._query_ffmpeg()
        app.available_ffmpeg_hwaccels = queried
        app.app_settings.config.performance.available_ffmpeg_hwaccels = queried

        default_hw = "auto"
        if "auto" not in queried:
            default_hw = "none" if "none" in queried else (queried[0] if queried else "none")

        current = app.app_settings.config.performance.hardware_acceleration_method or default_hw
        if current not in queried:
            app.logger.warning(
                f"Configured hardware acceleration '{current}' not listed by ffmpeg "
                f"({queried}). Falling back to '{default_hw}'.")
            app.hardware_acceleration_method = default_hw
            app.app_settings.config.performance.hardware_acceleration_method = default_hw
        else:
            app.hardware_acceleration_method = current
        app._hwaccel_query_done.set()

    def _query_ffmpeg(self) -> List[str]:
        """Parse `ffmpeg -hwaccels` output into a unique, ordered list."""
        app = self.app
        log = app.logger if hasattr(app, 'logger') and app.logger else None
        try:
            ffmpeg_path = app.app_settings.config.performance.ffmpeg_path
            result = subprocess.run(
                [ffmpeg_path, '-hide_banner', '-hwaccels'],
                capture_output=True, text=True, check=True, timeout=5,
            )
            lines = result.stdout.strip().split('\n')
            hwaccels: List[str] = []
            if lines and "Hardware acceleration methods:" in lines[0]:
                hwaccels = [line.strip() for line in lines[1:]
                            if line.strip() and line.strip() != "none"]

            standard_options = ["auto", "none"]
            unique_hwaccels = [h for h in hwaccels
                               if h not in standard_options
                               and h not in _DECODE_BLOCKLIST]
            final_options = standard_options + unique_hwaccels
            if log:
                log.debug(f"Available FFmpeg hardware accelerations: {final_options}")
            else:
                print(f"Available FFmpeg hardware accelerations: {final_options}")
            return final_options
        except FileNotFoundError:
            (log.error if log else print)("ffmpeg not found. Hardware acceleration detection failed.")
            return ["auto", "none"]
        except Exception as e:
            (log.error if log else print)(f"Error querying ffmpeg for hwaccels: {e}")
            return ["auto", "none"]
