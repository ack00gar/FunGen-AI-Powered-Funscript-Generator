"""
Temporary File Manager - Centralized temp file management.

Provides consistent temporary file locations across all modules,
with automatic cleanup and persistent storage across reboots.
"""

from pathlib import Path
import tempfile
import os
import time
import logging
from typing import Optional


class TempManager:
    """
    Centralized temporary file management.

    All temp files are stored in a consistent location:
    - Windows: %LOCALAPPDATA%/fungen/
    - macOS/Linux: ~/.local/share/fungen/

    Benefits:
    - Persistent across reboots (not in /tmp)
    - Organized by purpose (transcode, funscripts, etc.)
    - Automatic cleanup of old files
    """

    def __init__(self, app_name: str = "fungen"):
        """
        Initialize temp manager.

        Args:
            app_name: Application name for temp directory
        """
        self.logger = logging.getLogger(__name__)

        # Determine base directory (persistent location)
        if os.name == 'nt':  # Windows
            base = Path(os.getenv('LOCALAPPDATA', tempfile.gettempdir()))
        else:  # macOS, Linux
            base = Path.home() / '.local' / 'share'

        self.base_dir = base / app_name
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for different purposes
        self.transcode_cache = self.base_dir / 'transcode'
        self.funscript_cache = self.base_dir / 'funscripts'
        self.device_cache = self.base_dir / 'device_scripts'
        self.video_cache = self.base_dir / 'video_cache'

        # Create all directories
        for dir_path in [self.transcode_cache, self.funscript_cache,
                        self.device_cache, self.video_cache]:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.logger.debug(f"Temp manager initialized: {self.base_dir}")

    def get_transcode_path(self, video_hash: str) -> Path:
        """
        Get path for transcoded video.

        Args:
            video_hash: Hash of original video

        Returns:
            Path to transcoded video file
        """
        return self.transcode_cache / f"{video_hash}_h264.mp4"

    def get_funscript_cache_path(self, source: str, scene_id: str,
                                  filename: str) -> Path:
        """
        Get path for cached funscript.

        Args:
            source: Source name (xbvr, stash, local)
            scene_id: Scene identifier
            filename: Funscript filename

        Returns:
            Path to cached funscript
        """
        source_dir = self.funscript_cache / source
        source_dir.mkdir(exist_ok=True)
        return source_dir / f"{scene_id}_{filename}"

    def get_device_script_path(self, device_type: str, script_hash: str) -> Path:
        """
        Get path for device-specific script cache.

        Args:
            device_type: Device type (handy, osr, etc.)
            script_hash: Hash of script data

        Returns:
            Path to cached device script
        """
        device_dir = self.device_cache / device_type
        device_dir.mkdir(exist_ok=True)
        return device_dir / f"{script_hash}.script"

    def get_video_cache_path(self, video_hash: str, extension: str = "mp4") -> Path:
        """
        Get path for cached video file.

        Args:
            video_hash: Hash of video URL
            extension: File extension

        Returns:
            Path to cached video
        """
        return self.video_cache / f"{video_hash}.{extension}"

    def cleanup_old_files(self, max_age_days: int = 7, dry_run: bool = False):
        """
        Clean up old temp files.

        Args:
            max_age_days: Maximum file age in days
            dry_run: If True, only report what would be deleted

        Returns:
            Number of files deleted (or that would be deleted)
        """
        cutoff = time.time() - (max_age_days * 86400)
        deleted_count = 0
        total_size = 0

        for cache_dir in [self.transcode_cache, self.funscript_cache,
                         self.device_cache, self.video_cache]:
            if not cache_dir.exists():
                continue

            for file_path in cache_dir.rglob('*'):
                if file_path.is_file():
                    try:
                        stat = file_path.stat()
                        if stat.st_mtime < cutoff:
                            file_size = stat.st_size
                            if dry_run:
                                self.logger.debug(f"Would delete: {file_path.name} ({file_size / 1024 / 1024:.1f} MB)")
                            else:
                                file_path.unlink()
                                self.logger.debug(f"Deleted: {file_path.name}")
                            deleted_count += 1
                            total_size += file_size
                    except Exception as e:
                        self.logger.error(f"Error cleaning up {file_path}: {e}")

        if deleted_count > 0:
            action = "Would delete" if dry_run else "Deleted"
            self.logger.info(f"{action} {deleted_count} files ({total_size / 1024 / 1024:.1f} MB)")

        return deleted_count

    def get_cache_stats(self) -> dict:
        """
        Get statistics about cache directories.

        Returns:
            Dict with cache statistics
        """
        stats = {}

        for name, cache_dir in [
            ('transcode', self.transcode_cache),
            ('funscript', self.funscript_cache),
            ('device', self.device_cache),
            ('video', self.video_cache)
        ]:
            if cache_dir.exists():
                files = list(cache_dir.rglob('*'))
                file_count = sum(1 for f in files if f.is_file())
                total_size = sum(f.stat().st_size for f in files if f.is_file())

                stats[name] = {
                    'file_count': file_count,
                    'total_size_mb': total_size / 1024 / 1024,
                    'path': str(cache_dir)
                }
            else:
                stats[name] = {
                    'file_count': 0,
                    'total_size_mb': 0,
                    'path': str(cache_dir)
                }

        return stats


# Global instance
_temp_manager: Optional[TempManager] = None


def get_temp_manager() -> TempManager:
    """
    Get global temp manager instance.

    Returns:
        Shared TempManager instance
    """
    global _temp_manager
    if _temp_manager is None:
        _temp_manager = TempManager()
    return _temp_manager
