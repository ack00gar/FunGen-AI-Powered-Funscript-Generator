"""
Chapter Manager - Manages standalone chapter file operations

This module provides comprehensive chapter management capabilities:
- Save/load chapters to/from standalone JSON files
- Import/export chapters independently of project files
- Backup and restore functionality
- Chapter validation and merging
- Integration with existing project system

Author: k00gar
Version: 1.0.0
"""

import os
import json
import shutil
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

from application.utils.video_segment import VideoSegment


class ChapterManager:
    """
    Manages standalone chapter file operations.

    Features:
    - Save chapters to standalone JSON files
    - Load chapters from JSON files
    - Auto-backup before overwriting
    - Import/merge chapters from external sources
    - Validation and conflict resolution
    """

    VERSION = "1.0"
    CHAPTER_FILE_SUFFIX = "_chapters.json"
    BACKUP_SUFFIX = ".backup"

    def __init__(self, app):
        """
        Initialize the ChapterManager.

        Args:
            app: Application instance for accessing video info and settings
        """
        self.app = app
        self.logger = logging.getLogger(__name__)

    # ==================== CORE SAVE/LOAD ====================

    def get_default_chapter_filepath(self, video_path: str) -> str:
        """
        Get default chapter file path for a video.

        Args:
            video_path: Path to video file

        Returns:
            Path to chapter file (e.g., video_chapters.json)
        """
        if not video_path:
            return ""

        # Remove video extension and add chapter suffix
        base_path = os.path.splitext(video_path)[0]
        return f"{base_path}{self.CHAPTER_FILE_SUFFIX}"

    def save_chapters_to_file(self, filepath: str, chapters: List[VideoSegment],
                              video_info: Optional[Dict] = None) -> bool:
        """
        Save chapters to a standalone JSON file.

        Args:
            filepath: Destination file path
            chapters: List of VideoSegment objects to save
            video_info: Optional video metadata (fps, total_frames, etc.)

        Returns:
            True if successful
        """
        if not chapters:
            self.logger.warning("No chapters to save")
            return False

        # Auto-backup if file exists
        if os.path.exists(filepath):
            backup_enabled = self.app.app_settings.get("chapter_backup_on_regenerate", True)
            if backup_enabled:
                self._create_backup(filepath)

        # Build chapter data
        chapter_data = {
            "version": self.VERSION,
            "video_filename": os.path.basename(video_info.get("path", "")) if video_info else "",
            "video_fps": video_info.get("fps", 30.0) if video_info else 30.0,
            "video_total_frames": video_info.get("total_frames", 0) if video_info else 0,
            "created_timestamp": datetime.now().isoformat(),
            "source": "manual",
            "chapter_count": len(chapters),
            "chapters": [chapter.to_dict() for chapter in chapters]
        }

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(chapter_data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Saved {len(chapters)} chapters to {os.path.basename(filepath)}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save chapters to {filepath}: {e}", exc_info=True)
            return False

    def load_chapters_from_file(self, filepath: str) -> Tuple[List[VideoSegment], Dict[str, Any]]:
        """
        Load chapters from a standalone JSON file.

        Args:
            filepath: Source file path

        Returns:
            Tuple of (chapters_list, metadata_dict)
            Returns ([], {}) on failure
        """
        if not os.path.exists(filepath):
            self.logger.error(f"Chapter file not found: {filepath}")
            return ([], {})

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Validate version
            file_version = data.get("version", "1.0")
            if file_version != self.VERSION:
                self.logger.warning(f"Chapter file version mismatch: {file_version} vs {self.VERSION}")

            # Extract metadata
            metadata = {
                "video_filename": data.get("video_filename", ""),
                "video_fps": data.get("video_fps", 30.0),
                "video_total_frames": data.get("video_total_frames", 0),
                "created_timestamp": data.get("created_timestamp", ""),
                "source": data.get("source", "unknown")
            }

            # Load chapters
            chapters_data = data.get("chapters", [])
            chapters = []

            for chapter_dict in chapters_data:
                try:
                    chapter = VideoSegment.from_dict(chapter_dict)
                    chapters.append(chapter)
                except Exception as e:
                    self.logger.warning(f"Failed to load chapter: {e}")
                    continue

            self.logger.info(f"Loaded {len(chapters)} chapters from {os.path.basename(filepath)}")
            return (chapters, metadata)

        except Exception as e:
            self.logger.error(f"Failed to load chapters from {filepath}: {e}", exc_info=True)
            return ([], {})

    # ==================== AUTO-SAVE ====================

    def auto_save_chapters(self, chapters: List[VideoSegment], video_path: str) -> bool:
        """
        Auto-save chapters when enabled in settings.

        Args:
            chapters: Chapters to save
            video_path: Current video path

        Returns:
            True if saved or skipped (when disabled)
        """
        if not self.app.app_settings.get("chapter_auto_save_standalone", False):
            return True  # Not an error, just disabled

        if not video_path or not chapters:
            return True

        filepath = self.get_default_chapter_filepath(video_path)
        video_info = self._get_current_video_info()

        return self.save_chapters_to_file(filepath, chapters, video_info)

    # ==================== IMPORT/EXPORT ====================

    def export_chapters(self, filepath: str, chapters: List[VideoSegment],
                       include_metadata: bool = True) -> bool:
        """
        Export chapters to a shareable JSON file.
        Alias for save_chapters_to_file with user-friendly naming.

        Args:
            filepath: Destination file path
            chapters: Chapters to export
            include_metadata: Include video metadata

        Returns:
            True if successful
        """
        video_info = self._get_current_video_info() if include_metadata else None
        success = self.save_chapters_to_file(filepath, chapters, video_info)

        if success:
            self.logger.info(f"Exported {len(chapters)} chapters", extra={'status_message': True})

        return success

    def import_chapters(self, filepath: str, merge_mode: str = "replace") -> Tuple[List[VideoSegment], bool]:
        """
        Import chapters from an external file.

        Args:
            filepath: Source file path
            merge_mode: How to handle conflicts:
                - "replace": Replace all existing chapters
                - "merge": Merge with existing, skip duplicates
                - "append": Add all imported chapters to end

        Returns:
            Tuple of (imported_chapters, success)
        """
        imported_chapters, metadata = self.load_chapters_from_file(filepath)

        if not imported_chapters:
            self.logger.error("No chapters found in import file", extra={'status_message': True})
            return ([], False)

        # Validate against current video if applicable
        current_video_info = self._get_current_video_info()
        if current_video_info:
            validation_ok, warning_msg = self._validate_imported_chapters(
                imported_chapters, metadata, current_video_info
            )

            if warning_msg:
                self.logger.warning(warning_msg, extra={'status_message': True, 'duration': 5.0})

        self.logger.info(f"Imported {len(imported_chapters)} chapters ({merge_mode} mode)",
                        extra={'status_message': True})

        return (imported_chapters, True)

    # ==================== BACKUP & RESTORE ====================

    def _create_backup(self, filepath: str) -> bool:
        """
        Create a backup of an existing chapter file.

        Args:
            filepath: File to backup

        Returns:
            True if successful
        """
        if not os.path.exists(filepath):
            return True

        backup_path = f"{filepath}{self.BACKUP_SUFFIX}"

        try:
            shutil.copy2(filepath, backup_path)
            self.logger.debug(f"Created backup: {os.path.basename(backup_path)}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to create backup: {e}")
            return False

    def backup_chapters_manually(self, chapters: List[VideoSegment], video_path: str) -> bool:
        """
        Create a timestamped backup of current chapters.

        Args:
            chapters: Chapters to backup
            video_path: Current video path

        Returns:
            True if successful
        """
        if not video_path or not chapters:
            self.logger.warning("No video or chapters to backup")
            return False

        # Create timestamped backup filename
        base_path = os.path.splitext(video_path)[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{base_path}_chapters_backup_{timestamp}.json"

        video_info = self._get_current_video_info()
        success = self.save_chapters_to_file(backup_path, chapters, video_info)

        if success:
            self.logger.info(f"Backup created: {os.path.basename(backup_path)}",
                           extra={'status_message': True})

        return success

    def restore_from_backup(self, backup_path: str) -> Tuple[List[VideoSegment], bool]:
        """
        Restore chapters from a backup file.

        Args:
            backup_path: Path to backup file

        Returns:
            Tuple of (restored_chapters, success)
        """
        return self.load_chapters_from_file(backup_path)

    def list_available_backups(self, video_path: str) -> List[Tuple[str, str]]:
        """
        List all available backup files for a video.

        Args:
            video_path: Current video path

        Returns:
            List of (backup_filename, timestamp) tuples
        """
        if not video_path:
            return []

        directory = os.path.dirname(video_path)
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        pattern = f"{base_name}_chapters_backup_"

        backups = []
        try:
            for filename in os.listdir(directory):
                if filename.startswith(pattern) and filename.endswith('.json'):
                    # Extract timestamp from filename
                    timestamp_str = filename.replace(pattern, "").replace(".json", "")
                    backups.append((filename, timestamp_str))

            # Sort by timestamp (most recent first)
            backups.sort(key=lambda x: x[1], reverse=True)

        except Exception as e:
            self.logger.error(f"Failed to list backups: {e}")

        return backups

    # ==================== VALIDATION ====================

    def validate_chapters(self, chapters: List[VideoSegment],
                         video_info: Optional[Dict] = None) -> Tuple[bool, List[str]]:
        """
        Validate a list of chapters for consistency and correctness.

        Args:
            chapters: Chapters to validate
            video_info: Optional video metadata for additional validation

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        if not chapters:
            issues.append("No chapters provided")
            return (False, issues)

        # Sort chapters by start frame for validation
        sorted_chapters = sorted(chapters, key=lambda c: c.start_frame_id)

        # Check for overlaps and gaps
        for i in range(len(sorted_chapters) - 1):
            current = sorted_chapters[i]
            next_chapter = sorted_chapters[i + 1]

            # Check for invalid frame range
            if current.start_frame_id >= current.end_frame_id:
                issues.append(f"Chapter {i+1}: Invalid range ({current.start_frame_id} >= {current.end_frame_id})")

            # Check for overlap
            if current.end_frame_id >= next_chapter.start_frame_id:
                issues.append(f"Chapters {i+1} and {i+2}: Overlap detected")

            # Note: Gaps are allowed (user might have intentional gaps)

        # Check against video bounds if video_info provided
        if video_info and 'total_frames' in video_info:
            total_frames = video_info['total_frames']
            for i, chapter in enumerate(sorted_chapters):
                if chapter.end_frame_id >= total_frames:
                    issues.append(f"Chapter {i+1}: End frame ({chapter.end_frame_id}) exceeds video length ({total_frames})")
                if chapter.start_frame_id < 0:
                    issues.append(f"Chapter {i+1}: Negative start frame ({chapter.start_frame_id})")

        is_valid = len(issues) == 0
        return (is_valid, issues)

    def _validate_imported_chapters(self, imported_chapters: List[VideoSegment],
                                   import_metadata: Dict, current_video_info: Dict) -> Tuple[bool, str]:
        """
        Validate imported chapters against current video.

        Returns:
            Tuple of (is_compatible, warning_message)
        """
        # Check if video matches
        import_filename = import_metadata.get("video_filename", "")
        current_filename = os.path.basename(current_video_info.get("path", ""))

        if import_filename and current_filename and import_filename != current_filename:
            warning = f"Warning: Chapters were created for '{import_filename}', loading into '{current_filename}'"
            return (True, warning)  # Compatible but warn user

        # Check frame count
        import_frames = import_metadata.get("video_total_frames", 0)
        current_frames = current_video_info.get("total_frames", 0)

        if import_frames and current_frames and abs(import_frames - current_frames) > 10:
            warning = f"Warning: Frame count mismatch (import: {import_frames}, current: {current_frames})"
            return (True, warning)

        return (True, "")

    # ==================== MERGE OPERATIONS ====================

    def merge_chapters(self, existing_chapters: List[VideoSegment],
                      new_chapters: List[VideoSegment],
                      mode: str = "replace") -> List[VideoSegment]:
        """
        Merge two sets of chapters according to specified mode.

        Args:
            existing_chapters: Current chapters
            new_chapters: Chapters to merge in
            mode: Merge strategy:
                - "replace": Replace all with new
                - "merge": Keep non-overlapping from both
                - "append": Add new to end

        Returns:
            Merged chapter list
        """
        if mode == "replace":
            return new_chapters.copy()

        if mode == "append":
            return existing_chapters + new_chapters

        if mode == "merge":
            # Keep existing, add new only if no overlap
            merged = existing_chapters.copy()

            for new_chapter in new_chapters:
                has_overlap = False
                for existing in merged:
                    if self._chapters_overlap(new_chapter, existing):
                        has_overlap = True
                        break

                if not has_overlap:
                    merged.append(new_chapter)

            # Sort by start frame
            merged.sort(key=lambda c: c.start_frame_id)
            return merged

        return existing_chapters

    @staticmethod
    def _chapters_overlap(chapter1: VideoSegment, chapter2: VideoSegment) -> bool:
        """Check if two chapters overlap in time."""
        return not (chapter1.end_frame_id < chapter2.start_frame_id or
                   chapter2.end_frame_id < chapter1.start_frame_id)

    # ==================== UTILITY ====================

    def _get_current_video_info(self) -> Optional[Dict]:
        """Get current video information from app state."""
        if not hasattr(self.app, 'processor') or not self.app.processor:
            return None

        processor = self.app.processor

        return {
            "path": processor.video_path or "",
            "fps": processor.fps or 30.0,
            "total_frames": processor.total_frames or 0
        }

    def check_for_existing_chapters(self, video_path: str) -> bool:
        """
        Check if a chapter file exists for a video.

        Args:
            video_path: Video file path

        Returns:
            True if chapter file exists
        """
        chapter_path = self.get_default_chapter_filepath(video_path)
        return os.path.exists(chapter_path)

    def get_chapter_file_info(self, filepath: str) -> Optional[Dict]:
        """
        Get metadata from a chapter file without loading all chapters.

        Args:
            filepath: Chapter file path

        Returns:
            Metadata dict or None
        """
        if not os.path.exists(filepath):
            return None

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            return {
                "version": data.get("version", "1.0"),
                "video_filename": data.get("video_filename", ""),
                "chapter_count": data.get("chapter_count", len(data.get("chapters", []))),
                "created_timestamp": data.get("created_timestamp", ""),
                "source": data.get("source", "unknown")
            }

        except Exception as e:
            self.logger.error(f"Failed to read chapter file info: {e}")
            return None


# Global instance (initialized by ApplicationLogic)
_chapter_manager_instance: Optional[ChapterManager] = None


def get_chapter_manager() -> Optional[ChapterManager]:
    """Get global ChapterManager instance."""
    return _chapter_manager_instance


def set_chapter_manager(instance: ChapterManager):
    """Set global ChapterManager instance."""
    global _chapter_manager_instance
    _chapter_manager_instance = instance
