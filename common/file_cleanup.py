"""
File-pruning helpers used by components that maintain on-disk caches:
streamer's transcode cache, subtitle's .sub_audio.wav cache, etc.

All errors are logged at debug level and swallowed -- cleanup is best-effort.
"""

import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)


def cleanup_old_files(directory: str,
                      max_age_seconds: float,
                      *,
                      pattern: Optional[str] = None,
                      suffix: Optional[str] = None,
                      recursive: bool = False) -> int:
    """
    Delete files in `directory` whose mtime is older than max_age_seconds.

    Args:
        directory: Path to scan.
        max_age_seconds: Age threshold; files older than this are removed.
        pattern: Optional substring filter on the filename.
        suffix: Optional suffix filter (e.g. '.sub_audio.wav', '_h264.mp4').
        recursive: If True, walk subdirectories too.

    Returns:
        Number of files removed.
    """
    if not directory or not os.path.isdir(directory):
        return 0

    cutoff = time.time() - max_age_seconds
    removed = 0

    if recursive:
        try:
            walker = os.walk(directory)
        except OSError as e:
            logger.debug("cleanup_old_files: cannot walk %s: %s", directory, e)
            return 0
        for root, _dirs, files in walker:
            for name in files:
                if pattern is not None and pattern not in name:
                    continue
                if suffix is not None and not name.endswith(suffix):
                    continue
                removed += _maybe_remove(os.path.join(root, name), cutoff)
        return removed

    try:
        names = os.listdir(directory)
    except OSError as e:
        logger.debug("cleanup_old_files: cannot list %s: %s", directory, e)
        return 0

    for name in names:
        if pattern is not None and pattern not in name:
            continue
        if suffix is not None and not name.endswith(suffix):
            continue
        removed += _maybe_remove(os.path.join(directory, name), cutoff)
    return removed


def _maybe_remove(path: str, cutoff: float) -> int:
    try:
        if not os.path.isfile(path):
            return 0
        if os.path.getmtime(path) >= cutoff:
            return 0
        os.remove(path)
        return 1
    except OSError as e:
        logger.debug("cleanup_old_files: remove failed for %s: %s", path, e)
        return 0
