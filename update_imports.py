#!/usr/bin/env python3
"""
Script to update imports from old structure to new utils structure.

This script updates all import statements to use the new consolidated utils package.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

# Mapping of old imports to new imports
IMPORT_REPLACEMENTS = {
    # Core utilities
    'from application.utils import AppLogger': 'from utils.core import AppLogger',
    'from utils.core.logger import': 'from utils.core.logger import',
    'from utils.core.exceptions import': 'from utils.core.exceptions import',
    'from utils.core.result import': 'from utils.core.result import',
    'from utils.core.temp_manager import': 'from utils.core.temp_manager import',
    'from common import': 'from utils.core import',

    # UI utilities
    'from application.utils import get_icon_texture_manager': 'from utils.ui import get_icon_texture_manager',
    'from application.utils import get_logo_texture_manager': 'from utils.ui import get_logo_texture_manager',
    'from application.utils import primary_button_style': 'from utils.ui import primary_button_style',
    'from application.utils import destructive_button_style': 'from utils.ui import destructive_button_style',
    'from application.utils import success_button_style': 'from utils.ui import success_button_style',
    'from utils.ui.button_styles import': 'from utils.ui.button_styles import',
    'from utils.ui.icon_texture import': 'from utils.ui.icon_texture import',
    'from utils.ui.logo_texture import': 'from utils.ui.logo_texture import',
    'from application.utils.keyboard_layout_detector import': 'from utils.ui.keyboard_layout_detector import',
    'from application.utils.system_scaling import': 'from utils.ui.system_scaling import',

    # Network utilities
    'from utils.network.http_client_manager import': 'from utils.network.http_client_manager import',
    'from utils.network.network_utils import': 'from utils.network.network_utils import',
    'from application.utils import check_internet_connection': 'from utils.network import check_internet_connection',
    'from utils.network.github_token_manager import': 'from utils.network.github_token_manager import',
    'from application.utils import GitHubTokenManager': 'from utils.network import GitHubTokenManager',

    # Processing utilities
    'from application.utils import ProcessingThreadManager': 'from utils.processing import ProcessingThreadManager',
    'from application.utils import TaskType': 'from utils.processing import TaskType',
    'from application.utils import TaskPriority': 'from utils.processing import TaskPriority',
    'from utils.processing.processing_thread_manager import': 'from utils.processing.processing_thread_manager import',
    'from application.utils import CheckpointManager': 'from utils.processing import CheckpointManager',
    'from utils.processing.checkpoint_manager import': 'from utils.processing.checkpoint_manager import',
    'from application.utils.stage_output_validator import': 'from utils.processing.stage_output_validator import',
    'from application.utils.stage2_signal_enhancer import': 'from utils.processing.stage2_signal_enhancer import',

    # Video utilities
    'from application.utils import VideoSegment': 'from utils.video import VideoSegment',
    'from utils.video.video_segment import': 'from utils.video.video_segment import',
    'from application.utils import _format_time': 'from utils.video import format_time as _format_time',
    'from utils.video.time_format import': 'from utils.video.time_format import',
    'from application.utils import GeneratedFileManager': 'from utils.video import GeneratedFileManager',
    'from application.utils.generated_file_manager import': 'from utils.video.generated_file_manager import',
    'from application.utils import format_github_date': 'from utils.video import format_time',

    # System utilities
    'from application.utils import check_write_access': 'from utils.system import check_write_access',
    'from utils.system.write_access import': 'from utils.system.write_access import',
    'from utils.system.system_monitor import': 'from utils.system.system_monitor import',
    'from application.utils.feature_detection import': 'from utils.system.feature_detection import',
    'from utils.system.dependency_checker import': 'from utils.system.dependency_checker import',
    'from application.utils import DependencyChecker': 'from utils.system import DependencyChecker',

    # ML utilities
    'from utils.ml.model_pool import': 'from utils.ml.model_pool import',
    'from utils.ml.tensorrt_compiler import': 'from utils.ml.tensorrt_compiler import',
    'from application.utils import tensorrt_compiler': 'from utils.ml import tensorrt_compiler',
    'from application.utils.tensorrt_export_engine_model import': 'from utils.ml.tensorrt_export_engine_model import',

    # App utilities
    'from application.utils import AutoUpdater': 'from utils.app import AutoUpdater',
    'from utils.app.updater import': 'from utils.app.updater import',
    'from application.utils.rts_smoother import': 'from utils.app.rts_smoother import',
}

def update_file(filepath: Path) -> Tuple[bool, int]:
    """
    Update imports in a single file.

    Returns:
        Tuple of (was_modified, num_replacements)
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        replacements = 0

        for old_import, new_import in IMPORT_REPLACEMENTS.items():
            if old_import in content:
                content = content.replace(old_import, new_import)
                replacements += content.count(new_import) - original_content.count(new_import)
                original_content = content

        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, replacements

        return False, 0

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False, 0

def main():
    """Main function to update all imports."""
    root = Path('.')
    python_files = list(root.rglob('*.py'))

    # Exclude certain directories
    exclude_dirs = {'venv', '.venv', '__pycache__', 'build', 'dist', '.git', 'utils'}
    python_files = [
        f for f in python_files
        if not any(excluded in f.parts for excluded in exclude_dirs)
    ]

    total_files = len(python_files)
    modified_files = 0
    total_replacements = 0

    print(f"Scanning {total_files} Python files...")
    print()

    for filepath in python_files:
        was_modified, num_replacements = update_file(filepath)
        if was_modified:
            modified_files += 1
            total_replacements += num_replacements
            print(f"✓ Updated {filepath} ({num_replacements} replacements)")

    print()
    print(f"Summary:")
    print(f"  Files scanned: {total_files}")
    print(f"  Files modified: {modified_files}")
    print(f"  Total replacements: {total_replacements}")

if __name__ == '__main__':
    main()
