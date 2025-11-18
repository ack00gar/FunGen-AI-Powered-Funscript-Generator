#!/usr/bin/env python3
"""
Fix multi-item imports from application.utils

This script handles complex multi-item imports like:
  # TODO: Fix import for: A, B, C, D
    from application.utils import A, B, C, D
And converts them to categorized imports:
  from utils.ui import A
  from utils.video import B
  from utils.processing import C, D
"""

import re
from pathlib import Path
from typing import Dict, List, Set

# Mapping of specific items to their new locations
ITEM_LOCATIONS = {
    # Core
    'AppLogger': 'utils.core',
    'get_logger': 'utils.core',

    # UI
    'get_icon_texture_manager': 'utils.ui',
    'get_logo_texture_manager': 'utils.ui',
    'primary_button_style': 'utils.ui',
    'destructive_button_style': 'utils.ui',
    'success_button_style': 'utils.ui',
    'detect_keyboard_layout': 'utils.ui',
    'get_system_scaling': 'utils.ui',

    # Network
    'HTTPClientManager': 'utils.network',
    'check_internet_connection': 'utils.network',
    'download_file': 'utils.network',
    'GitHubTokenManager': 'utils.network',
    'format_github_date': 'utils.network',

    # Processing
    'ProcessingThreadManager': 'utils.processing',
    'TaskType': 'utils.processing',
    'TaskPriority': 'utils.processing',
    'CheckpointManager': 'utils.processing',
    'StageOutputValidator': 'utils.processing',
    'Stage2SignalEnhancer': 'utils.processing',

    # Video
    'VideoSegment': 'utils.video',
    '_format_time': 'utils.video',
    'format_time': 'utils.video',
    'parse_time': 'utils.video',
    'GeneratedFileManager': 'utils.video',

    # System
    'check_write_access': 'utils.system',
    'SystemMonitor': 'utils.system',
    'detect_features': 'utils.system',
    'DependencyChecker': 'utils.system',

    # ML
    'ModelPool': 'utils.ml',
    'TensorRTCompiler': 'utils.ml',
    'export_tensorrt_engine': 'utils.ml',
    'tensorrt_compiler': 'utils.ml',

    # App
    'AutoUpdater': 'utils.app',
    'RTSSmoother': 'utils.app',
}

def parse_multi_import(line: str) -> tuple:
    """
    Parse a multi-item import line.
    Returns (module, [items]) or None if not a match.
    """
    match = re.match(r'from\s+(application\.utils|common)\s+import\s+(.+)', line)
    if not match:
        return None

    module = match.group(1)
    items_str = match.group(2)

    # Split by comma, handle parentheses
    items_str = items_str.replace('(', '').replace(')', '')
    items = [item.strip() for item in items_str.split(',')]

    return (module, items)

def fix_multi_import(line: str, indent: str = '') -> List[str]:
    """
    Convert a multi-item import to categorized imports.
    Returns list of new import lines.
    """
    result = parse_multi_import(line.strip())
    if not result:
        return [line]

    module, items = result

    # Group items by their new location
    location_groups: Dict[str, List[str]] = {}
    unknown_items: List[str] = []

    for item in items:
        if item in ITEM_LOCATIONS:
            location = ITEM_LOCATIONS[item]
            if location not in location_groups:
                location_groups[location] = []
            location_groups[location].append(item)
        else:
            unknown_items.append(item)

    # Generate new import lines
    new_lines = []
    for location in sorted(location_groups.keys()):
        items_list = ', '.join(location_groups[location])
        new_lines.append(f"{indent}from {location} import {items_list}\n")

    # If there were unknown items, keep original as comment
    if unknown_items:
        unknown_str = ', '.join(unknown_items)
        new_lines.insert(0, f"{indent}# TODO: Fix import for: {unknown_str}\n")
        new_lines.insert(1, f"{indent}{line}")

    return new_lines

def fix_file(filepath: Path) -> bool:
    """Fix all multi-item imports in a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        new_lines = []
        modified = False

        for line in lines:
            # Check if it's a multi-item import
            if 'from application.utils import' in line or 'from common import' in line:
                # Get indentation
                indent = len(line) - len(line.lstrip())
                indent_str = line[:indent]

                # Check if it has multiple items (contains comma)
                if ',' in line:
                    fixed = fix_multi_import(line, indent_str)
                    new_lines.extend(fixed)
                    modified = True
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)

        if modified:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            return True

        return False

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def main():
    """Main function."""
    root = Path('.')
    python_files = list(root.rglob('*.py'))

    # Exclude certain directories
    exclude_dirs = {'venv', '.venv', '__pycache__', 'build', 'dist', '.git', 'utils'}
    python_files = [
        f for f in python_files
        if not any(excluded in f.parts for excluded in exclude_dirs)
    ]

    total = len(python_files)
    modified = 0

    print(f"Scanning {total} files for multi-item imports...")
    print()

    for filepath in python_files:
        if fix_file(filepath):
            modified += 1
            print(f"✓ Fixed {filepath}")

    print()
    print(f"Summary: {modified}/{total} files modified")

if __name__ == '__main__':
    main()
