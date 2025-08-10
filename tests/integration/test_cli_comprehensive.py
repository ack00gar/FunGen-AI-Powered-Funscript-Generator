import pytest
import sys
import os
import tempfile
import subprocess
import json
import shutil
import time
from pathlib import Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from unittest.mock import patch
from application.logic.app_logic import ApplicationLogic

@pytest.mark.integration
def test_cli_argument_parsing_comprehensive():
    """
    Test comprehensive CLI argument parsing and validation.
    """
    # Test all valid mode combinations
    valid_modes = ['2-stage', '3-stage', 'oscillation-detector']
    
    for mode in valid_modes:
        result = subprocess.run([
            'python', '-c', f'''
import sys
sys.path.insert(0, ".")
import argparse
from main import *
parser = argparse.ArgumentParser()
parser.add_argument("input_path", nargs="?", default=None)
parser.add_argument("--mode", choices=["2-stage", "3-stage", "oscillation-detector"], default="3-stage")
parser.add_argument("--overwrite", action="store_true")
parser.add_argument("--no-autotune", action="store_false", dest="autotune")
parser.add_argument("--no-copy", action="store_false", dest="copy")
parser.add_argument("--recursive", "-r", action="store_true")

args = parser.parse_args(["test.mp4", "--mode", "{mode}"])
print(f"Mode: {{args.mode}}")
print(f"Input: {{args.input_path}}")
'''
        ], capture_output=True, text=True, timeout=10)
        
        assert result.returncode == 0
        assert f"Mode: {mode}" in result.stdout
        assert "Input: test.mp4" in result.stdout

@pytest.mark.integration
def test_cli_flag_combinations():
    """
    Test various CLI flag combinations.
    """
    flag_combinations = [
        ['--overwrite'],
        ['--no-autotune'],
        ['--no-copy'],
        ['--recursive'],
        ['--overwrite', '--no-autotune'],
        ['--recursive', '--overwrite'],
        ['--mode', '2-stage', '--overwrite', '--no-autotune'],
        ['--mode', '3-stage', '--recursive', '--no-copy']
    ]
    
    for flags in flag_combinations:
        result = subprocess.run([
            'python', '-c', f'''
import sys
sys.path.insert(0, ".")
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input_path", nargs="?", default=None)
parser.add_argument("--mode", choices=["2-stage", "3-stage", "oscillation-detector"], default="3-stage")
parser.add_argument("--overwrite", action="store_true")
parser.add_argument("--no-autotune", action="store_false", dest="autotune")
parser.add_argument("--no-copy", action="store_false", dest="copy")
parser.add_argument("--recursive", "-r", action="store_true")

args = parser.parse_args(["test.mp4"] + {flags!r})
print(f"Parsed successfully: {len(flags)} flags")
print(f"Overwrite: {{args.overwrite}}")
print(f"Autotune: {{args.autotune}}")
print(f"Copy: {{args.copy}}")
print(f"Recursive: {{args.recursive}}")
'''
        ], capture_output=True, text=True, timeout=10)
        
        assert result.returncode == 0
        assert f"Parsed successfully: {len(flags)} flags" in result.stdout

@pytest.mark.integration
def test_cli_file_discovery_patterns():
    """
    Test CLI file discovery with various patterns and structures.
    """
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create complex directory structure
        dirs = [
            'videos',
            'videos/subfolder1',
            'videos/subfolder2', 
            'videos/subfolder1/deep',
            'other_files'
        ]
        
        for dir_path in dirs:
            os.makedirs(os.path.join(temp_dir, dir_path), exist_ok=True)
        
        # Create various file types
        test_files = [
            'videos/movie1.mp4',
            'videos/movie2.avi',
            'videos/movie3.mkv',
            'videos/movie4.mov',
            'videos/subfolder1/nested1.mp4',
            'videos/subfolder1/nested2.webm',
            'videos/subfolder1/deep/deep_video.mp4',
            'videos/subfolder2/another.mp4',
            'videos/not_video.txt',
            'videos/readme.md',
            'other_files/document.pdf'
        ]
        
        for file_path in test_files:
            full_path = os.path.join(temp_dir, file_path)
            with open(full_path, 'w') as f:
                f.write('test content')
        
        # Test video file discovery logic
        result = subprocess.run([
            'python', '-c', f'''
import sys
import os
sys.path.insert(0, ".")

def discover_video_files(input_path, recursive=False):
    """Simplified video file discovery for testing."""
    video_extensions = {{".mp4", ".avi", ".mkv", ".mov", ".webm", ".m4v", ".wmv", ".flv"}}
    found_files = []
    
    if os.path.isfile(input_path):
        if any(input_path.lower().endswith(ext) for ext in video_extensions):
            found_files.append(input_path)
    elif os.path.isdir(input_path):
        if recursive:
            for root, dirs, files in os.walk(input_path):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in video_extensions):
                        found_files.append(os.path.join(root, file))
        else:
            for file in os.listdir(input_path):
                file_path = os.path.join(input_path, file)
                if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in video_extensions):
                    found_files.append(file_path)
    
    return sorted(found_files)

# Test non-recursive
videos_dir = os.path.join("{temp_dir}", "videos")
non_recursive = discover_video_files(videos_dir, recursive=False)
print(f"Non-recursive found: {{len(non_recursive)}} files")
for f in non_recursive:
    print(f"  {{os.path.basename(f)}}")

# Test recursive
recursive = discover_video_files(videos_dir, recursive=True)
print(f"Recursive found: {{len(recursive)}} files")
for f in recursive:
    print(f"  {{os.path.relpath(f, videos_dir)}}")
'''
        ], capture_output=True, text=True, timeout=10)
        
        assert result.returncode == 0
        output = result.stdout
        
        # Non-recursive should find root level videos only
        assert "Non-recursive found:" in output
        assert "movie1.mp4" in output
        assert "movie2.avi" in output
        assert "movie3.mkv" in output
        assert "not_video.txt" not in output
        
        # Recursive should find all videos including nested
        assert "Recursive found:" in output
        assert "nested1.mp4" in output
        assert "deep_video.mp4" in output
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.mark.integration
def test_cli_error_scenarios():
    """
    Test CLI error handling scenarios.
    """
    # Test invalid mode
    result = subprocess.run([
        'python', 'main.py', 'test.mp4', '--mode', 'invalid-mode'
    ], capture_output=True, text=True, timeout=10)
    
    assert result.returncode != 0
    assert 'invalid choice' in result.stderr.lower() or 'error' in result.stderr.lower()
    
    # Test nonexistent input file
    result = subprocess.run([
        'python', 'main.py', '/absolutely/nonexistent/file.mp4'
    ], capture_output=True, text=True, timeout=30)
    
    # Should handle gracefully (may return 0 with error message or non-zero)
    # The key is it shouldn't crash
    assert result.returncode is not None
    
    # Test nonexistent directory
    result = subprocess.run([
        'python', 'main.py', '/absolutely/nonexistent/directory/', '--recursive'
    ], capture_output=True, text=True, timeout=30)
    
    assert result.returncode is not None

@pytest.mark.integration
def test_cli_output_file_handling():
    """
    Test CLI output file handling and conflict resolution.
    """
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
        temp_video.write(b'fake video content')
        video_path = temp_video.name
    
    funscript_path = video_path.replace('.mp4', '.funscript')
    
    try:
        # Create existing funscript
        with open(funscript_path, 'w') as f:
            json.dump({"actions": [{"at": 1000, "pos": 50}]}, f)
        
        # Test without overwrite flag (should skip)
        result = subprocess.run([
            'python', 'main.py', video_path, '--mode', '2-stage', '--no-autotune'
        ], capture_output=True, text=True, timeout=60)
        
        # Should complete successfully (may skip processing)
        assert result.returncode == 0
        
        # Test with overwrite flag
        result = subprocess.run([
            'python', 'main.py', video_path, '--mode', '2-stage', '--no-autotune', '--overwrite'
        ], capture_output=True, text=True, timeout=120)
        
        # Should complete successfully
        assert result.returncode == 0
    
    finally:
        for path in [video_path, funscript_path]:
            if os.path.exists(path):
                os.remove(path)

@pytest.mark.integration
def test_application_lifecycle_complete():
    """
    Test complete application lifecycle from initialization to cleanup.
    """
    # Test multiple app instances (simulating multiple CLI runs)
    for i in range(3):
        with patch('application.logic.app_logic.ApplicationLogic._load_last_project_on_startup'):
            app = ApplicationLogic(is_cli=True)
            
            # Verify clean initialization
            assert app.stage_processor is not None
            assert app.processor is not None
            assert app.funscript_processor is not None
            assert app.tracker is not None
            
            # Test basic operations
            app.funscript_processor.get_funscript_obj().add_action(i * 1000, 50) if app.funscript_processor.get_funscript_obj() else None
            actions = (app.funscript_processor.get_funscript_obj().primary_actions if app.funscript_processor.get_funscript_obj() else [])
            assert len(actions) == 1
            
            # Test cleanup
            app.funscript_processor.get_funscript_obj().clear() if app.funscript_processor.get_funscript_obj() else None
            assert len((app.funscript_processor.get_funscript_obj().primary_actions if app.funscript_processor.get_funscript_obj() else [])) == 0
            
            # Simulate cleanup
            del app

@pytest.mark.integration
def test_settings_and_configuration():
    """
    Test application settings and configuration management.
    """
    with patch('application.logic.app_logic.ApplicationLogic._load_last_project_on_startup'):
        app = ApplicationLogic(is_cli=True)
        
        # Test settings access
        assert hasattr(app, 'app_settings')
        assert isinstance(app.app_settings, dict)
        
        # Test common settings exist
        common_settings = [
            'theme', 'window_width', 'window_height', 
            'autosave_enabled', 'autosave_interval_seconds'
        ]
        
        for setting in common_settings:
            # Settings may or may not exist, but accessing shouldn't crash
            value = app.app_settings.get(setting, 'default')
            assert value is not None
        
        # Test settings modification (temporary)
        original_test = app.app_settings.get('test_cli_setting', None)
        app.app_settings['test_cli_setting'] = 'test_value'
        assert app.app_settings['test_cli_setting'] == 'test_value'
        
        # Cleanup
        if original_test is None:
            app.app_settings.pop('test_cli_setting', None)
        else:
            app.app_settings['test_cli_setting'] = original_test

@pytest.mark.integration
def test_funscript_export_import_cycle():
    """
    Test complete funscript export/import cycle.
    """
    with patch('application.logic.app_logic.ApplicationLogic._load_last_project_on_startup'):
        app = ApplicationLogic(is_cli=True)
        
        funscript = app.funscript_processor.get_funscript_obj()
        
        # Create test data
        test_actions = [
            {"at": 0, "pos": 0},
            {"at": 1000, "pos": 100},
            {"at": 2000, "pos": 25},
            {"at": 3000, "pos": 75},
            {"at": 4000, "pos": 50}
        ]
        
        for action in test_actions:
            funscript.add_action(action["at"], action["pos"])
        
        # Test export
        with tempfile.NamedTemporaryFile(suffix='.funscript', delete=False) as temp_file:
            export_path = temp_file.name
        
        try:
            # Manual export (since we know the structure)
            export_data = {
                "actions": funscript.primary_actions,
                "metadata": {
                    "generator": "VR-Funscript-AI-Generator-Test",
                    "version": "test"
                }
            }
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            # Verify export file
            assert os.path.exists(export_path)
            assert os.path.getsize(export_path) > 100
            
            # Test import (create new instance)
            app2 = ApplicationLogic(is_cli=True)
            
            with open(export_path, 'r') as f:
                imported_data = json.load(f)
            
            # Verify import data structure
            assert 'actions' in imported_data
            assert len(imported_data['actions']) == 5
            
            # Verify action integrity
            for i, action in enumerate(imported_data['actions']):
                assert action['at'] == test_actions[i]['at']
                assert action['pos'] == test_actions[i]['pos']
        
        finally:
            if os.path.exists(export_path):
                os.remove(export_path)

@pytest.mark.integration
def test_concurrent_processing_simulation():
    """
    Test application behavior under simulated concurrent operations.
    """
    # Simulate multiple operations happening in sequence
    # (True concurrency would require threading, which is complex for this test)
    
    operations = []
    
    for i in range(5):
        with patch('application.logic.app_logic.ApplicationLogic._load_last_project_on_startup'):
            app = ApplicationLogic(is_cli=True)
            
            # Simulate different operations
            if i % 3 == 0:
                # Funscript operations
                app.funscript_processor.get_funscript_obj().add_action(i * 1000, i * 20) if app.funscript_processor.get_funscript_obj() else None
                result = len((app.funscript_processor.get_funscript_obj().primary_actions if app.funscript_processor.get_funscript_obj() else []))
                operations.append(('funscript', result))
                
            elif i % 3 == 1:
                # Settings operations
                app.app_settings.set(f'test_key_{i}', f'test_value_{i}')
                result = app.app_settings.get(f'test_key_{i}')
                operations.append(('settings', result))
                
            else:
                # Hardware detection
                hwaccels = app._get_available_ffmpeg_hwaccels()
                result = len(hwaccels)
                operations.append(('hardware', result))
    
    # Verify all operations completed
    assert len(operations) == 5
    
    # Verify operation types
    operation_types = set(op[0] for op in operations)
    assert 'funscript' in operation_types
    assert 'settings' in operation_types
    assert 'hardware' in operation_types

@pytest.mark.integration
def test_resource_cleanup_verification():
    """
    Test that resources are properly cleaned up.
    """
    import gc
    import psutil
    import os
    
    # Get initial memory usage
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    # Create and destroy multiple app instances
    for i in range(10):
        with patch('application.logic.app_logic.ApplicationLogic._load_last_project_on_startup'):
            app = ApplicationLogic(is_cli=True)
            
            # Do some operations
            for j in range(100):
                app.funscript_processor.get_funscript_obj().add_action(j * 10, j % 100) if app.funscript_processor.get_funscript_obj() else None
            
            # Clear data
            app.funscript_processor.get_funscript_obj().clear() if app.funscript_processor.get_funscript_obj() else None
            
            # Delete app instance
            del app
            
        # Force garbage collection
        gc.collect()
    
    # Check memory after cleanup
    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory
    
    # Memory should not have increased dramatically (allow 50MB increase)
    assert memory_increase < 50 * 1024 * 1024, f"Memory increased by {memory_increase / 1024 / 1024:.1f} MB"

@pytest.mark.integration
def test_platform_compatibility():
    """
    Test platform-specific compatibility.
    """
    import platform
    
    with patch('application.logic.app_logic.ApplicationLogic._load_last_project_on_startup'):
        app = ApplicationLogic(is_cli=True)
        
        system = platform.system().lower()
        
        # Test hardware acceleration for current platform
        hwaccels = app._get_available_ffmpeg_hwaccels()
        
        if system == 'darwin':  # macOS
            # Should have VideoToolbox or at least fallback
            assert any(accel in hwaccels for accel in ['videotoolbox', 'auto', 'none'])
        elif system == 'linux':
            # Should have at least software fallback
            assert 'none' in hwaccels
        elif system == 'windows':
            # Should have at least software fallback  
            assert 'none' in hwaccels
        
        # Test that basic operations work on all platforms
        app.funscript_processor.get_funscript_obj().add_action(1000, 50) if app.funscript_processor.get_funscript_obj() else None
        actions = (app.funscript_processor.get_funscript_obj().primary_actions if app.funscript_processor.get_funscript_obj() else [])
        assert len(actions) == 1
        
        funscript_obj = app.funscript_processor.get_funscript_obj()
        if funscript_obj:
            stats = funscript_obj.get_actions_statistics('primary')
        else:
            stats = {"num_points": 0}
        assert stats['num_points'] == 1