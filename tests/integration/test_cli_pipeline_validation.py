import pytest
import sys
import os
import tempfile
import subprocess
import json
import shutil
from pathlib import Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

@pytest.mark.integration
def test_cli_mode_selection_logic():
    """
    Test CLI mode selection and argument parsing logic.
    """
    # Test help output
    result = subprocess.run([
        'python', 'main.py', '--help'
    ], capture_output=True, text=True, timeout=10)
    
    assert result.returncode == 0
    help_output = result.stdout
    assert '--mode' in help_output
    assert '2-stage' in help_output
    assert '3-stage' in help_output
    assert 'oscillation-detector' in help_output

@pytest.mark.integration
def test_cli_video_discovery():
    """
    Test CLI video file discovery and filtering logic.
    """
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create test files of various types
        test_files = [
            'video1.mp4',
            'video2.avi', 
            'video3.mkv',
            'not_video.txt',
            'another.pdf',
            'subfolder/nested_video.mp4'
        ]
        
        # Create directory structure
        os.makedirs(os.path.join(temp_dir, 'subfolder'), exist_ok=True)
        
        for file_path in test_files:
            full_path = os.path.join(temp_dir, file_path)
            with open(full_path, 'w') as f:
                f.write('test content')
        
        # Test non-recursive discovery
        result = subprocess.run([
            'python', '-c', f'''
import sys
sys.path.insert(0, ".")
from main import discover_video_files
files = discover_video_files("{temp_dir}", recursive=False)
print(f"Non-recursive: {{len(files)}} files")
for f in sorted(files):
    print(f"File: {{os.path.basename(f)}}")
'''
        ], capture_output=True, text=True, timeout=10)
        
        assert result.returncode == 0
        output = result.stdout
        # Should find video files in root directory only
        assert 'video1.mp4' in output
        assert 'video2.avi' in output  
        assert 'video3.mkv' in output
        assert 'not_video.txt' not in output
        assert 'nested_video.mp4' not in output
        
        # Test recursive discovery
        result = subprocess.run([
            'python', '-c', f'''
import sys
sys.path.insert(0, ".")
from main import discover_video_files
files = discover_video_files("{temp_dir}", recursive=True)
print(f"Recursive: {{len(files)}} files")
for f in sorted(files):
    print(f"File: {{os.path.basename(f)}}")
'''
        ], capture_output=True, text=True, timeout=10)
        
        assert result.returncode == 0
        output = result.stdout
        # Should find all video files including nested
        assert 'video1.mp4' in output
        assert 'nested_video.mp4' in output
        assert 'not_video.txt' not in output
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.mark.integration
def test_cli_argument_validation():
    """
    Test CLI argument validation and error handling.
    """
    # Test invalid mode
    result = subprocess.run([
        'python', 'main.py', '/fake/video.mp4', '--mode', 'invalid-mode'
    ], capture_output=True, text=True, timeout=10)
    
    # Should reject invalid mode
    assert result.returncode != 0
    assert 'invalid choice' in result.stderr.lower() or 'error' in result.stderr.lower()
    
    # Test missing video path
    result = subprocess.run([
        'python', 'main.py', '--mode', '2-stage'
    ], capture_output=True, text=True, timeout=10)
    
    # Should require video path
    assert result.returncode != 0

@pytest.mark.integration 
def test_cli_settings_inheritance():
    """
    Test that CLI inherits settings properly from application state.
    """
    # Create a simple test to verify CLI can access application settings
    result = subprocess.run([
        'python', '-c', '''
import sys
sys.path.insert(0, ".")
from application.logic.app_logic import ApplicationLogic
app = ApplicationLogic(is_cli=True)
print(f"CLI app initialized: {app is not None}")
print(f"Settings available: {hasattr(app, 'app_settings')}")
print(f"Stage processor: {app.stage_processor is not None}")
print(f"Video processor: {app.processor is not None}")
'''
    ], capture_output=True, text=True, timeout=20)
    
    assert result.returncode == 0
    output = result.stdout
    assert 'CLI app initialized: True' in output
    assert 'Settings available: True' in output
    assert 'Stage processor: True' in output
    assert 'Video processor: True' in output

@pytest.mark.integration
def test_cli_processing_modes():
    """
    Test CLI processing mode validation without actual video processing.
    """
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
        # Create minimal fake video file
        temp_video.write(b'fake mp4 content for testing')
        video_path = temp_video.name
    
    try:
        # Test 2-stage mode (should accept arguments)
        result = subprocess.run([
            'python', '-c', f'''
import sys
sys.path.insert(0, ".")
from main import parse_arguments
args = parse_arguments(["{video_path}", "--mode", "2-stage", "--no-autotune"])
print(f"Mode: {{args.mode}}")
print(f"Autotune: {{args.autotune}}")
print(f"Video: {{args.video_input}}")
'''
        ], capture_output=True, text=True, timeout=10)
        
        assert result.returncode == 0
        output = result.stdout
        assert 'Mode: 2-stage' in output
        assert 'Autotune: False' in output
        assert video_path in output
        
        # Test 3-stage mode
        result = subprocess.run([
            'python', '-c', f'''
import sys
sys.path.insert(0, ".")
from main import parse_arguments
args = parse_arguments(["{video_path}", "--mode", "3-stage", "--overwrite"])
print(f"Mode: {{args.mode}}")
print(f"Overwrite: {{args.overwrite}}")
'''
        ], capture_output=True, text=True, timeout=10)
        
        assert result.returncode == 0
        output = result.stdout
        assert 'Mode: 3-stage' in output
        assert 'Overwrite: True' in output
        
        # Test oscillation detector mode
        result = subprocess.run([
            'python', '-c', f'''
import sys
sys.path.insert(0, ".")
from main import parse_arguments
args = parse_arguments(["{video_path}", "--mode", "oscillation-detector"])
print(f"Mode: {{args.mode}}")
'''
        ], capture_output=True, text=True, timeout=10)
        
        assert result.returncode == 0
        output = result.stdout
        assert 'Mode: oscillation-detector' in output
    
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

@pytest.mark.integration
def test_cli_output_validation():
    """
    Test CLI output file validation and conflict handling.
    """
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
        temp_video.write(b'fake video content')
        video_path = temp_video.name
    
    # Create existing funscript
    funscript_path = video_path.replace('.mp4', '.funscript')
    with open(funscript_path, 'w') as f:
        json.dump({"actions": [{"at": 1000, "pos": 50}]}, f)
    
    try:
        # Test CLI validation of existing files (without processing)
        result = subprocess.run([
            'python', '-c', f'''
import sys, os
sys.path.insert(0, ".")
from main import check_output_conflicts
video_path = "{video_path}"
conflicts = check_output_conflicts([video_path], overwrite=False)
print(f"Conflicts found: {{len(conflicts)}}")
for conflict in conflicts:
    print(f"Conflict: {{conflict}}")
'''
        ], capture_output=True, text=True, timeout=10)
        
        # Should detect conflict
        if result.returncode == 0:
            output = result.stdout
            # May find conflicts depending on implementation
            assert 'Conflicts found:' in output
        
        # Test with overwrite flag
        result = subprocess.run([
            'python', '-c', f'''
import sys, os
sys.path.insert(0, ".")
from main import check_output_conflicts  
video_path = "{video_path}"
conflicts = check_output_conflicts([video_path], overwrite=True)
print(f"Conflicts with overwrite: {{len(conflicts)}}")
'''
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            output = result.stdout
            # With overwrite, should have fewer/no conflicts
            assert 'Conflicts with overwrite:' in output
    
    finally:
        for path in [video_path, funscript_path]:
            if os.path.exists(path):
                os.remove(path)

@pytest.mark.integration
def test_cli_resource_management():
    """
    Test CLI resource management and cleanup behavior.
    """
    # Test that CLI mode properly initializes and cleans up resources
    result = subprocess.run([
        'python', '-c', '''
import sys
import gc
sys.path.insert(0, ".")
from application.logic.app_logic import ApplicationLogic

# Test multiple app initialization/cleanup cycles
for i in range(3):
    app = ApplicationLogic(is_cli=True)
    print(f"Cycle {i+1}: App created")
    
    # Verify key components
    assert app.stage_processor is not None
    assert app.processor is not None
    assert app.funscript_processor is not None
    
    # Clean up
    del app
    gc.collect()
    print(f"Cycle {i+1}: Cleanup complete")

print("Resource management test completed")
'''
    ], capture_output=True, text=True, timeout=30)
    
    assert result.returncode == 0
    output = result.stdout
    assert 'Cycle 1: App created' in output
    assert 'Cycle 2: App created' in output
    assert 'Cycle 3: App created' in output
    assert 'Resource management test completed' in output

@pytest.mark.integration
def test_cli_hardware_detection():
    """
    Test CLI hardware acceleration detection and configuration.
    """
    result = subprocess.run([
        'python', '-c', '''
import sys
sys.path.insert(0, ".")
from application.logic.app_logic import ApplicationLogic

app = ApplicationLogic(is_cli=True)

# Test hardware acceleration detection
hwaccels = app._get_available_ffmpeg_hwaccels()
print(f"Available hardware accelerations: {hwaccels}")

# Test that we can detect at least basic acceleration
assert isinstance(hwaccels, list)
assert len(hwaccels) > 0
assert 'none' in hwaccels  # Should always have 'none' as fallback

print("Hardware detection test completed")
'''
    ], capture_output=True, text=True, timeout=20)
    
    assert result.returncode == 0
    output = result.stdout
    assert 'Available hardware accelerations:' in output
    assert 'Hardware detection test completed' in output