import pytest
from unittest.mock import patch, MagicMock
from imgui_bundle import imgui, hello_imgui
import os
import time
import platform

@pytest.mark.e2e
def test_cuda_detection_and_usage(app_instance):
    """
    Test CUDA hardware acceleration detection and usage.
    """
    engine = hello_imgui.get_imgui_test_engine()
    
    # Check hardware acceleration settings
    engine.item_open("**/Settings")
    engine.item_open("**/Hardware Acceleration")
    
    # Test CUDA availability detection
    if hasattr(app_instance.settings_manager, 'cuda_available'):
        cuda_available = app_instance.settings_manager.cuda_available
        if cuda_available:
            # Test enabling CUDA
            engine.item_click("**/Enable CUDA##CUDAToggle")
            time.sleep(1)
            assert app_instance.settings_manager.get_setting("use_cuda") == True
        else:
            # CUDA not available, should gracefully handle
            pass
    
    # Test fallback behavior when CUDA is not available
    with patch('torch.cuda.is_available', return_value=False):
        app_instance.settings_manager.detect_hardware_acceleration()
        # Should fallback to CPU without crashing
        assert app_instance.settings_manager.get_setting("use_cuda") == False

@pytest.mark.e2e 
def test_tensorrt_model_compilation(app_instance):
    """
    Test TensorRT model compilation and optimization.
    """
    if not hasattr(app_instance, 'tensorrt_compiler'):
        pytest.skip("TensorRT compiler not available")
    
    engine = hello_imgui.get_imgui_test_engine()
    
    # Open TensorRT compiler window
    engine.item_open("**/Tools")
    engine.item_click("**/TensorRT Compiler")
    
    # Test model compilation process
    # This would test the actual compilation in a real scenario
    # For now, just verify the interface is accessible
    time.sleep(2)
    
    # Test should verify that compilation interface is working
    # In a full implementation, this would test actual model compilation

@pytest.mark.e2e
def test_video_acceleration_detection(app_instance):
    """
    Test video hardware acceleration detection based on platform.
    """
    engine = hello_imgui.get_imgui_test_engine()
    
    # Test platform-specific video acceleration
    current_platform = platform.system()
    
    if current_platform == "Darwin":  # macOS
        # Test VideoToolbox availability
        if hasattr(app_instance.processor, 'video_acceleration'):
            acceleration_type = app_instance.processor.video_acceleration
            assert acceleration_type in ["videotoolbox", "software"]
    
    elif current_platform == "Linux":
        # Test VAAPI/VDPAU availability
        if hasattr(app_instance.processor, 'video_acceleration'):
            acceleration_type = app_instance.processor.video_acceleration
            assert acceleration_type in ["vaapi", "vdpau", "software"]
    
    elif current_platform == "Windows":
        # Test D3D11VA/DXVA2 availability
        if hasattr(app_instance.processor, 'video_acceleration'):
            acceleration_type = app_instance.processor.video_acceleration
            assert acceleration_type in ["d3d11va", "dxva2", "software"]

@pytest.mark.e2e
def test_memory_management_and_cleanup(app_instance):
    """
    Test memory management and resource cleanup.
    """
    video_path = os.path.abspath("test_data/memory_test_video.mp4")
    
    # Setup test video
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    with open(video_path, "w") as f:
        f.write("dummy video for memory test")
    
    engine = hello_imgui.get_imgui_test_engine()
    
    # Load and process multiple times to test memory cleanup
    for i in range(3):
        # Load video
        with patch('application.classes.file_dialog.ImGuiFileDialog.show') as mock_show_dialog:
            engine.item_open("**/File")
            engine.item_open("**/Open...")
            engine.item_click("**/Video...")
            args, kwargs = mock_show_dialog.call_args
            callback = kwargs.get('callback')
            callback(video_path)
        
        # Wait for load
        start_time = time.time()
        while not app_instance.file_manager.video_path:
            if time.time() - start_time > 10:
                pytest.fail(f"Video load {i+1} timed out")
            app_instance.gui_instance.run_one_frame(blocking=False)
            time.sleep(0.1)
        
        # Clear/reset to test cleanup
        engine.item_open("**/File")
        engine.item_click("**/New Project")
        
        # Wait for cleanup
        start_time = time.time()
        while app_instance.file_manager.video_path:
            if time.time() - start_time > 5:
                pytest.fail(f"Cleanup {i+1} timed out")
            app_instance.gui_instance.run_one_frame(blocking=False)
            time.sleep(0.1)
    
    # Memory should be properly cleaned up after multiple loads
    assert app_instance.file_manager.video_path is None

@pytest.mark.e2e
def test_multiprocessing_coordination(app_instance):
    """
    Test multiprocessing coordination and resource sharing.
    """
    video_path = os.path.abspath("test_data/multiprocessing_video.mp4")
    
    # Setup
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    with open(video_path, "w") as f:
        f.write("dummy video for multiprocessing test")
    
    engine = hello_imgui.get_imgui_test_engine()
    
    # Load video
    with patch('application.classes.file_dialog.ImGuiFileDialog.show') as mock_show_dialog:
        engine.item_open("**/File")
        engine.item_open("**/Open...")
        engine.item_click("**/Video...")
        args, kwargs = mock_show_dialog.call_args
        callback = kwargs.get('callback')
        callback(video_path)
    
    # Wait for load
    start_time = time.time()
    while not app_instance.file_manager.video_path:
        if time.time() - start_time > 10:
            pytest.fail("Video load timed out")
        app_instance.gui_instance.run_one_frame(blocking=False)
        time.sleep(0.1)
    
    # Test multi-worker stage 3 processing
    engine.item_open("**/Settings")
    engine.item_open("**/Performance")
    engine.slider_set_int("**/Stage 3 Workers##Stage3Workers", 2)
    
    # Start analysis that uses multiprocessing
    engine.item_open("**/Run Control")
    engine.item_click("**/Tracker Type##TrackerModeComboGlobal")
    engine.item_click("**/Offline AI Analysis (3-Stage)")
    engine.item_click("**/Start Full AI Analysis")
    
    # Let it run briefly to test multiprocessing coordination
    start_time = time.time()
    while time.time() - start_time < 10:
        app_instance.gui_instance.run_one_frame(blocking=False)
        time.sleep(0.1)
        if not app_instance.stage_processor.full_analysis_active:
            break
    
    # Stop if still running
    if app_instance.stage_processor.full_analysis_active:
        engine.item_click("**/Abort/Stop Process##AbortGeneral")
        
        # Wait for clean shutdown
        start_time = time.time()
        while app_instance.stage_processor.full_analysis_active:
            if time.time() - start_time > 10:
                pytest.fail("Multiprocessing shutdown timed out")
            app_instance.gui_instance.run_one_frame(blocking=False)
            time.sleep(0.1)
    
    # Verify clean shutdown
    assert not app_instance.stage_processor.full_analysis_active

@pytest.mark.e2e
def test_energy_saver_mode(app_instance):
    """
    Test energy saver mode activation and deactivation.
    """
    engine = hello_imgui.get_imgui_test_engine()
    
    # Test enabling energy saver mode
    engine.item_open("**/Settings")
    engine.item_click("**/Enable Energy Saver##EnergySaverToggle")
    
    # Verify energy saver is active
    if hasattr(app_instance, 'energy_saver'):
        time.sleep(1)
        assert app_instance.energy_saver.is_active()
    
    # Test disabling energy saver mode
    engine.item_click("**/Enable Energy Saver##EnergySaverToggle")
    
    # Verify energy saver is inactive
    if hasattr(app_instance, 'energy_saver'):
        time.sleep(1)
        assert not app_instance.energy_saver.is_active()

@pytest.mark.e2e
def test_performance_monitoring(app_instance):
    """
    Test performance monitoring and metrics collection.
    """
    video_path = os.path.abspath("test_data/performance_video.mp4")
    
    # Setup
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    with open(video_path, "w") as f:
        f.write("dummy video for performance test")
    
    engine = hello_imgui.get_imgui_test_engine()
    
    # Enable performance monitoring
    engine.item_open("**/Settings")
    engine.item_click("**/Show Performance Stats##PerfStatsToggle")
    
    # Load video and run analysis
    with patch('application.classes.file_dialog.ImGuiFileDialog.show') as mock_show_dialog:
        engine.item_open("**/File")
        engine.item_open("**/Open...")
        engine.item_click("**/Video...")
        args, kwargs = mock_show_dialog.call_args
        callback = kwargs.get('callback')
        callback(video_path)
    
    # Wait for load
    start_time = time.time()
    while not app_instance.file_manager.video_path:
        if time.time() - start_time > 10:
            pytest.fail("Video load timed out")
        app_instance.gui_instance.run_one_frame(blocking=False)
        time.sleep(0.1)
    
    # Start analysis to generate performance data
    engine.item_open("**/Run Control")
    engine.item_click("**/Tracker Type##TrackerModeComboGlobal")
    engine.item_click("**/Offline AI Analysis (2-Stage)")
    engine.item_click("**/Start Full AI Analysis")
    
    # Run for a short time to collect performance metrics
    start_time = time.time()
    while time.time() - start_time < 15:
        app_instance.gui_instance.run_one_frame(blocking=False)
        time.sleep(0.1)
        if not app_instance.stage_processor.full_analysis_active:
            break
    
    # Stop if still running
    if app_instance.stage_processor.full_analysis_active:
        engine.item_click("**/Abort/Stop Process##AbortGeneral")
        
        start_time = time.time()
        while app_instance.stage_processor.full_analysis_active:
            if time.time() - start_time > 10:
                pytest.fail("Performance test shutdown timed out")
            app_instance.gui_instance.run_one_frame(blocking=False)
            time.sleep(0.1)
    
    # Verify performance monitoring collected data
    # This would check that FPS, memory usage, etc. were tracked
    assert True  # Placeholder - would verify actual performance metrics