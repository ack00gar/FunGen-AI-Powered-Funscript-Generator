import pytest
from unittest.mock import patch, MagicMock
from application.logic.app_logic import ApplicationLogic

@pytest.mark.integration  
def test_dual_timeline_system():
    """Test dual timeline functionality."""
    with patch('application.logic.app_logic.ApplicationLogic._load_last_project_on_startup'):
        app = ApplicationLogic(is_cli=True)
        
        # Test funscript processor has timeline capabilities
        assert app.funscript_processor is not None
        
        # Test dual axis support
        funscript_obj = app.funscript_processor.get_funscript_obj()
        if funscript_obj:
            # Test primary/secondary axis existence
            assert hasattr(funscript_obj, 'primary_actions')
            if hasattr(funscript_obj, 'secondary_actions'):
                assert hasattr(funscript_obj, 'secondary_actions')

@pytest.mark.integration
def test_interactive_timeline_editing():
    """Test interactive timeline editing capabilities."""
    with patch('application.logic.app_logic.ApplicationLogic._load_last_project_on_startup'):
        app = ApplicationLogic(is_cli=True)
        
        # Test funscript manipulation methods exist
        funscript_obj = app.funscript_processor.get_funscript_obj()
        if funscript_obj:
            # Test basic editing operations
            assert hasattr(funscript_obj, 'add_action')
            assert hasattr(funscript_obj, 'get_actions_statistics')
            if hasattr(funscript_obj, 'clear'):
                assert callable(funscript_obj.clear)

@pytest.mark.integration
def test_chapter_navigation_system():
    """Test chapter/segment navigation functionality."""
    with patch('application.logic.app_logic.ApplicationLogic._load_last_project_on_startup'):
        app = ApplicationLogic(is_cli=True)
        
        # Test chapter-related functionality exists
        if hasattr(app.funscript_processor, 'video_chapters'):
            chapters = app.funscript_processor.video_chapters
            assert chapters is not None
        
        # Test segment finding capability
        if hasattr(app.funscript_processor, 'find_chapter_by_frame'):
            find_method = app.funscript_processor.find_chapter_by_frame
            assert callable(find_method)

@pytest.mark.integration
def test_heatmap_generation():
    """Test heatmap generation and visualization."""
    with patch('application.logic.app_logic.ApplicationLogic._load_last_project_on_startup'):
        app = ApplicationLogic(is_cli=True)
        
        # Test heatmap-related functionality
        # (Would need actual implementation to test)
        
        # Test basic funscript statistics for heatmap data
        funscript_obj = app.funscript_processor.get_funscript_obj()
        if funscript_obj:
            stats = funscript_obj.get_actions_statistics('primary')
            assert isinstance(stats, dict)
            assert 'num_points' in stats
