#!/usr/bin/env python3
"""
Test configuration for VR Funscript AI Generator tests.

This module provides centralized configuration for test data paths,
making tests more flexible and maintainable.
"""

import os
import tempfile
from pathlib import Path
from typing import List, Optional, Dict

class TestConfig:
    """Central configuration for test data and settings."""
    
    def __init__(self):
        # Base test data directory - can be configured via environment variable
        self.test_data_dir = self._get_test_data_dir()
        
        # Video test files configuration
        self.video_files = self._discover_video_files()
        
        # Test video categories
        self.short_videos = []
        self.medium_videos = []
        self.long_videos = []
        
        self._categorize_videos()
        
        # Project-specific paths  
        self.project_base = Path(__file__).parent.parent
        self.output_dir = self.project_base / "output"
        
        # Test artifacts
        self.test_artifacts_dir = self.test_data_dir / "artifacts"
        self.test_artifacts_dir.mkdir(exist_ok=True)
    
    def _get_test_data_dir(self) -> Path:
        """Get test data directory from environment or default location."""
        # Check environment variable first
        env_path = os.environ.get('FUNGEN_TEST_DATA_DIR')
        if env_path and os.path.exists(env_path):
            return Path(env_path)
        
        # Check for user's Downloads/test directory (legacy support)
        user_test_dir = Path.home() / "Downloads" / "test"
        if user_test_dir.exists():
            return user_test_dir
        
        # Default to project test_data directory
        project_test_data = Path(__file__).parent / "test_data"
        project_test_data.mkdir(exist_ok=True)
        return project_test_data
    
    def _discover_video_files(self) -> List[Path]:
        """Discover available video files for testing."""
        video_extensions = {'.mp4', '.mkv', '.avi', '.mov'}
        video_files = []
        
        if self.test_data_dir.exists():
            for ext in video_extensions:
                video_files.extend(self.test_data_dir.glob(f"*{ext}"))
                video_files.extend(self.test_data_dir.glob(f"**/*{ext}"))
        
        return sorted(set(video_files))
    
    def _categorize_videos(self):
        """Categorize videos by duration/size for appropriate test usage."""
        for video_path in self.video_files:
            file_size = video_path.stat().st_size if video_path.exists() else 0
            
            # Categorize by file size as proxy for duration
            if file_size < 10 * 1024 * 1024:  # < 10MB
                self.short_videos.append(video_path)
            elif file_size < 100 * 1024 * 1024:  # < 100MB  
                self.medium_videos.append(video_path)
            else:
                self.long_videos.append(video_path)
    
    def get_test_video(self, category: str = "short") -> Optional[Path]:
        """Get a test video from specified category."""
        videos = {
            "short": self.short_videos,
            "medium": self.medium_videos,
            "long": self.long_videos,
            "any": self.video_files
        }
        
        video_list = videos.get(category, self.short_videos)
        return video_list[0] if video_list else None
    
    def get_batch_test_videos(self, count: int = 2, category: str = "short") -> List[Path]:
        """Get multiple test videos for batch processing tests."""
        videos = {
            "short": self.short_videos,
            "medium": self.medium_videos,
            "long": self.long_videos,
            "any": self.video_files
        }
        
        video_list = videos.get(category, self.short_videos)
        return video_list[:count]
    
    def create_dummy_video(self, name: str = "dummy_video.mp4", content: str = "dummy video content") -> Path:
        """Create a dummy video file for tests that don't need real video content."""
        dummy_path = self.test_artifacts_dir / name
        dummy_path.parent.mkdir(exist_ok=True)
        
        with open(dummy_path, 'w') as f:
            f.write(content)
        
        return dummy_path
    
    def get_existing_project_dir(self, video_name: str) -> Optional[Path]:
        """Get existing project directory for a video (for carryover tests)."""
        # Remove extension from video name
        video_basename = Path(video_name).stem
        project_dir = self.output_dir / video_basename
        
        return project_dir if project_dir.exists() else None
    
    def get_stage2_overlay_path(self, video_name: str) -> Optional[Path]:
        """Get Stage 2 overlay file path for a video."""
        video_basename = Path(video_name).stem
        project_dir = self.output_dir / video_basename
        overlay_path = project_dir / f"{video_basename}_stage2_overlay.msgpack"
        
        return overlay_path if overlay_path.exists() else None
    
    def has_real_test_videos(self) -> bool:
        """Check if we have real video files for testing."""
        return len(self.video_files) > 0
    
    def get_test_summary(self) -> Dict[str, any]:
        """Get a summary of available test resources."""
        return {
            "test_data_dir": str(self.test_data_dir),
            "total_videos": len(self.video_files),
            "short_videos": len(self.short_videos),
            "medium_videos": len(self.medium_videos),
            "long_videos": len(self.long_videos),
            "has_real_videos": self.has_real_test_videos()
        }


# Global test configuration instance
config = TestConfig()


def skip_if_no_test_videos(category: str = "any"):
    """Pytest skip decorator for tests requiring real video files."""
    import pytest
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            if category == "any" and not config.has_real_test_videos():
                pytest.skip("No test videos available")
            elif category != "any" and not config.get_test_video(category):
                pytest.skip(f"No {category} test videos available")
            return func(*args, **kwargs)
        return wrapper
    return decorator


def get_test_video_path(category: str = "short") -> str:
    """Get test video path as string for backward compatibility."""
    video_path = config.get_test_video(category)
    if video_path is None:
        # Return a dummy path for tests that create their own files
        return str(config.create_dummy_video())
    return str(video_path)


def get_batch_test_video_paths(count: int = 2, category: str = "short") -> List[str]:
    """Get batch test video paths as strings for backward compatibility."""
    video_paths = config.get_batch_test_videos(count, category)
    if not video_paths:
        # Create dummy videos if no real ones available
        return [str(config.create_dummy_video(f"dummy_{i}.mp4")) for i in range(count)]
    return [str(path) for path in video_paths]


if __name__ == "__main__":
    # Print test configuration summary
    import json
    summary = config.get_test_summary()
    print("Test Configuration Summary:")
    print(json.dumps(summary, indent=2))
    
    if config.has_real_test_videos():
        print(f"\\nAvailable test videos:")
        for i, video in enumerate(config.video_files[:5], 1):
            print(f"  {i}. {video.name} ({video.stat().st_size // 1024} KB)")
        if len(config.video_files) > 5:
            print(f"  ... and {len(config.video_files) - 5} more")
    else:
        print("\\nNo real test videos found. Tests will use dummy files.")
        print(f"Set FUNGEN_TEST_DATA_DIR environment variable to specify test data location.")