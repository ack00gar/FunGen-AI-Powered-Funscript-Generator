"""
Tracker package with direct modular tracker management.
"""

# Import the new tracker manager as the main tracker interface
from .tracker_manager import TrackerManager, create_tracker_manager

# For backward compatibility with 2-stage and 3-stage processors
# They expect ROITracker to be importable from tracker package
ROITracker = TrackerManager
