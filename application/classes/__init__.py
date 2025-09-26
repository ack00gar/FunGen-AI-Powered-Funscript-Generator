"""
Classes package initialization.
"""

from .file_dialog import ImGuiFileDialog
from .gauge import GaugeWindow
from .interactive_timeline import InteractiveFunscriptTimeline
from .movement_bar import MovementBarWindow
# Backward compatibility alias
LRDialWindow = MovementBarWindow
from .menu import MainMenu
from .project_manager import ProjectManager
from .settings_manager import AppSettings
from .shortcut_manager import ShortcutManager
from .undo_redo_manager import UndoRedoManager
from .simulator_3d import Simulator3DWindow