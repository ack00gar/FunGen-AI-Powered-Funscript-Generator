"""
Utils package initialization.
"""

from .generated_file_manager import GeneratedFileManager
from .github_token_manager import GitHubTokenManager
from .logger import AppLogger, StatusMessageHandler, ColoredFormatter
from .logo_texture import LogoTextureManager, get_logo_texture_manager
from .icon_texture import IconTextureManager, get_icon_texture_manager
from .processing_thread_manager import ProcessingThreadManager, TaskType, TaskPriority
from .time_format import _format_time, format_github_date
from .network_utils import check_internet_connection
from .updater import GitHubAPIClient, AutoUpdater
from .video_segment import VideoSegment
from .write_access import check_write_access
from .button_styles import primary_button_style, destructive_button_style
