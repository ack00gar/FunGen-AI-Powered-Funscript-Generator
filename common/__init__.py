"""
Common utilities shared across FunGen modules.

Provides shared infrastructure for streamer and device_control modules.
"""

from .service_state import ServiceState
from .threaded_job import BackgroundJob
from .file_cleanup import cleanup_old_files
from .throttle import RateLimiter, SpeedLimiter
from .addon_status import (
    AddonStatus, register_status_provider, unregister_status_provider,
    get_all_statuses, get_status,
)

__version__ = "2.1.0"

__all__ = [
    'HTTPClientManager',
    'TempManager',
    'Result',
    'FunGenException',
    'ServiceState',
    'BackgroundJob',
    'cleanup_old_files',
    'RateLimiter',
    'SpeedLimiter',
    'AddonStatus',
    'register_status_provider',
    'unregister_status_provider',
    'get_all_statuses',
    'get_status',
]
