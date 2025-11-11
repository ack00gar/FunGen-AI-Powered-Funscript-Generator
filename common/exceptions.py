"""
Common exceptions and error handling.

Provides standardized exception types for better error handling
across all FunGen modules.
"""


class FunGenException(Exception):
    """Base exception for all FunGen errors."""
    pass


class ConnectionError(FunGenException):
    """Failed to connect to external service (XBVR, Stash, device, etc.)."""
    pass


class DeviceError(FunGenException):
    """Device operation failed."""
    pass


class VideoSourceError(FunGenException):
    """Video source operation failed."""
    pass


class TranscodingError(FunGenException):
    """Video transcoding failed."""
    pass


class SyncError(FunGenException):
    """Synchronization error."""
    pass
