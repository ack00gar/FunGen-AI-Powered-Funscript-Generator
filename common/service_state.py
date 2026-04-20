"""Shared lifecycle state enum for any long-running component."""

from enum import Enum


class ServiceState(Enum):
    """
    Single coherent lifecycle for things like the streamer HTTP server, the
    subtitle pipeline, the device-control auto-reconnect loop. Used so the UI
    can render every component's status with the same vocabulary.

    Transitions:
        STOPPED -> STARTING -> RUNNING
        RUNNING -> STOPPING -> STOPPED
        any    -> ERROR (terminal until reset to STOPPED)
    """
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"
