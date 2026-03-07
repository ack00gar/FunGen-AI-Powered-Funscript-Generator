"""Device Control tab UI mixin — re-exports from decomposed subpackage.

The actual implementation lives in application/gui_components/device_control/.
This file preserves the original import path for backwards compatibility.
"""
from application.gui_components.dc_panels import DeviceControlMixin

__all__ = ['DeviceControlMixin']
