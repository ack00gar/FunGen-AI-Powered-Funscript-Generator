"""Device Control UI — decomposed from monolithic cp_device_control_ui.py.

DeviceControlMixin is composed from focused sub-mixins, one per device type.
"""
from application.gui_components.dc_panels._mixin import DeviceControlCoreMixin
from application.gui_components.dc_panels.axis_config import AxisConfigMixin
from application.gui_components.dc_panels.osr_panel import OSRPanelMixin
from application.gui_components.dc_panels.handy_panel import HandyPanelMixin
from application.gui_components.dc_panels.buttplug_panel import ButtplugPanelMixin
from application.gui_components.dc_panels.ossm_panel import OSSMPanelMixin
from application.gui_components.dc_panels.advanced_settings import AdvancedSettingsMixin


class DeviceControlMixin(
    DeviceControlCoreMixin,
    AxisConfigMixin,
    OSRPanelMixin,
    HandyPanelMixin,
    ButtplugPanelMixin,
    OSSMPanelMixin,
    AdvancedSettingsMixin,
):
    """Device Control tab UI mixin — composed from focused sub-mixins."""
    pass
