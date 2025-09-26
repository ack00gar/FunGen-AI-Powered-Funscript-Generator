"""
System Scaling Detection Utility
Provides cross-platform detection of system DPI and scaling factors.
"""
import sys
import logging
from typing import Tuple, Optional


def get_system_scaling_info() -> Tuple[float, float, str]:
    """
    Get system DPI and scaling information in a cross-platform way.
    
    Returns:
        Tuple of (scaling_factor, dpi, platform_name)
        - scaling_factor: multiplier (1.0 = 100% scaling)
        - dpi: dots per inch
        - platform_name: 'Windows', 'macOS', 'Linux', or 'Unknown'
    """
    logger = logging.getLogger(__name__)
    
    if sys.platform.startswith('win'):
        return _get_windows_scaling_info(logger)
    elif sys.platform == 'darwin':  # macOS
        return _get_macos_scaling_info(logger)
    elif sys.platform.startswith('linux'):  # Linux
        return _get_linux_scaling_info(logger)
    else:
        logger.warning(f"Unknown platform: {sys.platform}")
        return (1.0, 96.0, "Unknown")


def _get_windows_scaling_info(logger: logging.Logger) -> Tuple[float, float, str]:
    """Get scaling info for Windows systems."""
    try:
        import ctypes
        from ctypes import wintypes
        
        # Try to set process DPI aware for accurate results
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass
        
        # Try to get per-monitor DPI
        try:
            shcore = ctypes.windll.shcore
            user32 = ctypes.windll.user32
            
            # Get the monitor that contains the point (0, 0)
            monitor = user32.MonitorFromPoint(0, 0, 2)  # MONITOR_DEFAULTTONEAREST
            
            # Get DPI for that monitor
            dpi_x = ctypes.c_uint()
            dpi_y = ctypes.c_uint()
            result = shcore.GetDpiForMonitor(monitor, 0, ctypes.byref(dpi_x), ctypes.byref(dpi_y))
            
            if result == 0:  # S_OK
                dpi = dpi_x.value
                scaling_factor = dpi / 96.0  # 96 DPI is 100% scaling
                logger.info(f"Windows DPI detected: {dpi} (scaling factor: {scaling_factor:.2f})")
                return (scaling_factor, float(dpi), "Windows")
        except Exception as e:
            logger.debug(f"Failed to get per-monitor DPI on Windows: {e}")
        
        # Fallback to system DPI
        try:
            user32 = ctypes.windll.user32
            dpi = user32.GetDpiForSystem()
            scaling_factor = dpi / 96.0
            logger.info(f"Windows system DPI detected: {dpi} (scaling factor: {scaling_factor:.2f})")
            return (scaling_factor, float(dpi), "Windows")
        except Exception as e:
            logger.debug(f"Failed to get system DPI on Windows: {e}")
            
        # Final fallback to device caps
        try:
            user32 = ctypes.windll.user32
            hdc = user32.GetDC(0)
            dpi_x = user32.GetDeviceCaps(hdc, 88)  # LOGPIXELSX
            user32.ReleaseDC(0, hdc)
            scaling_factor = dpi_x / 96.0
            logger.info(f"Windows device caps DPI detected: {dpi_x} (scaling factor: {scaling_factor:.2f})")
            return (scaling_factor, float(dpi_x), "Windows")
        except Exception as e:
            logger.debug(f"Failed to get device caps DPI on Windows: {e}")
            
    except ImportError:
        logger.warning("ctypes not available for Windows DPI detection")
    except Exception as e:
        logger.warning(f"Error detecting Windows DPI: {e}")
    
    # Default fallback
    logger.info("Using default Windows DPI: 96 (scaling factor: 1.0)")
    return (1.0, 96.0, "Windows")


def _get_macos_scaling_info(logger: logging.Logger) -> Tuple[float, float, str]:
    """Get scaling info for macOS systems."""
    try:
        # macOS uses scale factors (1x, 2x, 3x) rather than traditional DPI
        # Try to get the backing scale factor from the main screen
        try:
            from AppKit import NSScreen
            main_screen = NSScreen.mainScreen()
            if main_screen:
                scale_factor = main_screen.backingScaleFactor()
                # macOS base DPI is 72, but we'll use 96 for consistency with other platforms
                dpi = scale_factor * 72
                # Convert to our standard scaling factor (based on 96 DPI)
                standard_scaling_factor = dpi / 96.0
                logger.info(f"macOS scale factor: {scale_factor} (standard scaling: {standard_scaling_factor:.2f})")
                return (standard_scaling_factor, dpi, "macOS")
        except ImportError:
            logger.debug("PyObjC not available for macOS scaling detection")
        except Exception as e:
            logger.debug(f"Failed to get macOS scale factor: {e}")
            
    except Exception as e:
        logger.warning(f"Error detecting macOS scaling: {e}")
    
    # Default fallback
    logger.info("Using default macOS scaling factor: 1.0")
    return (1.0, 72.0, "macOS")


def _get_linux_scaling_info(logger: logging.Logger) -> Tuple[float, float, str]:
    """Get scaling info for Linux systems."""
    try:
        # Check GDK scaling environment variable first
        import os
        gdk_scale = os.environ.get('GDK_SCALE')
        if gdk_scale:
            try:
                scale_factor = float(gdk_scale)
                # GDK_SCALE is a direct multiplier
                dpi = scale_factor * 96.0
                logger.info(f"Linux GDK_SCALE detected: {scale_factor} (DPI: {dpi})")
                return (scale_factor, dpi, "Linux")
            except ValueError:
                pass
        
        # Check QT_FONT_DPI environment variable
        qt_dpi = os.environ.get('QT_FONT_DPI')
        if qt_dpi:
            try:
                dpi = float(qt_dpi)
                scaling_factor = dpi / 96.0
                logger.info(f"Linux QT_FONT_DPI detected: {dpi} (scaling factor: {scaling_factor:.2f})")
                return (scaling_factor, dpi, "Linux")
            except ValueError:
                pass
                
        # Try to use GDK directly if available
        try:
            import gi
            gi.require_version('Gdk', '3.0')
            from gi.repository import Gdk
            
            screen = Gdk.Screen.get_default()
            if screen:
                dpi = screen.get_resolution()
                if dpi > 0:
                    scaling_factor = dpi / 96.0
                    logger.info(f"Linux GDK DPI detected: {dpi} (scaling factor: {scaling_factor:.2f})")
                    return (scaling_factor, dpi, "Linux")
                else:
                    # Fallback calculation from screen dimensions
                    width_mm = screen.get_width_mm()
                    height_mm = screen.get_height_mm()
                    width_px = screen.get_width()
                    height_px = screen.get_height()
                    
                    if width_mm > 0 and height_mm > 0:
                        dpi_x = width_px / (width_mm / 25.4)
                        dpi_y = height_px / (height_mm / 25.4)
                        dpi = (dpi_x + dpi_y) / 2
                        scaling_factor = dpi / 96.0
                        logger.info(f"Linux calculated DPI: {dpi:.1f} (scaling factor: {scaling_factor:.2f})")
                        return (scaling_factor, dpi, "Linux")
        except (ImportError, ValueError, Exception) as e:
            logger.debug(f"Failed to get GDK DPI on Linux: {e}")
        
        # Try to use xrandr command
        try:
            import subprocess
            import re
            
            result = subprocess.run(['xrandr'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                output = result.stdout
                
                # Find primary display with dimensions
                # This is a simplified approach - in practice, you'd want to parse more carefully
                mm_match = re.search(r'(\d+)mm x (\d+)mm', output)
                if mm_match:
                    width_mm, height_mm = int(mm_match.group(1)), int(mm_match.group(2))
                    
                    # Find resolution in the same line or nearby
                    lines = output.split('\n')
                    for line in lines:
                        if f"{width_mm}mm x {height_mm}mm" in line:
                            res_match = re.search(r'(\d+)x(\d+)\+', line)
                            if res_match:
                                width_px, height_px = int(res_match.group(1)), int(res_match.group(2))
                                
                                if width_mm > 0 and height_mm > 0:
                                    dpi_x = width_px / (width_mm / 25.4)
                                    dpi_y = height_px / (height_mm / 25.4)
                                    dpi = (dpi_x + dpi_y) / 2
                                    scaling_factor = dpi / 96.0
                                    logger.info(f"Linux xrandr DPI detected: {dpi:.1f} (scaling factor: {scaling_factor:.2f})")
                                    return (scaling_factor, dpi, "Linux")
        except Exception as e:
            logger.debug(f"Failed to get DPI from xrandr on Linux: {e}")
            
        # Try to use xrdb command
        try:
            import subprocess
            
            result = subprocess.run(['xrdb', '-query'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'Xft.dpi' in line:
                        dpi_str = line.split()[-1]
                        dpi = float(dpi_str)
                        scaling_factor = dpi / 96.0
                        logger.info(f"Linux Xft.dpi detected: {dpi} (scaling factor: {scaling_factor:.2f})")
                        return (scaling_factor, dpi, "Linux")
        except Exception as e:
            logger.debug(f"Failed to get DPI from xrdb on Linux: {e}")
            
    except Exception as e:
        logger.warning(f"Error detecting Linux scaling: {e}")
    
    # Default fallback
    logger.info("Using default Linux DPI: 96 (scaling factor: 1.0)")
    return (1.0, 96.0, "Linux")


def get_recommended_font_scale(scaling_factor: float) -> float:
    """
    Convert system scaling factor to recommended font scale.
    
    Args:
        scaling_factor: System scaling factor (1.0 = 100%)
        
    Returns:
        Recommended font scale value (matches constants.FONT_SCALE_VALUES)
    """
    # Convert system scaling to one of our predefined font scale values
    target_scale = scaling_factor
    
    # Round to nearest predefined value
    from config.constants import FONT_SCALE_VALUES
    closest_value = min(FONT_SCALE_VALUES, key=lambda x: abs(x - target_scale))
    
    return closest_value


def apply_system_scaling_to_settings(app_settings) -> bool:
    """
    Apply system scaling to application settings if enabled.
    
    Args:
        app_settings: Application settings manager
        
    Returns:
        True if scaling was applied, False otherwise
    """
    try:
        # Check if automatic scaling is enabled
        auto_scaling_enabled = app_settings.get("auto_system_scaling_enabled", True)
        if not auto_scaling_enabled:
            return False
            
        # Get system scaling info
        scaling_factor, dpi, platform = get_system_scaling_info()
        
        # Only apply if scaling is significantly different from 100%
        if abs(scaling_factor - 1.0) > 0.1:  # More than 10% difference
            # Convert to recommended font scale
            recommended_scale = get_recommended_font_scale(scaling_factor)
            
            # Apply to settings
            current_scale = app_settings.get("global_font_scale", 1.0)
            if abs(recommended_scale - current_scale) > 0.05:  # Only update if significantly different
                app_settings.set("global_font_scale", recommended_scale)
                logging.getLogger(__name__).info(
                    f"Applied system scaling: {scaling_factor:.2f}x ({dpi:.0f} DPI) -> "
                    f"font scale: {recommended_scale}"
                )
                return True
        else:
            logging.getLogger(__name__).debug(
                f"System scaling is close to 100% ({scaling_factor:.2f}x), not applying changes"
            )
                
    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to apply system scaling: {e}")
        
    return False