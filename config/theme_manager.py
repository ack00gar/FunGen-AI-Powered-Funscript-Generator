"""
Theme manager for handling color themes.
Provides easy theme switching and management.
"""

from config.constants_colors import DarkTheme, CurrentTheme


class ThemeManager:
    """Manages color themes for the application."""
    
    def __init__(self):
        self._current_theme = DarkTheme
        self._available_themes = {
            'dark': DarkTheme,
        }
    
    @property
    def current_theme(self):
        """Get the current theme."""
        return self._current_theme
    
    @property
    def available_themes(self):
        """Get list of available theme names."""
        return list(self._available_themes.keys())
    
    def set_theme(self, theme_name: str) -> bool:
        """
        Set the current theme by name.
        
        Args:
            theme_name: Name of the theme to set
            
        Returns:
            True if theme was set successfully, False otherwise
        """
        if theme_name not in self._available_themes:
            return False
        
        self._current_theme = self._available_themes[theme_name]
        
        # Update the global CurrentTheme reference
        import config.constants_colors
        config.constants_colors.CurrentTheme = self._current_theme
        
        # Update all color element groups
        self._update_color_groups()
        
        return True
    
    def get_theme(self, theme_name: str):
        """
        Get a theme by name.
        
        Args:
            theme_name: Name of the theme to get
            
        Returns:
            Theme class or None if not found
        """
        return self._available_themes.get(theme_name)
    
    def add_theme(self, theme_name: str, theme_class):
        """
        Add a new theme.
        
        Args:
            theme_name: Name for the new theme
            theme_class: Theme class to add
        """
        self._available_themes[theme_name] = theme_class
    
    def _update_color_groups(self):
        """Update all color element groups to use the new theme."""
        # This would be called when switching themes
        # For now, the color groups automatically use CurrentTheme
        # so they'll update automatically
        pass


# Global theme manager instance
theme_manager = ThemeManager()


def set_theme(theme_name: str) -> bool:
    """
    Convenience function to set the current theme.
    
    Args:
        theme_name: Name of the theme to set
        
    Returns:
        True if theme was set successfully, False otherwise
    """
    return theme_manager.set_theme(theme_name)


def get_current_theme():
    """Get the current theme."""
    return theme_manager.current_theme


def get_available_themes():
    """Get list of available theme names."""
    return theme_manager.available_themes 