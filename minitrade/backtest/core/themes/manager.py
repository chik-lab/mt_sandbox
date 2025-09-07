"""
Theme management system for minitrade plotting.

This module handles theme registration, switching, and global state management.
"""

from typing import Dict, Optional, Union
from .base import BaseTheme
from .light import LightTheme
from .dark import DarkTheme
from .black import BlackTheme

# Global theme registry
_THEME_REGISTRY: Dict[str, BaseTheme] = {}
_CURRENT_THEME: Optional[BaseTheme] = None

# Register built-in themes
def _register_builtin_themes():
    """Register the built-in light, dark, and black themes."""
    global _THEME_REGISTRY
    
    light_theme = LightTheme()
    dark_theme = DarkTheme()
    black_theme = BlackTheme()
    
    _THEME_REGISTRY[light_theme.name] = light_theme
    _THEME_REGISTRY[dark_theme.name] = dark_theme
    _THEME_REGISTRY[black_theme.name] = black_theme

# Initialize built-in themes
_register_builtin_themes()

# Set default theme
_CURRENT_THEME = _THEME_REGISTRY['light']


def register_theme(theme: BaseTheme) -> None:
    """
    Register a custom theme.
    
    Args:
        theme: Theme instance implementing BaseTheme
    """
    global _THEME_REGISTRY
    _THEME_REGISTRY[theme.name] = theme


def get_available_themes() -> Dict[str, str]:
    """
    Get available themes as a dict of {name: display_name}.
    
    Returns:
        Dictionary mapping theme names to display names
    """
    return {name: theme.display_name for name, theme in _THEME_REGISTRY.items()}


def get_theme(name: str) -> Optional[BaseTheme]:
    """
    Get a theme by name.
    
    Args:
        name: Theme name
        
    Returns:
        Theme instance or None if not found
    """
    return _THEME_REGISTRY.get(name)


def set_theme(theme: Union[str, BaseTheme]) -> BaseTheme:
    """
    Set the current global theme.
    
    Args:
        theme: Theme name (str) or theme instance (BaseTheme)
        
    Returns:
        The activated theme instance
        
    Raises:
        ValueError: If theme name is not found
        TypeError: If theme is not a valid type
    """
    global _CURRENT_THEME
    
    if isinstance(theme, str):
        if theme not in _THEME_REGISTRY:
            available = list(_THEME_REGISTRY.keys())
            raise ValueError(f"Theme '{theme}' not found. Available themes: {available}")
        _CURRENT_THEME = _THEME_REGISTRY[theme]
    elif isinstance(theme, BaseTheme):
        _CURRENT_THEME = theme
        # Auto-register if not already registered
        if theme.name not in _THEME_REGISTRY:
            _THEME_REGISTRY[theme.name] = theme
    else:
        raise TypeError(f"Theme must be str or BaseTheme, got {type(theme)}")
    
    return _CURRENT_THEME


def get_current_theme() -> BaseTheme:
    """
    Get the current active theme.
    
    Returns:
        Current theme instance
    """
    global _CURRENT_THEME
    if _CURRENT_THEME is None:
        # Fallback to light theme
        _CURRENT_THEME = _THEME_REGISTRY['light']
    return _CURRENT_THEME


def set_light_theme() -> BaseTheme:
    """
    Convenience function to set light theme.
    
    Returns:
        Light theme instance
    """
    return set_theme('light')


def set_dark_theme() -> BaseTheme:
    """
    Convenience function to set dark theme.
    
    Returns:
        Dark theme instance
    """
    return set_theme('dark')


def reset_theme() -> BaseTheme:
    """
    Reset to default light theme.
    
    Returns:
        Light theme instance
    """
    return set_light_theme()


def create_theme_context(theme: Union[str, BaseTheme]):
    """
    Create a context manager for temporarily switching themes.
    
    Args:
        theme: Theme to use within the context
        
    Returns:
        Context manager
        
    Usage:
        with create_theme_context('dark'):
            # Code here uses dark theme
            bt.plot()
        # Theme reverts to previous after context
    """
    return ThemeContext(theme)


class ThemeContext:
    """Context manager for temporary theme switching."""
    
    def __init__(self, theme: Union[str, BaseTheme]):
        self.new_theme = theme
        self.previous_theme = None
    
    def __enter__(self):
        self.previous_theme = get_current_theme()
        set_theme(self.new_theme)
        return get_current_theme()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.previous_theme:
            set_theme(self.previous_theme)


def apply_theme_to_figure(fig, theme: Optional[BaseTheme] = None) -> None:
    """
    Apply theme styling to a Bokeh figure.
    
    Args:
        fig: Bokeh figure to style
        theme: Theme to apply (uses current theme if None)
    """
    if theme is None:
        theme = get_current_theme()
    
    theme.apply_to_figure(fig)


def get_theme_config(theme_name: Optional[str] = None) -> dict:
    """
    Get theme configuration as a dictionary.
    
    Args:
        theme_name: Name of theme to get config for (uses current if None)
        
    Returns:
        Theme configuration dictionary
    """
    if theme_name:
        theme = get_theme(theme_name)
        if theme is None:
            raise ValueError(f"Theme '{theme_name}' not found")
    else:
        theme = get_current_theme()
    
    return theme.get_config_dict()


# Convenience functions for backward compatibility
def is_dark_theme() -> bool:
    """Check if current theme is dark theme."""
    return get_current_theme().name == 'dark'


def is_light_theme() -> bool:
    """Check if current theme is light theme."""
    return get_current_theme().name == 'light'
