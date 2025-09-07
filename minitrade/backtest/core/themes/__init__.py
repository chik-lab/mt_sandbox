"""
Theming system for minitrade plotting.

This module provides a flexible theming system for customizing the appearance
of backtest plots, including support for dark and light themes.

Usage:
    from minitrade.backtest.core.themes import set_dark_theme, set_light_theme
    
    # Enable dark theme globally
    set_dark_theme()
    
    # Or for specific plots
    bt.plot(theme='dark')
"""

from .manager import (
    get_current_theme,
    set_theme,
    set_dark_theme, 
    set_light_theme,
    get_available_themes,
)

from .base import BaseTheme
from .dark import DarkTheme
from .light import LightTheme

__all__ = [
    'get_current_theme',
    'set_theme', 
    'set_dark_theme',
    'set_light_theme',
    'get_available_themes',
    'BaseTheme',
    'DarkTheme', 
    'LightTheme',
]
