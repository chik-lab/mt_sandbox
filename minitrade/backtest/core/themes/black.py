"""
Pure black theme implementation for minitrade plotting.
"""

from .dark import DarkTheme


class BlackTheme(DarkTheme):
    """Pure black theme - extends dark theme with pure black backgrounds."""
    
    @property
    def name(self) -> str:
        return "black"
    
    @property
    def display_name(self) -> str:
        return "Pure Black Theme"
    
    @property
    def background_fill_color(self) -> str:
        return "#131722"
    
    @property
    def border_fill_color(self) -> str:
        return "#131722"
    
    @property
    def nan_color(self) -> str:
        return "#131722"
    
    @property
    def legend_background_fill_color(self) -> str:
        return "#1A1A1A"  # Slightly lighter for legend contrast
