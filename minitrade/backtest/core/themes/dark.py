"""
Dark theme implementation for minitrade plotting.
"""

from typing import List
from .base import BaseTheme
from .colors import get_color_palette, get_bull_bear_colors, get_neutral_colors


class DarkTheme(BaseTheme):
    """Dark theme for plotting with light text on dark backgrounds."""
    
    @property
    def name(self) -> str:
        return "dark"
    
    @property
    def display_name(self) -> str:
        return "Dark Theme"
    
    @property
    def background_fill_color(self) -> str:
        return "#000000"
    
    @property
    def border_fill_color(self) -> str:
        return "#000000"
    
    @property
    def grid_line_color(self) -> str:
        return "#404040"
    
    @property
    def axis_line_color(self) -> str:
        return "#666666"
    
    @property
    def major_tick_line_color(self) -> str:
        return "#666666"
    
    @property
    def minor_tick_line_color(self) -> str:
        return "#404040"
    
    @property
    def axis_label_text_color(self) -> str:
        return "#CCCCCC"
    
    @property
    def major_label_text_color(self) -> str:
        return "#CCCCCC"
    
    @property
    def title_text_color(self) -> str:
        return "#FFFFFF"
    
    @property
    def legend_background_fill_color(self) -> str:
        return "#404040"
    
    @property
    def legend_border_line_color(self) -> str:
        return "#666666"
    
    @property
    def legend_label_text_color(self) -> str:
        return "#CCCCCC"
    
    @property
    def color_palette(self) -> List[str]:
        return get_color_palette("dark")
    
    @property
    def separator_line_color(self) -> str:
        return "#404040"
    
    @property
    def nan_color(self) -> str:
        return "#000000"
    
    @property
    def bull_color(self) -> str:
        """Bull (positive) color for dark theme."""
        return get_bull_bear_colors("dark")["bull"]
    
    @property
    def bear_color(self) -> str:
        """Bear (negative) color for dark theme.""" 
        return get_bull_bear_colors("dark")["bear"]
    
    @property
    def neutral_colors(self) -> dict:
        """Neutral color palette for dark theme."""
        return get_neutral_colors("dark")
    
    @property
    def heatmap_palette(self) -> str:
        """Color palette for heatmaps in dark theme."""
        return "Plasma256"  # Better for dark backgrounds
    
    @property
    def volume_alpha(self) -> float:
        """Alpha transparency for volume bars."""
        return 0.6  # Slightly more transparent for dark theme
    
    @property
    def trade_line_alpha(self) -> float:
        """Alpha transparency for trade lines."""
        return 0.9  # High visibility on dark background
    
    def get_lightness_adjusted_color(self, base_color: str, lightness: float = 0.7) -> str:
        """Get a lightness-adjusted version of a color for dark theme."""
        from ..._plotting import lightness as adjust_lightness
        from bokeh.colors import RGB
        
        # Convert hex to RGB if needed
        if isinstance(base_color, str) and base_color.startswith('#'):
            r = int(base_color[1:3], 16)
            g = int(base_color[3:5], 16) 
            b = int(base_color[5:7], 16)
            color_obj = RGB(r, g, b)
        else:
            # Assume it's already a Bokeh color object
            color_obj = base_color
            
        return adjust_lightness(color_obj, lightness)
    
    def get_config_dict(self) -> dict:
        """Extended config dict with dark theme specific properties."""
        config = super().get_config_dict()
        config.update({
            'bull_color': self.bull_color,
            'bear_color': self.bear_color,
            'neutral_colors': self.neutral_colors,
            'heatmap_palette': self.heatmap_palette,
            'volume_alpha': self.volume_alpha,
            'trade_line_alpha': self.trade_line_alpha,
        })
        return config
