"""
Integration utilities for connecting the theme system with _plotting.py

This module provides helper functions and utilities to integrate the theme system
with the existing plotting code without requiring major refactoring.
"""

from itertools import cycle
from typing import List, Tuple
from bokeh.transform import factor_cmap
from .manager import get_current_theme


def get_themed_colorgen():
    """Get a color generator using the current theme's palette."""
    theme = get_current_theme()
    yield from cycle(theme.color_palette)


def get_themed_bull_bear_colors() -> Tuple[str, str]:
    """Get bull/bear colors for the current theme."""
    theme = get_current_theme()
    config = theme.get_config_dict()
    return config.get('bear_color', '#FF4500'), config.get('bull_color')


def get_themed_factor_cmap(field_name: str, factors: List[str], colors: List[str] = None):
    """Create a factor color map using theme-appropriate colors."""
    if colors is None:
        bear_color, bull_color = get_themed_bull_bear_colors()
        colors = [bear_color, bull_color]
    
    return factor_cmap(field_name, colors, factors)


def apply_theme_to_new_figure(fig):
    """Apply current theme styling to a newly created figure."""
    theme = get_current_theme()
    theme.apply_to_figure(fig)
    return fig


def get_themed_separator_color() -> str:
    """Get separator/divider line color for current theme."""
    theme = get_current_theme()
    return theme.separator_line_color


def get_themed_nan_color() -> str:
    """Get NaN color for heatmaps in current theme."""
    theme = get_current_theme()
    return theme.nan_color


def get_themed_heatmap_palette() -> str:
    """Get heatmap color palette for current theme."""
    theme = get_current_theme()
    config = theme.get_config_dict()
    return config.get('heatmap_palette', 'Viridis256')


def get_themed_alpha_values() -> dict:
    """Get alpha transparency values for current theme."""
    theme = get_current_theme()
    config = theme.get_config_dict()
    return {
        'volume_alpha': config.get('volume_alpha', 0.5),
        'trade_line_alpha': config.get('trade_line_alpha', 0.8),
    }


def get_themed_lightness_factor() -> float:
    """Get lightness adjustment factor for current theme."""
    theme = get_current_theme()
    # Dark themes typically use higher lightness values for better contrast
    return 0.7 if theme.name == 'dark' else 0.94


def create_themed_figure_factory(original_figure_func):
    """
    Create a themed version of a figure creation function.
    
    Args:
        original_figure_func: Original figure creation function
        
    Returns:
        Wrapped function that applies theme styling
    """
    def themed_figure_func(*args, **kwargs):
        # Apply theme defaults
        theme = get_current_theme()
        
        theme_defaults = {
            'background_fill_color': theme.background_fill_color,
            'border_fill_color': theme.border_fill_color,
        }
        
        # Merge theme defaults with user kwargs (user takes precedence)
        final_kwargs = {**theme_defaults, **kwargs}
        
        # Create figure with theme defaults
        fig = original_figure_func(*args, **final_kwargs)
        
        # Apply additional theme styling
        apply_theme_to_new_figure(fig)
        
        return fig
    
    return themed_figure_func


# Convenience functions for specific styling needs
def get_themed_grid_color() -> str:
    """Get grid line color for current theme."""
    return get_current_theme().grid_line_color


def get_themed_axis_colors() -> dict:
    """Get axis-related colors for current theme."""
    theme = get_current_theme()
    return {
        'axis_line_color': theme.axis_line_color,
        'major_tick_line_color': theme.major_tick_line_color,
        'minor_tick_line_color': theme.minor_tick_line_color,
        'axis_label_text_color': theme.axis_label_text_color,
        'major_label_text_color': theme.major_label_text_color,
    }


def get_themed_legend_config() -> dict:
    """Get legend styling config for current theme."""
    theme = get_current_theme()
    return {
        'background_fill_color': theme.legend_background_fill_color,
        'border_line_color': theme.legend_border_line_color,
        'label_text_color': theme.legend_label_text_color,
        'background_fill_alpha': 0.9,
    }


def style_legend_with_theme(legend):
    """Apply theme styling to a legend object."""
    config = get_themed_legend_config()
    
    legend.background_fill_color = config['background_fill_color']
    legend.border_line_color = config['border_line_color'] 
    legend.label_text_color = config['label_text_color']
    legend.background_fill_alpha = config['background_fill_alpha']
