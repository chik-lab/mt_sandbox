"""
Color utilities and palettes for minitrade themes.
"""

from typing import List


# Standard Bokeh palette (for light themes)
BOKEH_PALETTE = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'
]

# Dark theme optimized palette - brighter, more saturated colors
DARK_PALETTE = [
    '#FF6B6B',  # Bright red
    '#4ECDC4',  # Bright teal  
    '#45B7D1',  # Bright blue
    '#96CEB4',  # Mint green
    '#FFEAA7',  # Light yellow
    '#DDA0DD',  # Plum
    '#98D8C8',  # Aquamarine
    '#F7DC6F',  # Light gold
    '#BB8FCE',  # Light purple
    '#85C1E9',  # Light blue
]

# Light theme palette (enhanced Bokeh colors)
LIGHT_PALETTE = [
    '#2E86AB',  # Deep blue
    '#A23B72',  # Deep pink
    '#F18F01',  # Orange
    '#C73E1D',  # Red
    '#6A994E',  # Green
    '#7209B7',  # Purple
    '#F77F00',  # Bright orange
    '#FCBF49',  # Yellow
]

# Bull/Bear colors for different themes
BULL_BEAR_COLORS = {
    'light': {
        'bull': '#26a69a', 
        'bear': '#ef5350', 
    },
    'dark': {
        'bull': '#26a69a', 
        'bear': '#ef5350',
    }
}

# Neutral colors for different themes
NEUTRAL_COLORS = {
    'light': {
        'gray_light': '#CCCCCC',
        'gray_medium': '#888888', 
        'gray_dark': '#444444',
        'separator': '#E0E0E0',
    },
    'dark': {
        'gray_light': '#666666',
        'gray_medium': '#999999',
        'gray_dark': '#CCCCCC', 
        'separator': '#404040',
    }
}


def get_color_palette(theme_type: str) -> List[str]:
    """Get color palette for specified theme type."""
    if theme_type.lower() == 'dark':
        return DARK_PALETTE
    else:
        return LIGHT_PALETTE


def get_bull_bear_colors(theme_type: str) -> dict:
    """Get bull/bear colors for specified theme type."""
    return BULL_BEAR_COLORS.get(theme_type.lower(), BULL_BEAR_COLORS['light'])


def get_neutral_colors(theme_type: str) -> dict:
    """Get neutral colors for specified theme type."""
    return NEUTRAL_COLORS.get(theme_type.lower(), NEUTRAL_COLORS['light'])


def lighten_color(color: str, factor: float = 0.2) -> str:
    """Lighten a hex color by a given factor."""
    # Remove '#' if present
    color = color.lstrip('#')
    
    # Convert to RGB
    rgb = [int(color[i:i+2], 16) for i in (0, 2, 4)]
    
    # Lighten each component
    rgb = [min(255, int(c + (255 - c) * factor)) for c in rgb]
    
    # Convert back to hex
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def darken_color(color: str, factor: float = 0.2) -> str:
    """Darken a hex color by a given factor."""
    # Remove '#' if present
    color = color.lstrip('#')
    
    # Convert to RGB
    rgb = [int(color[i:i+2], 16) for i in (0, 2, 4)]
    
    # Darken each component
    rgb = [max(0, int(c * (1 - factor))) for c in rgb]
    
    # Convert back to hex
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
