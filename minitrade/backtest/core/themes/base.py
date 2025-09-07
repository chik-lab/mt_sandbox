"""
Base theme interface for minitrade plotting themes.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List


class BaseTheme(ABC):
    """Abstract base class for plotting themes."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Theme name identifier."""
        pass
    
    @property 
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable theme name."""
        pass
    
    @property
    @abstractmethod
    def background_fill_color(self) -> str:
        """Main background color for plots."""
        pass
        
    @property
    @abstractmethod 
    def border_fill_color(self) -> str:
        """Border/frame color for plots."""
        pass
        
    @property
    @abstractmethod
    def grid_line_color(self) -> str:
        """Grid line color."""
        pass
        
    @property
    @abstractmethod
    def axis_line_color(self) -> str:
        """Axis line color."""
        pass
        
    @property
    @abstractmethod
    def major_tick_line_color(self) -> str:
        """Major tick line color."""
        pass
        
    @property
    @abstractmethod
    def minor_tick_line_color(self) -> str:
        """Minor tick line color."""
        pass
        
    @property
    @abstractmethod
    def axis_label_text_color(self) -> str:
        """Axis label text color."""
        pass
        
    @property
    @abstractmethod
    def major_label_text_color(self) -> str:
        """Major label text color."""
        pass
        
    @property
    @abstractmethod
    def title_text_color(self) -> str:
        """Title text color."""
        pass
        
    @property
    @abstractmethod
    def legend_background_fill_color(self) -> str:
        """Legend background color."""
        pass
        
    @property
    @abstractmethod
    def legend_border_line_color(self) -> str:
        """Legend border color."""
        pass
        
    @property
    @abstractmethod
    def legend_label_text_color(self) -> str:
        """Legend text color."""
        pass
        
    @property
    @abstractmethod
    def color_palette(self) -> List[str]:
        """Color palette for data series."""
        pass
        
    @property
    @abstractmethod
    def separator_line_color(self) -> str:
        """Color for separator/divider lines."""
        pass
        
    @property
    @abstractmethod
    def nan_color(self) -> str:
        """Color for NaN/missing values in heatmaps."""
        pass
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Get theme configuration as a dictionary."""
        return {
            'name': self.name,
            'display_name': self.display_name,
            'background_fill_color': self.background_fill_color,
            'border_fill_color': self.border_fill_color,
            'grid_line_color': self.grid_line_color,
            'axis_line_color': self.axis_line_color,
            'major_tick_line_color': self.major_tick_line_color,
            'minor_tick_line_color': self.minor_tick_line_color,
            'axis_label_text_color': self.axis_label_text_color,
            'major_label_text_color': self.major_label_text_color,
            'title_text_color': self.title_text_color,
            'legend_background_fill_color': self.legend_background_fill_color,
            'legend_border_line_color': self.legend_border_line_color,
            'legend_label_text_color': self.legend_label_text_color,
            'color_palette': self.color_palette,
            'separator_line_color': self.separator_line_color,
            'nan_color': self.nan_color,
        }
    
    def apply_to_figure(self, fig) -> None:
        """Apply theme styling to a Bokeh figure."""
        # Grid styling
        if hasattr(fig, 'grid') and fig.grid:
            for grid in fig.grid:
                grid.grid_line_color = self.grid_line_color
                grid.grid_line_alpha = 0.3
        
        # Axis styling
        for axis in [fig.xaxis, fig.yaxis]:
            for ax in axis:
                ax.axis_line_color = self.axis_line_color
                ax.major_tick_line_color = self.major_tick_line_color
                ax.minor_tick_line_color = self.minor_tick_line_color
                ax.axis_label_text_color = self.axis_label_text_color
                ax.major_label_text_color = self.major_label_text_color
        
        # Legend styling
        if hasattr(fig, 'legend') and fig.legend:
            for legend in fig.legend:
                legend.background_fill_color = self.legend_background_fill_color
                legend.border_line_color = self.legend_border_line_color
                legend.label_text_color = self.legend_label_text_color
                legend.background_fill_alpha = 0.9
        
        # Background styling
        fig.background_fill_color = self.background_fill_color
        fig.border_fill_color = self.border_fill_color
        
        # Title styling
        if hasattr(fig, 'title') and fig.title:
            fig.title.text_color = self.title_text_color
