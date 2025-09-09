import os
import re
import sys
import warnings
from colorsys import hls_to_rgb, rgb_to_hls
from functools import partial
from itertools import combinations, cycle
from typing import Callable, List, Union, Dict

import numpy as np
import pandas as pd
from bokeh.colors import RGB
from bokeh.colors.named import forestgreen as BULL_COLOR
from bokeh.colors.named import orangered as BEAR_COLOR
from bokeh.models import (
    BoxAnnotation,
    ColumnDataSource,
    CrosshairTool,
    CustomJS,
    DatetimeTickFormatter,
    HoverTool,
    Label,
    Legend,
    LinearColorMapper,
    NumeralTickFormatter,
    Range1d,
    Span,
)
from bokeh.plotting import figure as _figure


def _windos_safe_filename(filename):
    """
    Create a safe filename for all platforms by removing/replacing problematic characters.
    """
    # Remove or replace problematic characters
    safe_filename = re.sub(r"[<>:\"/\\|?*]", "_", filename)
    safe_filename = re.sub(r"[^a-zA-Z0-9,_\-\.]", "_", safe_filename)

    # Ensure it's not empty
    if not safe_filename.strip():
        safe_filename = "strategy_backtest"

    # Limit length to avoid filesystem issues
    if len(safe_filename) > 100:
        safe_filename = safe_filename[:100]

    return safe_filename


try:
    from bokeh.models import CustomJSTickFormatter
except ImportError:
    from bokeh.models import FuncTickFormatter as CustomJSTickFormatter

from bokeh.io import output_file, output_notebook, show
from bokeh.io.state import curstate
from bokeh.layouts import gridplot
from bokeh.palettes import Bokeh
from bokeh.transform import factor_cmap

from ._util import _as_list, _data_period

# Theme system imports
try:
    from .themes import get_current_theme, set_theme
    from .themes.plotting_integration import (
        get_themed_colorgen,
        get_themed_bull_bear_colors,
        get_themed_factor_cmap,
        apply_theme_to_new_figure,
        get_themed_heatmap_palette,
        get_themed_nan_color,
        style_legend_with_theme,
    )
    THEMES_AVAILABLE = True
except ImportError:
    # Fallback if themes not available
    THEMES_AVAILABLE = False

with open(
    os.path.join(os.path.dirname(__file__), "autoscale_cb.js"), encoding="utf-8"
) as _f:
    _AUTOSCALE_JS_CALLBACK = _f.read()

# Detect Jupyter Notebook, works in Jupyter and VS Code
IS_JUPYTER_NOTEBOOK = "ipykernel" in sys.modules

if IS_JUPYTER_NOTEBOOK:
    output_notebook(hide_banner=True)


def set_bokeh_output(notebook=False):
    """
    Set Bokeh to output either to a file or Jupyter notebook.
    By default, Bokeh outputs to notebook if running from within
    notebook was detected.
    """
    global IS_JUPYTER_NOTEBOOK
    IS_JUPYTER_NOTEBOOK = notebook


def get_output_directory():
    """
    Get the current working directory where HTML files will be saved.
    """
    import os

    return os.getcwd()


def set_output_directory(directory):
    """
    Set the output directory for HTML files.

    Args:
        directory (str): Path to the directory where HTML files should be saved
    """
    import os

    if os.path.exists(directory) and os.path.isdir(directory):
        os.chdir(directory)
        print(f"Output directory set to: {os.getcwd()}")
    else:
        print(f"Warning: Directory {directory} does not exist or is not a directory")
        print(f"Current working directory remains: {os.getcwd()}")


def _bokeh_reset(filename=None):
    curstate().reset()
    if filename and filename != "":  # Allow empty string to disable file output
        if not filename.endswith(".html"):
            filename += ".html"

        # Remove existing file to prevent "already exists, will be overwritten" warning
        try:
            import os

            if os.path.exists(filename):
                os.remove(filename)
        except Exception:
            pass  # Ignore if we can't remove it

        # Use the filename (strategy name) as the title, but clean it up
        title = filename.replace(".html", "").replace("_", " ")
        output_file(filename, title=title)
    elif IS_JUPYTER_NOTEBOOK:
        curstate().output_notebook()

    return filename


def _inject_theme_css(filename):
    """Inject theme-aware CSS into the HTML file to fix body background and mobile responsiveness."""
    if not filename or not os.path.exists(filename):
        return
        
    # Get theme colors if available, otherwise use defaults
    if THEMES_AVAILABLE:
        theme = get_current_theme()
        bg_color = theme.background_fill_color
    else:
        bg_color = "#ffffff"  # Default white background
    
    # CSS to apply theme background and mobile responsive text
    css_style = f"""
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{
            background-color: {bg_color} !important;
            margin: 0 !important;
            padding: 0 !important;
        }}
        .bk-root {{
            background-color: {bg_color} !important;
        }}
        
        /* Base font size scaling for mobile - more aggressive approach */
        html {{
            font-size: 16px;
        }}
        
        /* Mobile viewport and touch optimization */
        @media screen and (max-width: 768px) {{
            body {{
                overflow-x: auto !important;
                overflow-y: auto !important;
                -webkit-overflow-scrolling: touch !important;
                touch-action: pan-x pan-y !important;
            }}
            
            .bk-root {{
                overflow: visible !important;
                touch-action: pan-x pan-y !important;
            }}
            
            /* Ensure scrollable content */
            .bk-canvas-wrapper {{
                touch-action: pan-x pan-y !important;
            }}
        }}
        
        @media screen and (max-width: 480px) {{
            body {{
                overflow-x: auto !important;
                overflow-y: auto !important;
                -webkit-overflow-scrolling: touch !important;
                touch-action: pan-x pan-y !important;
            }}
        }}
        
        /* Mobile responsive text scaling - comprehensive Bokeh targeting */
        @media screen and (max-width: 768px) {{
            /* Enlarge axis labels and tick labels - multiple selectors for Bokeh versions */
            .bk-axis-label, 
            .bk-axis .bk-axis-label,
            div[data-bk-view] .bk-axis-label,
            .bk div.bk-axis-label {{
                font-size: 16px !important;
            }}
            
            .bk-tick-label,
            .bk-axis .bk-tick-label,
            div[data-bk-view] .bk-tick-label,
            .bk div.bk-tick-label,
            text.bk-axis-label,
            text.bk-tick-label {{
                font-size: 14px !important;
            }}
            
            /* Enlarge plot titles - multiple selectors */
            .bk-title,
            .bk-plot-layout .bk-title,
            div[data-bk-view] .bk-title,
            .bk div.bk-title,
            .bk-plot-wrapper .bk-title,
            h1.bk-title {{
                font-size: 18px !important;
                font-weight: bold !important;
            }}
            
            /* Enlarge tooltip text */
            .bk-tooltip,
            .bk-tooltip-content,
            div.bk-tooltip,
            div.bk-tooltip-content {{
                font-size: 14px !important;
            }}
            
            /* Enlarge legend text */
            .bk-legend-label,
            .bk-legend .bk-legend-label,
            div.bk-legend-label {{
                font-size: 14px !important;
            }}
            
            /* Make toolbar buttons larger for touch */
            .bk-toolbar-button,
            .bk-btn,
            .bk-toolbar .bk-toolbar-button {{
                width: 40px !important;
                height: 40px !important;
                font-size: 14px !important;
            }}
            
            /* Target all text elements in Bokeh plots - SVG and HTML */
            .bk-root text,
            .bk text,
            div[data-bk-view] text,
            svg text,
            .bk-root svg text,
            .bk svg text {{
                font-size: 14px !important;
            }}
            
            /* Ensure proper touch scrolling */
            .bk-plot-wrapper,
            .bk-canvas-wrapper {{
                touch-action: pan-x pan-y !important;
                overflow: visible !important;
            }}
        }}
        
        /* Extra small mobile devices */
        @media screen and (max-width: 480px) {{
            .bk-axis-label, 
            .bk-axis .bk-axis-label,
            div[data-bk-view] .bk-axis-label,
            .bk div.bk-axis-label {{
                font-size: 18px !important;
            }}
            
            .bk-tick-label,
            .bk-axis .bk-tick-label,
            div[data-bk-view] .bk-tick-label,
            .bk div.bk-tick-label,
            text.bk-axis-label,
            text.bk-tick-label {{
                font-size: 16px !important;
            }}
            
            .bk-title,
            .bk-plot-layout .bk-title,
            div[data-bk-view] .bk-title,
            .bk div.bk-title,
            .bk-plot-wrapper .bk-title,
            h1.bk-title {{
                font-size: 20px !important;
                font-weight: bold !important;
            }}
            
            .bk-tooltip,
            .bk-tooltip-content,
            div.bk-tooltip,
            div.bk-tooltip-content {{
                font-size: 16px !important;
            }}
            
            .bk-legend-label,
            .bk-legend .bk-legend-label,
            div.bk-legend-label {{
                font-size: 16px !important;
            }}
            
            .bk-toolbar-button,
            .bk-btn,
            .bk-toolbar .bk-toolbar-button {{
                width: 45px !important;
                height: 45px !important;
                font-size: 16px !important;
            }}
            
            /* Target all text elements in Bokeh plots - SVG and HTML */
            .bk-root text,
            .bk text,
            div[data-bk-view] text,
            svg text,
            .bk-root svg text,
            .bk svg text {{
                font-size: 16px !important;
            }}
            
            /* Ensure proper touch scrolling for small devices */
            .bk-plot-wrapper,
            .bk-canvas-wrapper {{
                touch-action: pan-x pan-y !important;
                overflow: visible !important;
            }}
            
            /* Prevent touch conflicts */
            .bk-root .bk-canvas-wrapper text,
            .bk-root .bk-plot-wrapper text,
            .bk-plot-layout text,
            .bk-plot text {{
                font-size: 16px !important;
                pointer-events: none !important;
            }}
        }}
    </style>
    """
    
    try:
        # Read the HTML file
        with open(filename, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Inject the CSS right after the <head> tag
        if '<head>' in html_content:
            html_content = html_content.replace('<head>', f'<head>\n{css_style}')
            
            # Write back the modified HTML
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(html_content)
    except Exception:
        pass  # Silently fail if we can't modify the HTML


def _get_bokeh_browser_arg(open_browser: bool):
    """Return a browser argument suitable for bokeh.io.show on this platform.

    - When open_browser is False, return "none" to suppress opening a browser.
    - On WSL, prefer "windows-default" so it opens in the Windows default browser.
    - Otherwise, return None to let Bokeh use system defaults without forcing a specific browser.
    """
    if not open_browser:
        return "none"

    # Detect WSL environments
    try:
        import os
        import platform

        if os.environ.get("WSL_DISTRO_NAME"):
            return "windows-default"
        release = platform.release().lower()
        if "microsoft" in release:
            return "windows-default"
        # Fallback: check /proc/version
        try:
            with open("/proc/version", "r", encoding="utf-8") as _v:
                if "microsoft" in _v.read().lower():
                    return "windows-default"
        except Exception:
            pass
    except Exception:
        pass

    return None


def _open_html_external(file_path: str) -> None:
    """Open the given HTML file in a suitable browser, with special handling for WSL.

    This avoids relying on Python's webbrowser in environments where no Linux GUI
    browser is available (e.g., WSL without X/Wayland).
    """
    try:
        import os
        import platform
        import shutil
        import subprocess

        abs_path = os.path.abspath(file_path)

        # Detect WSL
        is_wsl = False
        if os.environ.get("WSL_DISTRO_NAME"):
            is_wsl = True
        else:
            try:
                with open("/proc/version", "r", encoding="utf-8") as _v:
                    is_wsl = "microsoft" in _v.read().lower()
            except Exception:
                is_wsl = "microsoft" in platform.release().lower()

        if is_wsl:
            # Prefer wslview if available
            if shutil.which("wslview"):
                subprocess.Popen(["wslview", abs_path])
                return

            # Fallback to Windows PowerShell Start-Process
            # Convert /mnt/<drive>/path -> <Drive>:\\path
            win_path = abs_path
            try:
                import re as _re

                m = _re.match(r"^/mnt/([a-zA-Z])/(.*)", abs_path)
                if m:
                    drive = m.group(1).upper() + ":"
                    tail = m.group(2).replace("/", "\\")
                    win_path = f"{drive}\\{tail}"
            except Exception:
                pass

            # Only attempt if the file actually exists
            if not os.path.exists(abs_path):
                return

            pwsh = shutil.which("powershell.exe") or "powershell.exe"
            subprocess.Popen(
                [pwsh, "-NoProfile", "-Command", f'Start-Process "{win_path}"']
            )
            return

        # Non-WSL: use webbrowser
        import webbrowser

        webbrowser.open_new_tab("file://" + abs_path)
    except Exception:
        pass


def colorgen():
    if THEMES_AVAILABLE:
        yield from get_themed_colorgen()
    else:
        yield from cycle(Bokeh[8])


def lightness(color, lightness=0.94):
    rgb = np.array([color.r, color.g, color.b]) / 255
    h, _, s = rgb_to_hls(*rgb)
    rgb = np.array(hls_to_rgb(h, lightness, s)) * 255.0
    return RGB(*rgb)


_MAX_CANDLES = 10_000


def _maybe_resample_data(
    resample_rule, data, baseline, indicators, equity_data, trades
):
    if isinstance(resample_rule, str):
        freq = resample_rule
    else:
        if resample_rule is False or len(baseline) <= _MAX_CANDLES:
            return data, baseline, indicators, equity_data, trades

        freq_minutes = pd.Series(
            {
                "1T": 1,
                "5T": 5,
                "10T": 10,
                "15T": 15,
                "30T": 30,
                "1H": 60,
                "2H": 60 * 2,
                "4H": 60 * 4,
                "8H": 60 * 8,
                "1D": 60 * 24,
                "1W": 60 * 24 * 7,
                "1M": np.inf,
            }
        )
        timespan = baseline.index[-1] - baseline.index[0]
        require_minutes = (timespan / _MAX_CANDLES).total_seconds() // 60
        freq = freq_minutes.where(freq_minutes >= require_minutes).first_valid_index()
        warnings.warn(
            f"Data contains too many candlesticks to plot; downsampling to {freq!r}. "
            "See `Backtest.plot(resample=...)`"
        )

    from .lib import _EQUITY_AGG, OHLCV_AGG, TRADES_AGG

    data = data.ta.apply(
        lambda s: s.resample(freq, label="right").agg(OHLCV_AGG)
    ).dropna()

    baseline = baseline.resample(freq, label="right").agg(OHLCV_AGG).dropna()

    indicators = [
        i.resample(freq, label="right").mean().dropna().reindex(baseline.index)
        for i in indicators
    ]
    assert not indicators or indicators[0].index.equals(baseline.index)

    column_agg = {
        ticker: _EQUITY_AGG[ticker] if ticker in _EQUITY_AGG else "last"
        for ticker in equity_data.columns
    }
    equity_data = (
        equity_data.resample(freq, label="right").agg(column_agg).dropna(how="all")
    )
    assert equity_data.index.equals(baseline.index)

    def _weighted_returns(s, trades=trades):
        df = trades.loc[s.index]
        return ((df["Size"].abs() * df["Gross%"]) / df["Size"].abs().sum()).sum()

    def _group_trades(column):
        def f(s, new_index=pd.Index(baseline.index.view(int)), bars=trades[column]):
            if s.size:
                # Via int64 because on pandas recently broken datetime
                mean_time = int(bars.loc[s.index].astype(int).mean())
                new_bar_idx = new_index.get_indexer([mean_time], method="nearest")[0]
                return new_bar_idx

        return f

    if len(trades):  # Avoid pandas "resampling on Int64 index" error
        trades = (
            trades.assign(count=1)
            .resample(freq, on="ExitTime", label="right")
            .agg(
                {
                    **TRADES_AGG,
                    "Gross%": _weighted_returns,
                    "count": "sum",
                    "EntryBar": _group_trades("EntryTime"),
                    "ExitBar": _group_trades("ExitTime"),
                }
            )
            .dropna()
        )

    return data, baseline, indicators, equity_data, trades


def _extract_traded_tickers(trades):
    """Extract tickers that have trade records."""
    traded_tickers = set()
    try:
        if isinstance(trades, pd.DataFrame) and "Ticker" in trades.columns:
            traded_tickers = set(trades["Ticker"].dropna().unique().tolist())
    except Exception:
        pass
    return traded_tickers


def _filter_data_to_traded_tickers(data, traded_tickers):
    """Filter universe data to traded tickers if available."""
    try:
        if traded_tickers and isinstance(data.columns, pd.MultiIndex):
            tickers_in_data = [t for t in data.columns.levels[0] if t in traded_tickers]
            if tickers_in_data:
                return data.loc[:, (tickers_in_data, slice(None))]
    except Exception:
        pass
    return data


def _determine_tickers_to_plot(data, traded_tickers):
    """Determine tickers to plot for universe and for gating decisions."""
    try:
        if isinstance(data.columns, pd.MultiIndex):
            available_tickers = list(data.columns.levels[0])
            return [
                t
                for t in available_tickers
                if (not traded_tickers or t in traded_tickers)
            ]
    except Exception:
        return []


def _process_baseline_data(baseline):
    """Process baseline data to handle multi-symbol cases."""
    from .lib import OHLCV_AGG

    if isinstance(baseline.columns, pd.MultiIndex):
        # Store the full multi-symbol data for later use
        baseline_multi = baseline
        # Create baseline for the main chart (use first symbol, simplified columns)
        first_symbol = baseline.columns.get_level_values(0)[0]
        baseline_main = baseline.loc[:, (first_symbol, slice(None))]
        baseline_main.columns = baseline_main.columns.get_level_values(1)
        if "Volume" not in baseline_main:
            baseline_main["Volume"] = 0
        baseline_main = baseline_main[list(OHLCV_AGG.keys())].copy(deep=False)
        return baseline_main, baseline_multi
    else:
        # Force single-symbol into multi-symbol shape so plotting always treats it as multi
        single = baseline.copy(deep=False)
        if "Volume" not in single:
            single["Volume"] = 0
        single = single[list(OHLCV_AGG.keys())].copy(deep=False)
        # Wrap into a MultiIndex with a synthetic ticker name
        ticker_name = getattr(single, "attrs", {}).get("symbol") or "Asset"
        baseline_multi = pd.concat({ticker_name: single}, axis=1)
        # For main chart, keep simplified single-symbol columns
        baseline_main = single
        return baseline_main, baseline_multi


def _prepare_plot_data(results, data, baseline, indicators, resample):
    """Prepare and process all data for plotting."""
    equity_data = results["_equity_curve"].copy()
    trades = results["_trades"]

    # Extract phase data if available
    phase_data = results.get("_phase_data")

    # Extract traded tickers
    traded_tickers = _extract_traded_tickers(trades)

    # Filter data to traded tickers
    data = _filter_data_to_traded_tickers(data, traded_tickers)

    # Determine tickers to plot
    tickers_to_plot = _determine_tickers_to_plot(data, traded_tickers)

    # Process baseline data
    baseline, baseline_multi = _process_baseline_data(baseline)

    # Handle resampling if needed
    is_datetime_index = isinstance(baseline.index, pd.DatetimeIndex)
    if is_datetime_index:
        data, baseline, indicators, equity_data, trades = _maybe_resample_data(
            resample, data, baseline, indicators, equity_data, trades
        )

    # Prepare index data
    baseline.index.name = None  # Provides source name @index
    baseline["datetime"] = baseline.index  # Save original, maybe datetime index
    baseline = baseline.reset_index(drop=True)
    equity_data = equity_data.reset_index(drop=True)
    index = baseline.index

    return {
        "data": data,
        "baseline": baseline,
        "baseline_multi": baseline_multi,
        "indicators": indicators,
        "equity_data": equity_data,
        "trades": trades,
        "phase_data": phase_data,
        "traded_tickers": traded_tickers,
        "tickers_to_plot": tickers_to_plot,
        "index": index,
        "is_datetime_index": is_datetime_index,
    }


def plot(
    *,
    results: pd.Series,
    data: pd.DataFrame,
    baseline: pd.DataFrame,
    indicators: List[Union[pd.DataFrame, pd.Series]],
    filename="",
    plot_width=None,
    plot_equity=True,
    plot_return=False,
    plot_pl=True,
    plot_volume=True,
    plot_drawdown=False,
    plot_trades=True,
    smooth_equity=False,
    superimpose=False,
    resample=True,
    reverse_indicators=True,
    show_legend=True,
    open_browser=True,
    plot_allocation=False,
    relative_allocation=True,
    plot_indicator=True,
    theme=None,
):
    """
    Like much of GUI code everywhere, this is a mess.
    """
    # Set theme if provided
    if theme is not None and THEMES_AVAILABLE:
        set_theme(theme)
    
    # We need to reset global Bokeh state, otherwise subsequent runs of
    # plot() contain some previous run's cruft data (was noticed when
    # TestPlot.test_file_size() test was failing).
    # Use provided filename if available, otherwise auto-generate only in Jupyter notebook
    if filename is None or filename == "":
        if IS_JUPYTER_NOTEBOOK:
            filename = ""  # Empty string to disable file output in notebook
        else:
            # Try to construct the expected filename based on the simulation pattern
            # The expected pattern is: /mnt/e/code/algo/output/simulations/YYYYMMDD_HHMMSS_plot.html
            import datetime
            import glob
            import os

            # Try to find the simulations directory
            current_dir = os.getcwd()

            # Look for simulations directory in common locations
            possible_paths = [
                os.path.join(current_dir, "output", "simulations"),
                os.path.join(current_dir, "simulations"),
                "/mnt/e/code/algo/output/simulations",  # Expected path from logs
                os.path.join(os.path.dirname(current_dir), "output", "simulations"),
            ]

            simulations_dir = None
            for path in possible_paths:
                if os.path.exists(path):
                    simulations_dir = path
                    break

            if simulations_dir:
                # Look for existing result files to match their timestamp
                result_files = glob.glob(os.path.join(simulations_dir, "*_result.log"))
                if result_files:
                    # Sort by modification time to get the most recent
                    result_files.sort(key=os.path.getmtime, reverse=True)
                    most_recent = result_files[0]

                    # Extract timestamp from result filename
                    basename = os.path.basename(most_recent)
                    if basename.endswith("_result.log"):
                        timestamp = basename.replace("_result.log", "")

                        # Construct plot filename with same timestamp
                        filename = os.path.join(
                            simulations_dir, f"{timestamp}_plot.html"
                        )
                    else:
                        filename = ""
                else:
                    # Fallback to current timestamp
                    now = datetime.datetime.now()
                    timestamp = now.strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(simulations_dir, f"{timestamp}_plot.html")
            else:
                filename = ""

    # Prepare output filename (without initializing Bokeh output yet)
    if filename and not str(filename).endswith(".html"):
        processed_filename = f"{filename}.html"
    else:
        processed_filename = filename or ""

    # Get theme-aware colors
    if THEMES_AVAILABLE:
        bear_color, bull_color = get_themed_bull_bear_colors()
        COLORS = [bear_color, bull_color]
    else:
        COLORS = [BEAR_COLOR, BULL_COLOR]
    BAR_WIDTH = 0.8

    assert baseline.index.equals(results["_equity_curve"].index)

    # Prepare all plot data
    plot_data = _prepare_plot_data(results, data, baseline, indicators, resample)
    data = plot_data["data"]
    baseline = plot_data["baseline"]
    baseline_multi = plot_data["baseline_multi"]
    indicators = plot_data["indicators"]
    equity_data = plot_data["equity_data"]
    trades = plot_data["trades"]
    phase_data = plot_data["phase_data"]
    traded_tickers = plot_data["traded_tickers"]
    tickers_to_plot = plot_data["tickers_to_plot"]
    index = plot_data["index"]
    is_datetime_index = plot_data["is_datetime_index"]

    def new_bokeh_figure(**kwargs):
        # Set default parameters with full width support
        defaults = {
            'x_axis_type': "linear",
            'y_axis_type': "linear", 
            'y_axis_location': "right",
            'width': plot_width,
            'height': 400,
            'sizing_mode': 'stretch_width' if plot_width is None else None,  # Full width when no specific width set
            'tools': "xpan,xwheel_zoom,box_zoom,undo,redo,reset,save",
            'active_drag': "xpan",
            'active_scroll': "xwheel_zoom",
        }
        
        # Apply theme defaults if available
        if THEMES_AVAILABLE:
            theme = get_current_theme()
            theme_defaults = {
                'background_fill_color': theme.background_fill_color,
                'border_fill_color': theme.border_fill_color,
            }
            defaults.update(theme_defaults)
        
        # Merge defaults with user kwargs (user takes precedence)
        final_kwargs = {**defaults, **kwargs}
        
        fig = _figure(**final_kwargs)
        
        # Apply theme styling
        if THEMES_AVAILABLE:
            apply_theme_to_new_figure(fig)
        
        return fig

    # pad = (index[-1] - index[0]) / 20  # unused

    if index.size > 1:
        fig_ohlc = new_bokeh_figure(x_range=Range1d(int(index[0]), int(index[-1])))
    else:
        fig_ohlc = new_bokeh_figure()
    figs_above_ohlc, figs_below_ohlc = [], []

    source = ColumnDataSource(baseline)
    source.add((baseline.Close >= baseline.Open).astype(np.uint8).astype(str), "inc")

    trade_source = ColumnDataSource(
        dict(
            index=trades["ExitBar"],
            datetime=trades["ExitTime"],
            exit_price=trades["ExitPrice"],
            ticker=trades["Ticker"],
            size=trades["Size"],
            returns_positive=(trades["Gross%"] > 0).astype(int).astype(str),
        )
    )

    inc_cmap = factor_cmap("inc", COLORS, ["0", "1"])
    cmap = factor_cmap("returns_positive", COLORS, ["0", "1"])
    colors_darker = [lightness(BEAR_COLOR, 0.35), lightness(BULL_COLOR, 0.35)]
    trades_cmap = factor_cmap("returns_positive", colors_darker, ["0", "1"])

    if is_datetime_index:
        # LEVERAGE BOKEH'S DYNAMIC CAPABILITIES: Let Bokeh handle tick selection automatically
        # Use intelligent formatting that adapts to zoom level
        
        # Simplify to get the graph working first, then add complexity
        fig_ohlc.xaxis.formatter = CustomJSTickFormatter(
            args=dict(
                axis=fig_ohlc.xaxis[0],
                formatter=DatetimeTickFormatter(days="%d %b", months="%b %Y", years="%Y"),
                source=source,
            ),
            code="""
// Simple, robust formatter - just map ticks to datetime values
this.labels = this.labels || formatter.doFormat(ticks
                                                .map(i => source.data.datetime[i])
                                                .filter(t => t !== undefined));
return this.labels[index] || "";
        """,
        )

    ohlc_extreme_values = baseline[["High", "Low"]].copy(deep=False)
    ohlc_tooltips = []

    def new_indicator_figure(**kwargs):
        kwargs.setdefault("height", 80)
        fig = new_bokeh_figure(
            x_range=fig_ohlc.x_range,
            active_scroll="xwheel_zoom",
            active_drag="xpan",
            **kwargs,
        )
        
        # Apply same ticker and formatter as main chart for consistency
        if is_datetime_index:
            fig.xaxis.ticker = fig_ohlc.xaxis.ticker  # Use same custom ticker
            fig.xaxis.formatter = fig_ohlc.xaxis[0].formatter  # Use same formatter
        
        fig.xaxis.visible = False
        fig.yaxis.minor_tick_line_color = None
        fig.add_layout(Legend(), "center")
        fig.legend.orientation = "horizontal"
        fig.legend.background_fill_alpha = 0.8
        fig.legend.border_line_alpha = 0
        return fig

    def set_tooltips(fig, tooltips=(), vline=True, renderers=()):
        tooltips = list(tooltips)
        renderers = list(renderers)

        if is_datetime_index:
            formatters = {"@datetime": "datetime"}
            tooltips = [("Date", "@datetime{%c}")] + tooltips
        else:
            formatters = {}
            tooltips = [("#", "@index")] + tooltips
        fig.add_tools(
            HoverTool(
                point_policy="follow_mouse",
                renderers=renderers,
                formatters=formatters,
                tooltips=tooltips,
                mode="vline" if vline else "mouse",
            )
        )

    def _plot_broker_session(is_return=False):
        """Broker session showing cash and total value"""
        cash = equity_data["Cash"].copy() if "Cash" in equity_data.columns else None
        total_value = equity_data["Equity"].copy()  # Total portfolio value
        if smooth_equity:
            interest_points = pd.Index(
                [
                    # Beginning and end
                    total_value.index[0],
                    total_value.index[-1],
                    # Peak and valley points
                    total_value.idxmax(),
                    total_value.idxmin(),
                    cash.idxmax() if not cash.empty else total_value.index[0],
                ]
            )
            select = pd.Index(trades["ExitBar"]).union(interest_points)
            select = select.unique().dropna()
            total_value = total_value.iloc[select].reindex(total_value.index)
            cash = cash.iloc[select].reindex(cash.index)

            total_value.interpolate(inplace=True)
            cash.interpolate(inplace=True)

        if is_return:
            initial_total = total_value.iloc[0]
            initial_cash = cash.iloc[0]
            total_value -= initial_total
            cash -= initial_cash

        yaxis_label = "Return" if is_return else "Broker Session"

        # Add data to source
        source.add(total_value, "total_value")
        source.add(cash, "cash")

        fig = new_indicator_figure(
            y_axis_label=yaxis_label, **({} if plot_drawdown else dict(height=110))
        )

        # Format settings - always show absolute dollar values as raw numbers
        tooltip_format_total = "@total_value{$ 0,0}"
        tooltip_format_cash = "@cash{$ 0,0}"
        tick_format = "$ 0,0"

        # Plot lines
        r1 = fig.line(
            "index",
            "total_value",
            source=source,
            line_width=2,
            line_alpha=1,
            color="#3E54D3",
            legend_label="Total Value",
        )
        fig.line(
            "index",
            "cash",
            source=source,
            line_width=1.5,
            line_alpha=0.8,
            color="#4FE086",
            legend_label="Cash",
        )

        # Set tooltips - use only one renderer to avoid duplicate legends
        set_tooltips(
            fig,
            [
                ("Total Value", tooltip_format_total),
                ("Cash", tooltip_format_cash),
            ],
            renderers=[r1],
        )

        fig.yaxis.formatter = NumeralTickFormatter(format=tick_format)

        # Peak markers
        argmax_total = total_value.idxmax()
        fig.scatter(argmax_total, total_value[argmax_total], color="cyan", size=8)

        # Final values
        fig.scatter(index[-1], total_value.iloc[-1], color="blue", size=8)

        # Set reasonable tick intervals to prevent label cramming - do this after data is plotted
        # Calculate data range from the actual data values
        total_min = float(total_value.min()) if hasattr(total_value, "min") else 0
        total_max = float(total_value.max()) if hasattr(total_value, "max") else 100000
        cash_min = (
            float(cash.min())
            if cash is not None and hasattr(cash, "min")
            else total_min
        )
        cash_max = (
            float(cash.max())
            if cash is not None and hasattr(cash, "max")
            else total_max
        )

        data_min = min(total_min, cash_min)
        data_max = max(total_max, cash_max)
        value_range = (
            abs(data_max - data_min) if data_max != data_min else abs(data_max)
        )

        if value_range > 0:
            # Aim for 5-8 ticks maximum
            desired_interval = value_range / 6
            # Round to nice numbers
            if desired_interval > 10000:
                tick_interval = round(desired_interval / 10000) * 10000
            elif desired_interval > 1000:
                tick_interval = round(desired_interval / 1000) * 1000
            elif desired_interval > 100:
                tick_interval = round(desired_interval / 100) * 100
            else:
                tick_interval = max(1, round(desired_interval))

            # Set the ticker to use our calculated interval
            from bokeh.models import FixedTicker

            if tick_interval > 0:
                start_tick = (data_min // tick_interval) * tick_interval
                end_tick = ((data_max // tick_interval) + 1) * tick_interval
                ticks = list(
                    range(
                        int(start_tick),
                        int(end_tick + tick_interval),
                        int(tick_interval),
                    )
                )
                if len(ticks) > 10:  # Fallback if too many ticks
                    ticks = ticks[::2]  # Take every other tick
                if len(ticks) >= 2:  # Only set if we have reasonable ticks
                    fig.yaxis.ticker = FixedTicker(ticks=ticks)

        figs_above_ohlc.append(fig)

    def _plot_equity_stack_section(relative=True):
        """Equity stack area chart section"""
        equity = equity_data.iloc[:, 1:-2].copy().abs()
        equity = equity.loc[:, equity.sum() > 0]
        names = list(equity.columns)
        if relative:
            equity = equity.divide(equity.sum(axis=1), axis=0)
        equity_source = ColumnDataSource(equity)
        equity_source.add(data.index, "datetime")

        yaxis_label = "Allocation"
        fig = new_indicator_figure(
            y_axis_label=yaxis_label, height=max(60 + len(names), 80)
        )

        if relative:
            tooltip_format = [f"@{ticker}{{+0,0.[000]%}}" for ticker in names]
            tick_format = "0,0.[00]%"
            equity_source.add(pd.Series(1, index=data.index), "equity")
        else:
            tooltip_format = [f"@{ticker}{{$ 0,0}}" for ticker in names]
            tick_format = "$ 0.0 a"
            equity_source.add(equity_data["Equity"], "equity")

        cg = colorgen()
        colors = [next(cg) for _ in range(len(names))]
        r = fig.line(
            "index", "equity", source=equity_source, line_width=1, line_alpha=0
        )
        fig.varea_stack(
            stackers=names,
            x="index",
            color=colors,
            legend_label=names,
            source=equity_source,
        )
        set_tooltips(fig, list(zip(names, tooltip_format)), renderers=[r])
        fig.yaxis.formatter = NumeralTickFormatter(format=tick_format)

        figs_above_ohlc.append(fig)

    def _plot_drawdown_section():
        """Drawdown section"""
        fig = new_indicator_figure(y_axis_label="Drawdown")
        drawdown = equity_data["DrawdownPct"]
        argmax = drawdown.idxmax()
        source.add(drawdown, "drawdown")
        r = fig.line("index", "drawdown", source=source, line_width=1.3)
        fig.scatter(
            argmax,
            drawdown[argmax],
            legend_label="Peak (-{:.1f}%)".format(100 * drawdown[argmax]),
            color="red",
            size=8,
        )
        set_tooltips(fig, [("Drawdown", "@drawdown{-0.[0]%}")], renderers=[r])
        fig.yaxis.formatter = NumeralTickFormatter(format="-0.[0]%")
        return fig

    def _plot_pl_section():
        """Profit/Loss markers section"""
        fig = new_indicator_figure(y_axis_label="Profit / Loss", y_axis_type="linear")
        fig.add_layout(
            Span(
                location=0,
                dimension="width",
                line_color="#666666",
                line_dash="dashed",
                line_width=1,
            )
        )
        returns_long = np.where(trades["Size"] > 0, trades["Gross%"], np.nan)
        returns_short = np.where(trades["Size"] < 0, trades["Gross%"], np.nan)
        size = (trades["Size"] * trades["EntryPrice"]).abs().astype(int)
        size = np.interp(size, (size.min(), size.max()), (4, 12))
        pos_value = (trades["Size"] * trades["EntryPrice"]).abs()
        trade_source.add(returns_long, "returns_long")
        trade_source.add(returns_short, "returns_short")
        trade_source.add(pos_value, "position_value")
        trade_source.add(size, "marker_size")
        if "count" in trades:
            trade_source.add(trades["count"], "count")
        r1 = fig.scatter(
            "index",
            "returns_long",
            source=trade_source,
            fill_color=cmap,
            marker="circle",
            size="marker_size",
        )
        r2 = fig.scatter(
            "index",
            "returns_short",
            source=trade_source,
            fill_color=cmap,
            marker="star",
            size="marker_size",
        )
        tooltips = [("Ticker", "@ticker"), ("Pos.Value", "@position_value{$ 0,0.00}")]
        if "count" in trades:
            tooltips.append(("Count", "@count{0,0}"))
        set_tooltips(
            fig,
            tooltips + [("P/L", "@returns_long{+0.[000]%}")],
            vline=False,
            renderers=[r1],
        )
        set_tooltips(
            fig,
            tooltips + [("P/L", "@returns_short{+0.[000]%}")],
            vline=False,
            renderers=[r2],
        )
        fig.yaxis.formatter = NumeralTickFormatter(format="0.[00]%")
        return fig

    def _plot_volume_section():
        """Volume section"""
        fig = new_bokeh_figure(y_axis_label="Volume")
        
        # Apply same ticker as main chart for consistency
        if is_datetime_index:
            fig.xaxis.ticker = fig_ohlc.xaxis.ticker  # Use same ticker as main chart
        
        fig.xaxis.formatter = fig_ohlc.xaxis[0].formatter
        fig.xaxis.visible = True
        fig_ohlc.xaxis.visible = False  # Show only Volume's xaxis
        r = fig.vbar("index", BAR_WIDTH, "Volume", source=source, color=inc_cmap)
        set_tooltips(fig, [("Volume", "@Volume{0.00 a}")], renderers=[r])
        fig.yaxis.formatter = NumeralTickFormatter(format="0 a")
        
        # Add phase highlighting if available
        if hasattr(bt._strategy, 'phase_manager') and hasattr(bt._strategy.phase_manager, 'phase_completion_points'):
            try:
                from algo.utils.plotting.phase import calculate_phase_ranges_from_completion_points, render_phase_highlights_with_data
                
                # Get the first ticker for phase data (assuming main chart shows first ticker)
                if tickers_to_plot and len(tickers_to_plot) > 0:
                    ticker = tickers_to_plot[0]
                    
                    # Calculate phase ranges
                    phase_data = {'completion_points': bt._strategy.phase_manager.phase_completion_points}
                    symbol_data = bt._strategy.data  # Assuming this has the price data
                    symbol_trades = pd.DataFrame()  # Empty trades for now
                    
                    phase_ranges = calculate_phase_ranges_from_completion_points(
                        phase_data['completion_points'], ticker, symbol_data, symbol_trades
                    )
                    
                    # Get volume data range for proper scaling
                    volume_data = source.data['Volume']
                    volume_max = max(volume_data) if volume_data else 1
                    volume_range = volume_max
                    
                    # Apply phase highlights to main volume chart
                    render_phase_highlights_with_data(fig, phase_ranges, volume_max, volume_range)
            except (ImportError, AttributeError, KeyError):
                # Skip phase highlighting if not available
                pass
        
        # Add white vertical lines at the beginning of each month
        if is_datetime_index:
            from bokeh.models import Span
            import pandas as pd
            
            # Access datetime data through the source object
            datetime_values = source.data['datetime']
            month_starts = []
            seen_months = set()
            
            # Find month start positions
            for idx, dt_timestamp in enumerate(datetime_values):
                dt = pd.Timestamp(dt_timestamp)
                month_key = (dt.year, dt.month)
                if month_key not in seen_months:
                    month_starts.append(idx)
                    seen_months.add(month_key)
            
            # Add white vertical lines at month start positions (left edge of bars)
            for month_pos in month_starts:
                span = Span(
                    location=month_pos - BAR_WIDTH/2,
                    dimension='height',
                    line_color='white',
                    line_width=2,
                    line_alpha=0.8
                )
                fig.add_layout(span)
        
        return fig

    def _plot_superimposed_ohlc():
        """Superimposed, downsampled vbars"""
        from .lib import OHLCV_AGG

        time_resolution = pd.DatetimeIndex(baseline["datetime"]).resolution
        resample_rule = (
            superimpose
            if isinstance(superimpose, str)
            else dict(day="M", hour="D", minute="H", second="T", millisecond="S").get(
                time_resolution
            )
        )
        if not resample_rule:
            warnings.warn(
                f"'Can't superimpose OHLC data with rule '{resample_rule}'"
                f"(index datetime resolution: '{time_resolution}'). Skipping.",
                stacklevel=4,
            )
            return

        df2 = (
            baseline.assign(_width=1)
            .set_index("datetime")
            .resample(resample_rule, label="left")
            .agg({**OHLCV_AGG, "_width": "count"})
        )

        # Check if resampling was downsampling; error on upsampling
        orig_freq = _data_period(baseline["datetime"])
        resample_freq = _data_period(df2.index)
        if resample_freq < orig_freq:
            raise ValueError(
                "Invalid value for `superimpose`: Upsampling not supported."
            )
        if resample_freq == orig_freq:
            warnings.warn(
                "Superimposed OHLC plot matches the original plot. Skipping.",
                stacklevel=4,
            )
            return

        df2.index = df2["_width"].cumsum().shift(1).fillna(0)
        df2.index += df2["_width"] / 2 - 0.5
        df2["_width"] -= 0.1  # Candles don't touch

        df2["inc"] = (df2.Close >= df2.Open).astype(int).astype(str)
        df2.index.name = None
        source2 = ColumnDataSource(df2)
        colors_lighter = [lightness(BEAR_COLOR, 0.92), lightness(BULL_COLOR, 0.92)]
        fig_ohlc.segment(
            "index", "High", "index", "Low", source=source2, 
            color=factor_cmap("inc", colors_lighter, ["0", "1"])
        )
        fig_ohlc.vbar(
            "index",
            "_width",
            "Open",
            "Close",
            source=source2,
            line_color=None,
            fill_color=factor_cmap("inc", colors_lighter, ["0", "1"]),
        )

    def _plot_ohlc_trades():
        """Trade entry / exit markers on OHLC plot"""
        trade_source.add(
            trades[["EntryBar", "ExitBar"]].to_numpy().tolist(), "position_lines_xs"
        )
        trade_source.add(
            trades[["EntryPrice", "ExitPrice"]].to_numpy().tolist(), "position_lines_ys"
        )
        fig_ohlc.multi_line(
            xs="position_lines_xs",
            ys="position_lines_ys",
            source=trade_source,
            line_color=trades_cmap,
            legend_label=f"Trades ({len(trades)})",
            line_width=8,
            line_alpha=1,
            line_dash="dotted",
        )

    def _plot_ohlc_universe():
        fig = fig_ohlc
        ohlc_colors = colorgen()
        label_tooltip_pairs = []
        # Respect traded tickers only (if provided)
        for ticker in tickers_to_plot[:10]:
            color = next(ohlc_colors)
            source_name = ticker
            arr = data.loc[:, (ticker, "Close")]
            source.add(arr, source_name)
            label_tooltip_pairs.append(
                (source_name, f"@{{{source_name}}}{{0,0.0[0000]}}")
            )
            ohlc_extreme_values[source_name] = arr.reset_index(drop=True)
            fig.line(
                "index",
                source_name,
                source=source,
                legend_label=source_name,
                line_color=color,
                line_width=2,
            )
        ohlc_tooltips.extend(label_tooltip_pairs)
        if len(tickers_to_plot) > 10:
            fig.line(
                0,
                0,
                legend_label=f"{len(tickers_to_plot)-10} more tickers hidden",
                line_color="black",
            )
        fig.legend.orientation = "horizontal"
        fig.legend.background_fill_alpha = 0.8
        fig.legend.border_line_alpha = 0

    def _plot_indicators():
        """Strategy indicators"""

        def _too_many_dims(value):
            if value.ndim > 2:
                warnings.warn(
                    f"Can't plot indicators with >2D ('{value.name}')", stacklevel=5
                )
                return True
            return False

        class LegendStr(str):
            # The legend string is such a string that only matches
            # itself if it's the exact same object. This ensures
            # legend items are listed separately even when they have the
            # same string contents. Otherwise, Bokeh would always consider
            # equal strings as one and the same legend item.
            def __eq__(self, other):
                return self is other

        ohlc_colors = colorgen()
        indicator_figs = []

        for i, value in enumerate(indicators):

            if not value.attrs.get("plot") or _too_many_dims(value):
                continue

            is_overlay = value.attrs["overlay"]

            # Skip overlay indicators for multi-symbol data since they'll be plotted on individual charts
            if is_overlay and baseline_multi is not None:
                continue
            is_scatter = value.attrs["scatter"]
            # If indicator is per-ticker DataFrame, restrict to traded tickers if present
            if isinstance(value, pd.DataFrame):
                if traded_tickers:
                    cols = [c for c in value.columns if c in traded_tickers]
                    if cols:
                        value = value[cols]
                legend_label = (
                    [LegendStr(f'{value.attrs["name"]} {col}') for col in value.columns]
                    if value.attrs["name"]
                    else [LegendStr(col) for col in value.columns]
                )
                series_lst = [value[col] for col in value.columns]
            else:
                legend_label = [LegendStr(value.attrs["name"] or value.name)]
                series_lst = [value]
            if is_overlay:
                fig = fig_ohlc
            else:
                fig = new_indicator_figure(height=60 + 20 * len(series_lst))
                indicator_figs.append(fig)
            tooltips = []
            colors = value.attrs["color"]
            colors = (
                colors
                and cycle(_as_list(colors))
                or (cycle([next(ohlc_colors)]) if is_overlay else colorgen())
            )
            for j, arr in enumerate(series_lst, 1):
                color = next(colors)
                source_name = f"{legend_label[j-1]}_{i}_{j}"
                if arr.dtype == bool:
                    arr = arr.astype(int)
                source.add(arr, source_name)
                tooltips.append(f"@{{{source_name}}}{{0,0.0[0000]}}")
                if is_overlay:
                    # Align overlay series to baseline integer index so autoscale picks it up
                    arr_aligned = pd.Series(arr).reset_index(drop=True)
                    ohlc_extreme_values[source_name] = arr_aligned
                    if is_scatter:
                        fig.scatter(
                            "index",
                            source_name,
                            source=source,
                            legend_label=legend_label[j - 1],
                            color=color,
                            line_color="black",
                            fill_alpha=0.8,
                            size=BAR_WIDTH / 2 * 1.5,
                        )
                    else:
                        fig.line(
                            "index",
                            source_name,
                            source=source,
                            legend_label=legend_label[j - 1],
                            line_color=color,
                            line_width=1.3,
                        )
                else:
                    if is_scatter:
                        r = fig.scatter(
                            "index",
                            source_name,
                            source=source,
                            legend_label=legend_label[j - 1],
                            color=color,
                            size=BAR_WIDTH / 2 * 0.9,
                        )
                    else:
                        r = fig.line(
                            "index",
                            source_name,
                            source=source,
                            legend_label=legend_label[j - 1],
                            line_color=color,
                            line_width=1.3,
                        )
                    # Add dashed centerline just because
                    mean = float(pd.Series(arr).mean())
                    if not np.isnan(mean) and (
                        abs(mean) < 0.1
                        or round(abs(mean), 1) == 0.5
                        or round(abs(mean), -1) in (50, 100, 200)
                    ):
                        fig.add_layout(
                            Span(
                                location=float(mean),
                                dimension="width",
                                line_color="#666666",
                                line_dash="dashed",
                                line_width=0.5,
                            )
                        )

            label_tooltip_pairs = [
                (label, tooltip) for label, tooltip in zip(legend_label, tooltips)
            ]
            if is_overlay:
                ohlc_tooltips.extend(label_tooltip_pairs)
            else:
                # Always set tooltips for non-overlay indicators in their own panels
                set_tooltips(fig, label_tooltip_pairs, vline=True, renderers=[r])
                # If the sole indicator line on this figure,
                # have the legend only contain text without the glyph
                if len(value) == 1:
                    fig.legend.glyph_width = 0
        return indicator_figs

    def _plot_top_summary_chart():
        """Plot multiple symbols as line on the same OHLC chart, plot OHLC bars if only one symbol"""
        if baseline_multi is None:
            return None

        # Plot each symbol with different colors
        first_line = None
        fig = fig_ohlc
        ohlc_colors = colorgen()
        label_tooltip_pairs = []
        # Respect traded tickers only (if provided)
        for ticker in tickers_to_plot:
            color = next(ohlc_colors)
            source_name = ticker
            arr = data.loc[:, (ticker, "Close")]
            source.add(arr, source_name)
            label_tooltip_pairs.append(
                (source_name, f"@{{{source_name}}}{{0,0.0[0000]}}")
            )
            ohlc_extreme_values[source_name] = arr.reset_index(drop=True)
            if first_line is None:
                first_line = fig.line(
                    "index",
                    source_name,
                    source=source,
                    legend_label=source_name,
                    line_color=color,
                    line_width=2,
                )
            else:
                fig.line(
                    "index",
                    source_name,
                    source=source,
                    legend_label=source_name,
                    line_color=color,
                    line_width=2,
                )
        ohlc_tooltips.extend(label_tooltip_pairs)
        if len(tickers_to_plot) > 10:
            fig.line(
                0,
                0,
                legend_label=f"{len(tickers_to_plot)-10} more tickers hidden",
                line_color="black",
            )
        fig.legend.orientation = "horizontal"
        fig.legend.background_fill_alpha = 0.8
        fig.legend.border_line_alpha = 0

        return first_line

    def _create_unified_legend(
        symbol_fig, symbol, symbol_source, indicators, ohlc_renderer=None
    ):
        """Create a single unified dynamic tooltip showing both OHLCV and indicator values"""

        
        # Add indicators to the symbol source and tooltip list
        for i, value in enumerate(indicators):
            # Only plot indicators with plot=True and overlay=True (same logic as before)
            if not value.attrs.get("plot") or value.ndim > 2:
                continue

            is_overlay = value.attrs.get("overlay")
            if not is_overlay:
                continue

            is_scatter = value.attrs.get("scatter")

            # Handle indicators for this specific symbol
            if isinstance(value, pd.DataFrame):
                if symbol in value.columns:
                    symbol_indicator = value[symbol]
                    indicator_name = value.attrs.get("name")
                else:
                    continue
            else:
                symbol_indicator = value
                indicator_name = value.attrs.get("name") or value.name or "Indicator"

            # Add indicator data to symbol source
            source_name = f"{indicator_name}_{i}"
            if symbol_indicator.dtype == bool:
                symbol_indicator = symbol_indicator.astype(int)

            # Reindex to match symbol data
            source_index = pd.Index(symbol_source.data["index"])
            if len(symbol_indicator) == len(source_index):
                reindexed_values = symbol_indicator.values
            else:
                symbol_indicator = symbol_indicator.reindex(
                    source_index, method="ffill"
                ).fillna(method="bfill")
                reindexed_values = symbol_indicator.values

            symbol_source.add(reindexed_values, source_name)

            # Add to unified tooltip

            # Get indicator color
            colors = value.attrs.get("color")
            if colors:
                color = colors if isinstance(colors, str) else colors[0]
            else:
                default_colors = [
                    "orange",
                    "cyan",
                    "yellow",
                    "magenta",
                    "brown",
                    "pink",
                    "gray",
                ]
                color = default_colors[i % len(default_colors)]

            # Plot the actual indicator (visible)
            if is_scatter:
                symbol_fig.scatter(
                    "index",
                    source_name,
                    source=symbol_source,
                    color=color,
                    line_color="black",
                    fill_alpha=0.8,
                    size=BAR_WIDTH / 2 * 1.5,
                )
            else:
                symbol_fig.line(
                    "index",
                    source_name,
                    source=symbol_source,
                    color=color,
                    line_width=1.3,
                )

        # Create a single comprehensive hover tool that shows everything - ONLY ONCE after processing all indicators
        # Remove any existing HoverTools to ensure we only have our unified one
        existing_hover_tools = [
            tool for tool in symbol_fig.tools if isinstance(tool, HoverTool)
        ]
        for hover_tool in existing_hover_tools:
            symbol_fig.tools.remove(hover_tool)



        # Add dynamic title update callback
        title_callback = CustomJS(args=dict(source=symbol_source, title=symbol_fig.title), code=f"""
            if (cb_data.index && cb_data.index.indices && cb_data.index.indices.length > 0) {{
                const i = cb_data.index.indices[0];
                const data = source.data;
                
                // Format OHLC values
                const open = data['Open'][i] ? data['Open'][i].toFixed(2) : 'N/A';
                const high = data['High'][i] ? data['High'][i].toFixed(2) : 'N/A';
                const low = data['Low'][i] ? data['Low'][i].toFixed(2) : 'N/A';
                const close = data['Close'][i] ? data['Close'][i].toFixed(2) : 'N/A';
                const volume = data['Volume'][i] ? data['Volume'][i].toLocaleString() : 'N/A';
                
                // Format date if available
                let dateStr = '';
                if (data['datetime'] && data['datetime'][i]) {{
                    const date = new Date(data['datetime'][i]);
                    dateStr = date.toLocaleDateString() + ' ';
                }}
                
                // Format SMA values if available
                let smaStr = '';
                for (const key in data) {{
                    if (key.includes('SMA') && key.includes('10') && data[key][i] !== undefined && data[key][i] !== null) {{
                        const sma10 = data[key][i].toFixed(2);
                        if (!smaStr.includes('SMA10=')) {{
                            smaStr += ` SMA10=${{sma10}}`;
                        }}
                    }}
                    if (key.includes('SMA') && key.includes('20') && data[key][i] !== undefined && data[key][i] !== null) {{
                        const sma20 = data[key][i].toFixed(2);
                        if (!smaStr.includes('SMA20=')) {{
                            smaStr += ` SMA20=${{sma20}}`;
                        }}
                    }}
                    if (key.includes('SMA') && key.includes('50') && data[key][i] !== undefined && data[key][i] !== null) {{
                        const sma50 = data[key][i].toFixed(2);
                        if (!smaStr.includes('SMA50=')) {{
                            smaStr += ` SMA50=${{sma50}}`;
                        }}
                    }}
                }}
                
                // Update title with OHLCV and SMA data
                title.text = `{symbol} OHLC - ${{dateStr}}O=${{open}} H=${{high}} L=${{low}} C=${{close}}${{smaStr}} Volume=${{volume}}`;
            }} else {{
                // Reset to default title when not hovering
                title.text = `{symbol} OHLC`;
            }}
        """)

        # Create a minimal hover tool without tooltips - just for triggering the title callback
        minimal_hover = HoverTool(
            tooltips=None,  # No tooltips
            mode="vline",
            renderers=[ohlc_renderer] if ohlc_renderer else []
        )
        minimal_hover.callback = title_callback
        symbol_fig.add_tools(minimal_hover)

        # Hide the default legend since we're using the comprehensive tooltip instead
        if symbol_fig.legend:
            symbol_fig.legend.visible = False

    def _plot_separate_symbols():
        """Create separate combined OHLC+Volume charts for each symbol that has trades"""
        if baseline_multi is None:
            return []

        # Get all available symbols
        all_symbols = baseline_multi.columns.get_level_values(0).unique()
        
        # Filter to only symbols that have trades
        if traded_tickers:
            symbols_to_plot = [symbol for symbol in all_symbols if symbol in traded_tickers]
            if not symbols_to_plot:
                # No symbols have trades, return empty list
                return []
        else:
            # If no traded tickers info, plot all symbols (fallback)
            symbols_to_plot = all_symbols
        
        symbol_figs = []

        for symbol in symbols_to_plot:
            # TODO - Volume will displa
            # Extract symbol data
            symbol_data = baseline_multi.loc[:, (symbol, slice(None))]
            symbol_data.columns = symbol_data.columns.get_level_values(1)

            # Reset index to match the main chart
            original_datetime = symbol_data.index  # Save original datetime index
            symbol_data = symbol_data.reset_index(drop=True)
            symbol_data["datetime"] = original_datetime  # Add actual datetime column

            # Create source for this symbol
            symbol_source = ColumnDataSource(symbol_data)
            # Ensure a numeric 'index' column exists and matches panel length
            if "index" not in symbol_source.data or len(
                symbol_source.data["index"]
            ) != len(symbol_data):
                symbol_source.add(list(range(len(symbol_data))), "index")

            # Add inc column for coloring
            symbol_source.add(
                (symbol_data["Close"] >= symbol_data["Open"]).astype(int).astype(str),
                "inc",
            )

            # Create OHLC figure with dynamic title
            ohlc_fig = new_bokeh_figure(
                title=f"{symbol} OHLC",
                height=800,  # Standard height for OHLC only
                x_range=fig_ohlc.x_range,  # Share x-axis with main chart
            )
            
            # Add crosshair tool to individual OHLC chart
            ohlc_crosshair = CrosshairTool(
                dimensions="both",  # Show both x and y crosshairs
                line_color="#758696",  # Custom gray-blue color for individual charts
                line_alpha=0.7,
                line_width=1.5,
            )
            ohlc_fig.add_tools(ohlc_crosshair)
            # Don't override active_inspect - let both CrosshairTool and HoverTool be active
            
            # Apply datetime formatting to x-axis (same as main chart)
            if is_datetime_index:
                ohlc_fig.xaxis.formatter = CustomJSTickFormatter(
                    args=dict(
                        axis=ohlc_fig.xaxis[0],
                        formatter=DatetimeTickFormatter(days="%d %b", months="%b %Y", years="%Y"),
                        source=symbol_source,
                    ),
                    code="""
// Simple, robust formatter - just map ticks to datetime values
this.labels = this.labels || formatter.doFormat(ticks
                                                .map(i => source.data.datetime[i])
                                                .filter(t => t !== undefined));
return this.labels[index] || "";
""",
                )

            # Plot OHLC bars with color-matched high-low lines
            ohlc_fig.segment(
                "index", "High", "index", "Low", source=symbol_source, 
                color=factor_cmap("inc", COLORS, ["0", "1"])
            )
            ohlc_bars = ohlc_fig.vbar(
                "index",
                BAR_WIDTH,
                "Open",
                "Close",
                source=symbol_source,
                line_color=factor_cmap("inc", COLORS, ["0", "1"]),
                fill_color=factor_cmap("inc", COLORS, ["0", "1"]),
            )

            # Dynamic title will be added later with the unified legend

            # Create separate Volume figure
            volume_fig = new_bokeh_figure(
                title=f"{symbol} Volume",
                height=150,  # Smaller height for volume
                x_range=ohlc_fig.x_range,  # Share x-axis with OHLC chart
                y_axis_label="Volume",
            )
            
            # Add crosshair tool to individual Volume chart
            volume_crosshair = CrosshairTool(
                dimensions="both",  # Show both x and y crosshairs
                line_color="cyan",  # Different color for volume charts
                line_alpha=0.6,
                line_width=1,
            )
            volume_fig.add_tools(volume_crosshair)
            # Don't override active_inspect - let both CrosshairTool and HoverTool be active
            
            # Apply datetime formatting to volume chart x-axis
            if is_datetime_index:
                # Apply same ticker and formatter as OHLC chart
                volume_fig.xaxis.ticker = ohlc_fig.xaxis.ticker
                volume_fig.xaxis.formatter = ohlc_fig.xaxis[0].formatter
            
            # Hide OHLC x-axis, show only volume x-axis
            ohlc_fig.xaxis.visible = False
            volume_fig.xaxis.visible = True

            # Add the columns that the autoscaling callback expects
            symbol_source.add(symbol_data["High"], "ohlc_high")
            symbol_source.add(symbol_data["Low"], "ohlc_low")

            # Plot volume bars in separate chart
            volume_bars = volume_fig.vbar(
                "index",
                BAR_WIDTH,
                "Volume",
                0,  # Start from zero
                source=symbol_source,
                line_color="black",
                fill_color=factor_cmap("inc", COLORS, ["0", "1"]),
                alpha=0.7,
            )
            
            # Add dynamic title callback for volume chart without tooltips
            volume_title_callback = CustomJS(args=dict(source=symbol_source, title=volume_fig.title), code=f"""
                if (cb_data.index && cb_data.index.indices && cb_data.index.indices.length > 0) {{
                    const i = cb_data.index.indices[0];
                    const data = source.data;
                    
                    // Format OHLC values
                    const open = data['Open'][i] ? data['Open'][i].toFixed(2) : 'N/A';
                    const high = data['High'][i] ? data['High'][i].toFixed(2) : 'N/A';
                    const low = data['Low'][i] ? data['Low'][i].toFixed(2) : 'N/A';
                    const close = data['Close'][i] ? data['Close'][i].toFixed(2) : 'N/A';
                    const volume = data['Volume'][i] ? data['Volume'][i].toLocaleString() : 'N/A';
                    
                    // Format SMA values if available
                    let smaStr = '';
                    for (const key in data) {{
                        if (key.includes('SMA') && key.includes('10') && data[key][i] !== undefined && data[key][i] !== null) {{
                            const sma10 = data[key][i].toFixed(2);
                            if (!smaStr.includes('SMA10=')) {{
                                smaStr += ` SMA10=${{sma10}}`;
                            }}
                        }}
                        if (key.includes('SMA') && key.includes('20') && data[key][i] !== undefined && data[key][i] !== null) {{
                            const sma20 = data[key][i].toFixed(2);
                            if (!smaStr.includes('SMA20=')) {{
                                smaStr += ` SMA20=${{sma20}}`;
                            }}
                        }}
                        if (key.includes('SMA') && key.includes('50') && data[key][i] !== undefined && data[key][i] !== null) {{
                            const sma50 = data[key][i].toFixed(2);
                            if (!smaStr.includes('SMA50=')) {{
                                smaStr += ` SMA50=${{sma50}}`;
                            }}
                        }}
                    }}
                    
                    // Format date if available
                    let dateStr = '';
                    if (data['datetime'] && data['datetime'][i]) {{
                        const date = new Date(data['datetime'][i]);
                        dateStr = date.toLocaleDateString() + ' ';
                    }}
                    
                    // Update title with OHLCV, SMA and volume data
                    title.text = `{symbol} Volume=${{volume}}`;
                }} else {{
                    // Reset to default title when not hovering
                    title.text = `{symbol} Volume`;
                }}
            """)
            
            # Create minimal hover tool without tooltips for volume chart
            volume_hover = HoverTool(
                tooltips=None,  # No tooltips
                renderers=[volume_bars],
                mode="vline"
            )
            volume_hover.callback = volume_title_callback
            volume_fig.add_tools(volume_hover)
            volume_fig.yaxis.formatter = NumeralTickFormatter(format="0 a")

            # Add white vertical lines at the beginning of each month (per-symbol volume)
            if is_datetime_index:
                from bokeh.models import Span
                import pandas as pd
                
                # Use the symbol-specific source for date detection
                datetime_values = symbol_source.data.get('datetime')
                month_starts = []
                seen_months = set()
                
                for idx, dt_timestamp in enumerate(datetime_values):
                    dt = pd.Timestamp(dt_timestamp)
                    month_key = (dt.year, dt.month)
                    if month_key not in seen_months:
                        month_starts.append(idx)
                        seen_months.add(month_key)
                
                # Add white vertical lines at month start positions (left edge of bars)
                for month_pos in month_starts:
                    span = Span(
                        location=month_pos - BAR_WIDTH/2 - 0.1,
                        dimension='height',
                        line_color='white',
                        line_width=2,
                        line_alpha=0.8
                    )
                    volume_fig.add_layout(span)

            # Filter trades for this specific symbol (used for both trade markers and phase highlights)
            symbol_trades = pd.DataFrame()  # Default empty DataFrame
            if not trades.empty and "Ticker" in trades.columns:
                symbol_trades = trades[trades["Ticker"] == symbol].copy()

            # Add trade markers for this specific symbol
            if not symbol_trades.empty:
                # Create trade source for this symbol
                symbol_trade_source = ColumnDataSource(
                    dict(
                        entry_bar=symbol_trades["EntryBar"],
                        exit_bar=symbol_trades["ExitBar"],
                        entry_price=symbol_trades["EntryPrice"],
                        exit_price=symbol_trades["ExitPrice"],
                        entry_time=symbol_trades["EntryTime"],
                        exit_time=symbol_trades["ExitTime"],
                        size=symbol_trades["Size"],
                        pnl=symbol_trades["PnL"],
                        return_pct=symbol_trades["Gross%"],
                        returns_positive=(symbol_trades["Gross%"] > 0)
                        .astype(int)
                        .astype(str),
                    )
                )

                # Add position lines (entry to exit)
                symbol_trade_source.add(
                    symbol_trades[["EntryBar", "ExitBar"]].to_numpy().tolist(),
                    "position_lines_xs",
                )
                symbol_trade_source.add(
                    symbol_trades[["EntryPrice", "ExitPrice"]].to_numpy().tolist(),
                    "position_lines_ys",
                )

                # Plot position lines
                ohlc_fig.multi_line(
                    xs="position_lines_xs",
                    ys="position_lines_ys",
                    source=symbol_trade_source,
                    line_color=factor_cmap(
                        "returns_positive", ["red", "green"], ["0", "1"]
                    ),
                    line_width=3,
                    line_alpha=0.8,
                    line_dash="solid",
                )

                # Plot entry markers (triangles pointing up) - colored by trade outcome
                ohlc_fig.scatter(
                    "entry_bar",
                    "entry_price",
                    source=symbol_trade_source,
                    marker="triangle",
                    size=10,
                    color=factor_cmap(
                        "returns_positive", ["red", "green"], ["0", "1"]
                    ),
                    alpha=0.8,
                    legend_label=f"{symbol} Entries",
                )

                # Plot exit markers (triangles pointing down) - colored by trade outcome
                ohlc_fig.scatter(
                    "exit_bar",
                    "exit_price",
                    source=symbol_trade_source,
                    marker="inverted_triangle",
                    size=10,
                    color=factor_cmap(
                        "returns_positive", ["red", "green"], ["0", "1"]
                    ),
                    alpha=0.8,
                    legend_label=f"{symbol} Exits",
                )

            # Phase highlighting (clean delegation)
            if "completion_points" in phase_data:
                from algo.utils.plotting.phase import calculate_phase_ranges_from_completion_points, render_phase_highlights_with_data
                
                # Calculate price range for phase highlighting
                price_max = symbol_data[["High", "Low", "Open", "Close"]].max().max()
                price_min = symbol_data[["High", "Low", "Open", "Close"]].min().min()
                price_range = price_max - price_min
                
                # Calculate phase ranges using the new function
                phase_ranges = calculate_phase_ranges_from_completion_points(
                    phase_data["completion_points"], symbol, symbol_data, symbol_trades
                )
                
                # Render phase highlights using the new function
                render_phase_highlights_with_data(ohlc_fig, phase_ranges, price_max, price_range)
                
                # Also add phase highlighting to volume chart
                if 'volume_fig' in locals():
                    # Calculate volume range for phase highlighting
                    volume_max = symbol_data["Volume"].max()
                    volume_range = volume_max
                    
                    # Render phase highlights on volume chart
                    render_phase_highlights_with_data(volume_fig, phase_ranges, volume_max, volume_range)

            # Plot wedge trend lines if available
            if hasattr(results.get('_strategy'), 'higher_lows_wedge_data'):
                from algo.utils.plotting.phase import plot_wedge_trend_lines
                strategy_obj = results['_strategy']
                wedge_data = strategy_obj.higher_lows_wedge_data.df
                
                # Pass all required data instead of strategy object
                plot_wedge_trend_lines(
                    ohlc_fig, 
                    symbol, 
                    wedge_data, 
                    phase_data,  # Already available in _plotting.py
                )

            # Create unified legend with OHLCV + indicators - do this LAST to override any automatic tooltips
            _create_unified_legend(
                ohlc_fig, symbol, symbol_source, indicators, ohlc_bars
            )

            # Per-symbol autoscale on zoom/pan: compute lows/highs and wire callback to shared x_range
            # Ensure ohlc_low/high exist on the per-symbol source
            ohlc_low_data = symbol_data[["High", "Low"]].min(1).reset_index(drop=True)
            ohlc_high_data = symbol_data[["High", "Low"]].max(1).reset_index(drop=True)

            symbol_source.add(ohlc_low_data, "ohlc_low")
            symbol_source.add(ohlc_high_data, "ohlc_high")

            # Add individual y-axis autoscaling for both OHLC and Volume charts
            symbol_js_args = dict(
                ohlc_range=ohlc_fig.y_range, 
                volume_range=volume_fig.y_range,
                source=symbol_source
            )
            # Clean autoscaling callback for individual symbol charts
            symbol_autoscale_code = """
            // Enforce strict x-axis limits to prevent zooming out beyond data
            let total_bars = source.data['ohlc_high'].length;
            
            // Clamp x-axis range to data boundaries
            if (cb_obj.start < 0) {
                cb_obj.start = 0;
            }
            if (cb_obj.end > total_bars) {
                cb_obj.end = total_bars;
            }
            
            // Prevent zooming out to show less than all data
            let visible_bars = cb_obj.end - cb_obj.start;
            if (visible_bars > total_bars) {
                cb_obj.start = 0;
                cb_obj.end = total_bars;
            }
            
            // Do autoscaling for both OHLC and Volume y-axes
            let i = Math.max(Math.floor(cb_obj.start), 0);
            let j = Math.min(Math.ceil(cb_obj.end), total_bars);
            
            if (i < j && source.data['ohlc_high'].length > 0) {
                // OHLC autoscaling
                let visible_highs = source.data['ohlc_high'].slice(i, j);
                let visible_lows = source.data['ohlc_low'].slice(i, j);
                
                let max = Math.max.apply(null, visible_highs);
                let min = Math.min.apply(null, visible_lows);
                
                if (min !== Infinity && max !== -Infinity) {
                    let pad = (max - min) * 0.03;
                    ohlc_range.start = min - pad;
                    ohlc_range.end = max + pad;
                }
                
                // Volume autoscaling
                if (source.data['Volume'] && source.data['Volume'].length > 0) {
                    let visible_volumes = source.data['Volume'].slice(i, j);
                    let volume_max = Math.max.apply(null, visible_volumes);
                    let volume_min = 0; // Volume always starts from 0
                    
                    if (volume_max > 0) {
                        volume_range.start = volume_min;
                        volume_range.end = volume_max * 1.05; // 5% padding at top
                    }
                }
            }
            """  # noqa: W293

            symbol_autoscale_cb = CustomJS(
                args=symbol_js_args, code=symbol_autoscale_code
            )

            ohlc_fig.x_range.js_on_change("start", symbol_autoscale_cb)
            ohlc_fig.x_range.js_on_change("end", symbol_autoscale_cb)

            # Combine OHLC and Volume figures with minimal spacing
            from bokeh.layouts import column
            combined_symbol_fig = column(
                ohlc_fig, 
                volume_fig, 
                spacing=0,  # Remove spacing between OHLC and volume
                sizing_mode="stretch_width"
            )
            symbol_figs.append(combined_symbol_fig)

        return symbol_figs

    # Construct figure ...

    if plot_equity:
        _plot_broker_session()

    if plot_allocation:
        _plot_equity_stack_section(relative_allocation)

    if plot_return:
        _plot_broker_session(is_return=True)

    if plot_drawdown:
        figs_above_ohlc.append(_plot_drawdown_section())

    if plot_pl:
        figs_above_ohlc.append(_plot_pl_section())

    if plot_volume:
        fig_volume = _plot_volume_section()
        figs_below_ohlc.append(fig_volume)

    if superimpose and is_datetime_index:
        _plot_superimposed_ohlc()
    # Get the first line to set tooltips
    first_line = _plot_top_summary_chart()
    if plot_trades and (not tickers_to_plot or len(tickers_to_plot) <= 10):
        _plot_ohlc_trades()

    if plot_indicator:
        indicator_figs = _plot_indicators()
        if reverse_indicators:
            indicator_figs = indicator_figs[::-1]
        figs_below_ohlc.extend(indicator_figs)

    # Add separate symbol charts
    symbol_figs = []
    if baseline_multi is not None:
        # Extract rising wedge data from indicators if available
        higher_lows_wedge_data = None
        for indicator in indicators:
            if (
                hasattr(indicator, "name")
                and "higher_lows_wedge" in str(indicator.name).lower()
            ):
                higher_lows_wedge_data = indicator.df
                break

        symbol_figs = _plot_separate_symbols()
        figs_below_ohlc.extend(symbol_figs)

    set_tooltips(fig_ohlc, ohlc_tooltips, vline=True, renderers=[first_line])

    source.add(ohlc_extreme_values.min(1), "ohlc_low")
    source.add(ohlc_extreme_values.max(1), "ohlc_high")

    custom_js_args = dict(ohlc_range=fig_ohlc.y_range, source=source)
    if plot_volume:
        custom_js_args.update(volume_range=fig_volume.y_range)

    # Create a safer autoscaling callback that handles missing volume_range
    main_autoscale_code = """
    // Clear any existing timeout
    if (window._bt_autoscale_timeout) {
        clearTimeout(window._bt_autoscale_timeout);
    }
    
    // Set new timeout for autoscaling
    window._bt_autoscale_timeout = setTimeout(function () {
        let i = Math.max(Math.floor(cb_obj.start), 0);
        let j = Math.min(Math.ceil(cb_obj.end), source.data['ohlc_high'].length);
        
        if (i < j && source.data['ohlc_high'].length > 0) {
            let max = Math.max.apply(null, source.data['ohlc_high'].slice(i, j));
            let min = Math.min.apply(null, source.data['ohlc_low'].slice(i, j));
            
            if (min !== Infinity && max !== -Infinity) {
                let pad = (max - min) * 0.03;
                ohlc_range.start = min - pad;
                ohlc_range.end = max + pad;
            }
        }
        
        // Only update volume range if it exists
        if (typeof volume_range !== 'undefined' && volume_range) {
            if (i < j && source.data['Volume'].length > 0) {
                let max = Math.max.apply(null, source.data['Volume'].slice(i, j));
                volume_range.start = 0;
                volume_range.end = max * 1.03;
            }
        }
    }, 50);
    """  # noqa: W293

    const_cb = CustomJS(args=custom_js_args, code=main_autoscale_code)
    fig_ohlc.x_range.js_on_change("start", const_cb)
    fig_ohlc.x_range.js_on_change("end", const_cb)

    # Kick once so initial y-range reflects current view
    try:
        fig_ohlc.x_range.start += 1e-9
        fig_ohlc.x_range.start -= 1e-9
    except Exception:
        pass

    plots = figs_above_ohlc + [fig_ohlc] + figs_below_ohlc

    # Always use full width with minimal padding
    kwargs = {
        "sizing_mode": "stretch_width",  # Always stretch to full width
    }

    # Create the main gridplot
    main_plot = gridplot(
        plots,
        ncols=1,
        toolbar_location="right",
        merge_tools=True,
        **kwargs,  # type: ignore
    )

    # Wrap in a layout with minimal padding for full width
    from bokeh.layouts import column

    # Apply minimal margins for full width display
    layout_kwargs = {
        'width_policy': "max",  
        'sizing_mode': "stretch_width",  # Ensure full width
        'margin': (0, 5, 0, 5),  # Minimal 5px padding on sides only
    }
    
    # Apply theme background to layout if available
    if THEMES_AVAILABLE:
        theme = get_current_theme()
        layout_kwargs['background'] = theme.background_fill_color

    fig = column(main_plot, **layout_kwargs)
    # Persist and/or open depending on configuration
    if processed_filename:
        # Set up Bokeh output configuration first
        _bokeh_reset(processed_filename)

        # Save to file
        import os

        from bokeh.io import save

        dir_path = os.path.dirname(processed_filename)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        save(fig, processed_filename)
        
        # Inject theme-aware CSS to fix body background
        _inject_theme_css(processed_filename)

        # Open if requested (avoid Bokeh show double-write)
        if open_browser:
            _open_html_external(processed_filename)
    else:
        # No filename configured; only open if requested using Bokeh
        if open_browser:
            try:
                # Initialize Bokeh output for an unnamed session
                _bokeh_reset("")
                show(fig, browser=_get_bokeh_browser_arg(True))
            except Exception:
                pass
    return fig


def plot_heatmaps(
    heatmap: pd.Series,
    agg: Union[Callable, str],
    ncols: int,
    filename: str = "",
    plot_width: int = 1200,
    open_browser: bool = True,
):
    if not (
        isinstance(heatmap, pd.Series) and isinstance(heatmap.index, pd.MultiIndex)
    ):
        raise ValueError(
            "heatmap must be heatmap Series as returned by "
            "`Backtest.optimize(..., return_heatmap=True)`"
        )

    # Only call _bokeh_reset if we have a valid filename
    if filename and filename != "":
        # Get the processed filename (with .html extension if needed)
        processed_filename = _bokeh_reset(filename)
    else:
        processed_filename = ""

    param_combinations = combinations(heatmap.index.names, 2)
    dfs = [
        heatmap.groupby(list(dims)).agg(agg).to_frame(name="_Value")
        for dims in param_combinations
    ]
    plots = []
    # Get theme-aware palette and colors
    if THEMES_AVAILABLE:
        palette = get_themed_heatmap_palette()
        nan_color = get_themed_nan_color()
    else:
        palette = "Viridis256"
        nan_color = "white"
    
    cmap = LinearColorMapper(
        palette=palette,
        low=min(df.min().min() for df in dfs),
        high=max(df.max().max() for df in dfs),
        nan_color=nan_color,
    )
    for df in dfs:
        name1, name2 = df.index.names
        level1 = df.index.levels[0].astype(str).tolist()
        level2 = df.index.levels[1].astype(str).tolist()
        df = df.reset_index()
        df[name1] = df[name1].astype("str")
        df[name2] = df[name2].astype("str")

        fig = _figure(
            x_range=level1,
            y_range=level2,
            x_axis_label=name1,
            y_axis_label=name2,
            width=plot_width // ncols,
            height=plot_width // ncols,
            tools="box_zoom,reset,save",
            tooltips=[
                (name1, "@" + name1),
                (name2, "@" + name2),
                ("Value", "@_Value{0.[000]}"),
            ],
        )
        fig.grid.grid_line_color = None
        fig.axis.axis_line_color = None
        fig.axis.major_tick_line_color = None
        fig.axis.major_label_standoff = 0

        fig.rect(
            x=name1,
            y=name2,
            width=1,
            height=1,
            source=df,
            line_color=None,
            fill_color=dict(field="_Value", transform=cmap),
        )
        fig.toolbar.logo = None
        plots.append(fig)

    fig = gridplot(
        plots,  # type: ignore
        ncols=ncols,
        toolbar_location="above",
        merge_tools=True,
    )

    # Persist and/or open depending on configuration
    if processed_filename:
        # Save to file
        import os

        from bokeh.io import save

        dir_path = os.path.dirname(processed_filename)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        save(fig, processed_filename)
        
        # Inject theme-aware CSS to fix body background
        _inject_theme_css(processed_filename)

        # Open if requested (avoid Bokeh show double-write)
        if open_browser:
            _open_html_external(processed_filename)
    else:
        # No filename configured; only open if requested using Bokeh
        if open_browser:
            try:
                # Initialize Bokeh output for an unnamed session
                _bokeh_reset("")
                show(fig, browser=_get_bokeh_browser_arg(True))
            except Exception:
                pass
    return fig