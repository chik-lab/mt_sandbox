import warnings
import logging

# Suppress Streamlit warnings when running in non-Streamlit environment
warnings.filterwarnings("ignore", category=UserWarning, module="streamlit")
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")
warnings.filterwarnings("ignore", message=".*Thread.*MainThread.*missing ScriptRunContext.*")

# Also suppress at logging level
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)

from .core import Backtest, Strategy
from .utils import (
    backtest_strategy_on_portfolios,
    backtest_strategy_parameters,
    calculate_positions,
    calculate_trade_stats,
    generate_random_portfolios,
    plot_heatmap,
    shuffle_ohlcv,
)

__all__ = [
    "Backtest",
    "Strategy",
    "generate_random_portfolios",
    "backtest_strategy_parameters",
    "backtest_strategy_on_portfolios",
    "plot_heatmap",
    "calculate_positions",
    "calculate_trade_stats",
    "shuffle_ohlcv",
]
