import warnings
from typing import TYPE_CHECKING, List, Union

import numpy as np
import pandas as pd

from ._util import _data_period

if TYPE_CHECKING:
    from .backtesting import Order, Strategy, Trade


def compute_drawdown_duration_peaks(dd: pd.Series):
    iloc = np.unique(np.r_[(dd == 0).values.nonzero()[0], len(dd) - 1])
    iloc = pd.Series(iloc, index=dd.index[iloc])
    df = iloc.to_frame("iloc").assign(prev=iloc.shift())
    df = df[df["iloc"] > df["prev"] + 1].astype(int)

    # If no drawdown since no trade, avoid below for pandas sake and return nan series
    if not len(df):
        return (dd.replace(0, np.nan),) * 2

    df["duration"] = df["iloc"].map(dd.index.__getitem__) - df["prev"].map(
        dd.index.__getitem__
    )
    df["peak_dd"] = df.apply(
        lambda row: dd.iloc[row["prev"] : row["iloc"] + 1].max(), axis=1
    )
    df = df.reindex(dd.index)
    return df["duration"], df["peak_dd"]


def geometric_mean(returns: pd.Series) -> float:
    returns = returns.fillna(0) + 1
    if np.any(returns <= 0):
        return 0
    return np.exp(np.log(returns).sum() / (len(returns) or np.nan)) - 1


def _prepare_orders_dataframe(orders):
    """Convert orders to DataFrame format if needed."""
    if isinstance(orders, pd.DataFrame):
        return orders

    return pd.DataFrame(
        {
            "SignalTime": [t.entry_time for t in orders],
            "Ticker": [t.ticker for t in orders],
            "Side": ["Buy" if t.size > 0 else "Sell" for t in orders],
            "Size": [int(t.size) for t in orders],
        }
    ).set_index("SignalTime")


def _prepare_trades_dataframe(trades):
    """Convert trades to DataFrame format if needed."""
    if isinstance(trades, pd.DataFrame):
        return trades

    # Came straight from Backtest.run()
    trades_df = pd.DataFrame(
        {
            "EntryBar": [t.entry_bar for t in trades],
            "ExitBar": [t.exit_bar for t in trades],
            "Ticker": [t.ticker for t in trades],
            "Size": [t.size for t in trades],
            "EntryPrice": [t.entry_price for t in trades],
            "ExitPrice": [t.exit_price for t in trades],
            "PnL": [t.pl for t in trades],
            "Gross%": [t.pl_pct for t in trades],
            "EntryTime": [t.entry_time for t in trades],
            "ExitTime": [t.exit_time for t in trades],
            "CommissionTotal": [
                getattr(t, "commission_total", 0.0) or 0.0 for t in trades
            ],
        }
    )
    trades_df["Duration"] = trades_df["ExitTime"] - trades_df["EntryTime"]
    # Net PnL after commissions when available
    if "CommissionTotal" in trades_df:
        trades_df["NetPnL"] = trades_df["PnL"] - trades_df["CommissionTotal"]

    return trades_df


def _calculate_period_returns(equity_df, index, trade_start_bar):
    """Calculate period returns and trading periods based on data frequency."""
    gmean_period_return = 0
    period_returns = np.array(np.nan)
    annual_trading_periods = np.nan

    if isinstance(index, pd.DatetimeIndex):
        period = equity_df.index.to_series().diff().mean().days

        if period <= 1:  # Daily data
            period_returns = (
                equity_df["Equity"]
                .iloc[trade_start_bar:]
                .resample("D")
                .last()
                .dropna()
                .pct_change()
            )
            gmean_period_return = geometric_mean(period_returns)
            annual_trading_periods = float(
                365
                if index.dayofweek.to_series().between(5, 6).mean() > 2 / 7 * 0.6
                else 252
            )
        elif 28 <= period <= 31:  # Monthly data
            period_returns = equity_df["Equity"].iloc[trade_start_bar:].pct_change()
            gmean_period_return = geometric_mean(period_returns)
            annual_trading_periods = 12
        elif 365 <= period <= 366:  # Yearly data
            period_returns = equity_df["Equity"].iloc[trade_start_bar:].pct_change()
            gmean_period_return = geometric_mean(period_returns)
            annual_trading_periods = 1
        else:
            warnings.warn(f"Unsupported data period from index: {period} days.")

    return gmean_period_return, period_returns, annual_trading_periods


def _calculate_risk_metrics(
    gmean_period_return,
    period_returns,
    annual_trading_periods,
    annualized_return,
    risk_free_rate,
):
    """Calculate various risk metrics including Sharpe, Sortino, etc."""
    metrics = {}

    # Volatility calculation
    volatility = (
        np.sqrt(
            (
                period_returns.var(ddof=int(bool(period_returns.shape)))
                + (1 + gmean_period_return) ** 2
            )
            ** annual_trading_periods
            - (1 + gmean_period_return) ** (2 * annual_trading_periods)
        )
        * 100
    )
    metrics["Volatility (Ann.) %"] = volatility

    # Sharpe Ratio
    metrics["Sharpe Ratio"] = (annualized_return * 100 - risk_free_rate) / (
        volatility or np.nan
    )

    # Sortino Ratio

    with warnings.catch_warnings():
        warnings.filterwarnings("error")
        try:
            # Calculate daily risk-free rate
            daily_rf = risk_free_rate / annual_trading_periods
            # Use risk-free rate as downside threshold
            metrics["Sortino Ratio"] = (annualized_return - risk_free_rate) / (
                np.sqrt(np.mean((period_returns - daily_rf).clip(-np.inf, 0) ** 2))
                * np.sqrt(annual_trading_periods)
            )
        except Warning:
            metrics["Sortino Ratio"] = np.nan

    return metrics


def _calculate_trade_metrics(trades_df, pl, pl_net, returns):
    """Calculate trade-related metrics."""
    metrics = {}
    n_trades = len(trades_df)

    metrics["# Trades"] = n_trades
    win_rate = np.nan if not n_trades else (pl > 0).mean()
    metrics["Win Rate %"] = win_rate * 100
    metrics["Best Trade %"] = returns.max() * 100
    metrics["Worst Trade %"] = returns.min() * 100
    metrics["Avg. Trade %"] = geometric_mean(returns) * 100
    metrics["Profit Factor"] = returns[returns > 0].sum() / (
        abs(returns[returns < 0].sum()) or np.nan
    )
    metrics["Expectancy %"] = returns.mean() * 100
    metrics["SQN"] = np.sqrt(n_trades) * pl.mean() / (pl.std() or np.nan)
    metrics["Kelly Criterion"] = win_rate - (1 - win_rate) / (
        pl[pl > 0].mean() / -pl[pl < 0].mean()
    )

    # Commission metrics
    if "CommissionTotal" in trades_df.columns:
        metrics["Total Commission [$]"] = float(trades_df["CommissionTotal"].sum())
        metrics["Profit Factor (Net)"] = pl_net[pl_net > 0].sum() / (
            abs(pl_net[pl_net < 0].sum()) or np.nan
        )

    return metrics


def compute_stats(
    orders: Union[List["Order"], pd.DataFrame],
    trades: Union[List["Trade"], pd.DataFrame],
    equity: pd.DataFrame,
    ohlc_data: pd.DataFrame,
    strategy_instance: "Strategy",
    risk_free_rate: float = 0,
    positions: dict = None,
    trade_start_bar: int = 0,
) -> pd.Series:
    assert -1 < risk_free_rate < 1

    index = ohlc_data.index
    dd = 1 - equity["Equity"] / np.maximum.accumulate(equity["Equity"])
    dd_dur, dd_peaks = compute_drawdown_duration_peaks(pd.Series(dd, index=index))

    # Prepare DataFrames
    orders_df = _prepare_orders_dataframe(orders)
    trades_df = _prepare_trades_dataframe(trades)
    del trades  # Free memory

    equity_df = pd.concat(
        [
            equity,
            pd.DataFrame({"DrawdownPct": dd, "DrawdownDuration": dd_dur}, index=index),
        ],
        axis=1,
    )

    # Extract key metrics
    pl = trades_df["PnL"]
    pl_net = trades_df["NetPnL"] if "NetPnL" in trades_df.columns else pl
    returns = trades_df["Gross%"]
    durations = trades_df["Duration"]

    def _round_timedelta(value, _period=_data_period(index)):
        if not isinstance(value, pd.Timedelta):
            return value
        resolution = getattr(_period, "resolution_string", None) or _period.resolution
        return value.ceil(resolution)

    # Initialize results series
    s = pd.Series(dtype=object)
    s.loc["Start"] = index[0]
    s.loc["End"] = index[-1]
    s.loc["Duration"] = s.End - s.Start

    # Calculate exposure time
    have_position = np.repeat(0, len(index))
    for t in trades_df.itertuples(index=False):
        have_position[t.EntryBar : t.ExitBar + 1] = 1

    s.loc["Exposure Time %"] = have_position.mean() * 100
    s.loc["Equity Final [$]"] = equity["Equity"].iloc[-1]
    s.loc["Equity Peak [$]"] = equity["Equity"].max()
    s.loc["Return %"] = (
        (equity["Equity"].iloc[-1] - equity["Equity"].iloc[0])
        / equity["Equity"].iloc[0]
        * 100
    )

    # Calculate period returns and annualized metrics
    gmean_period_return, period_returns, annual_trading_periods = (
        _calculate_period_returns(equity_df, index, trade_start_bar)
    )

    annualized_return = (1 + gmean_period_return) ** annual_trading_periods - 1
    s.loc["Return (Ann.) %"] = annualized_return * 100

    # Calculate risk metrics
    risk_metrics = _calculate_risk_metrics(
        gmean_period_return,
        period_returns,
        annual_trading_periods,
        annualized_return,
        risk_free_rate,
    )
    for key, value in risk_metrics.items():
        s.loc[key] = value

    # Drawdown metrics
    max_dd = -np.nan_to_num(dd.max())
    s.loc["Calmar Ratio"] = annualized_return / (-max_dd or np.nan)
    s.loc["Max. Drawdown %"] = max_dd * 100
    s.loc["Avg. Drawdown %"] = -dd_peaks.mean() * 100
    s.loc["Max. Drawdown Duration"] = _round_timedelta(dd_dur.max())
    s.loc["Avg. Drawdown Duration"] = _round_timedelta(dd_dur.mean())

    # Duration metrics
    s.loc["Max. Trade Duration"] = _round_timedelta(durations.max())
    s.loc["Avg. Trade Duration"] = _round_timedelta(durations.mean())

    # Trade metrics
    trade_metrics = _calculate_trade_metrics(trades_df, pl, pl_net, returns)
    for key, value in trade_metrics.items():
        s.loc[key] = value

    # Store reference data
    s.loc["_strategy"] = strategy_instance
    s.loc["_equity_curve"] = equity_df
    s.loc["_trades"] = trades_df
    s.loc["_orders"] = orders_df
    s.loc["_positions"] = positions
    s.loc["_trade_start_bar"] = trade_start_bar

    return _Stats(s)


class _Stats(pd.Series):
    def __repr__(self):
        # Prevent expansion due to Equity and _trades dfs
        with pd.option_context("max_colwidth", 20):
            return super().__repr__()
