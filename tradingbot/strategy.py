"""Trend-following momentum strategy.

Entry: fast EMA crosses above slow EMA while RSI confirms bullish momentum.
Exit: fast EMA crosses below slow EMA, or a stop-loss / take-profit is hit
(stop-loss and take-profit are enforced by the risk manager / broker, not here).

This is a simple, well-studied approach (EMA crossover + momentum filter).
It does not predict tops or bottoms -- it reacts to confirmed trend changes,
which means it will always give back some gain at the top and enter after
the bottom. That lag is the cost of not needing to predict the future.

Stochastic filter (optional, on by default): each crossover is also priced
as a first-passage problem. Recent drift and volatility give the
probability that the take-profit is hit before the stop-loss. The entry is
skipped when that bracket's expected return, after fees and slippage, is
below min_edge_pct. See stochastic.py.
"""
from dataclasses import dataclass

import pandas as pd

from .config import StrategyParams
from .indicators import ema, rsi
from .stochastic import bracket_edge, ewma_drift_vol, expected_exit_bars, take_profit_probability


@dataclass
class Signal:
    long_entry: bool
    long_exit: bool


def compute_indicators(df: pd.DataFrame, params: StrategyParams,
                       round_trip_cost: float = 0.0) -> pd.DataFrame:
    out = df.copy()
    out["ema_fast"] = ema(out["close"], params.fast_ema)
    out["ema_slow"] = ema(out["close"], params.slow_ema)
    out["rsi"] = rsi(out["close"], params.rsi_period)

    drift, vol = ewma_drift_vol(out["close"], params.stoch_lookback, params.drift_shrinkage)
    out["drift"] = drift
    out["vol"] = vol
    out["p_take"] = take_profit_probability(drift, vol, params.stop_loss_pct, params.take_profit_pct)
    out["expected_bars"] = expected_exit_bars(drift, vol, params.stop_loss_pct, params.take_profit_pct)
    out["edge"] = bracket_edge(out["p_take"], params.stop_loss_pct, params.take_profit_pct, round_trip_cost)
    return out


def generate_signals(df: pd.DataFrame, params: StrategyParams,
                     round_trip_cost: float = 0.0) -> pd.DataFrame:
    """Given OHLCV data with a 'close' column, return the dataframe with
    ema_fast, ema_slow, rsi, drift, vol, p_take, expected_bars, edge, and
    boolean long_entry / long_exit columns. round_trip_cost is the fraction
    lost to fees and slippage entering and exiting a position.
    """
    out = compute_indicators(df, params, round_trip_cost)

    prev_fast = out["ema_fast"].shift(1)
    prev_slow = out["ema_slow"].shift(1)

    crossed_up = (prev_fast <= prev_slow) & (out["ema_fast"] > out["ema_slow"])
    crossed_down = (prev_fast >= prev_slow) & (out["ema_fast"] < out["ema_slow"])

    out["long_entry"] = crossed_up & (out["rsi"] > params.rsi_bull_threshold)
    if params.use_stochastic_filter:
        out["long_entry"] &= out["edge"] > params.min_edge_pct
    out["long_exit"] = crossed_down

    out["long_entry"] = out["long_entry"].fillna(False)
    out["long_exit"] = out["long_exit"].fillna(False)
    return out
