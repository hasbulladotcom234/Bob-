"""Trend-following momentum strategy.

Entry: fast EMA crosses above slow EMA while RSI confirms bullish momentum.
Exit: fast EMA crosses below slow EMA, or a stop-loss / take-profit is hit
(stop-loss and take-profit are enforced by the risk manager / broker, not here).

This is a simple, well-studied approach (EMA crossover + momentum filter).
It does not predict tops or bottoms -- it reacts to confirmed trend changes,
which means it will always give back some gain at the top and enter after
the bottom. That lag is the cost of not needing to predict the future.
"""
from dataclasses import dataclass

import pandas as pd

from .config import StrategyParams
from .indicators import ema, rsi


@dataclass
class Signal:
    long_entry: bool
    long_exit: bool


def compute_indicators(df: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
    out = df.copy()
    out["ema_fast"] = ema(out["close"], params.fast_ema)
    out["ema_slow"] = ema(out["close"], params.slow_ema)
    out["rsi"] = rsi(out["close"], params.rsi_period)
    return out


def generate_signals(df: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
    """Given OHLCV data with a 'close' column, return the dataframe with
    ema_fast, ema_slow, rsi, and boolean long_entry / long_exit columns.
    """
    out = compute_indicators(df, params)

    prev_fast = out["ema_fast"].shift(1)
    prev_slow = out["ema_slow"].shift(1)

    crossed_up = (prev_fast <= prev_slow) & (out["ema_fast"] > out["ema_slow"])
    crossed_down = (prev_fast >= prev_slow) & (out["ema_fast"] < out["ema_slow"])

    out["long_entry"] = crossed_up & (out["rsi"] > params.rsi_bull_threshold)
    out["long_exit"] = crossed_down

    out["long_entry"] = out["long_entry"].fillna(False)
    out["long_exit"] = out["long_exit"].fillna(False)
    return out
