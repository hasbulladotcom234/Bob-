"""Glue between the plumbing and your strategy file (my_strategy.py).

Nothing in here makes trading decisions; it loads your file, runs your
indicators, and checks them for a common mistake (lookahead).
"""
import importlib

import pandas as pd


def load_strategy(module_name: str = "my_strategy"):
    """Import the strategy file. It must define PARAMS, indicators() and decide()."""
    strat = importlib.import_module(module_name)
    for name in ("PARAMS", "indicators", "decide"):
        if not hasattr(strat, name):
            raise AttributeError(f"{module_name}.py is missing `{name}`")
    return strat


def with_indicators(candles: pd.DataFrame, strat, params: dict) -> pd.DataFrame:
    out = strat.indicators(candles.copy().reset_index(drop=True), params)
    if not isinstance(out, pd.DataFrame):
        raise TypeError("indicators() must return the DataFrame")
    return out


class Bars:
    """The rows of an indicator DataFrame, prepared for fast bar-by-bar access.
    Each bar is a namedtuple, so the strategy writes bar.close, bar.rsi, ..."""

    def __init__(self, data: pd.DataFrame):
        self.rows = list(data.itertuples(index=False))
        self.ready = (~data.isna().any(axis=1)).tolist()  # False while indicators warm up

    def __len__(self):
        return len(self.rows)


def decide(bars: Bars, i: int, in_position: bool, strat, params: dict) -> str:
    """Ask the strategy what to do at the close of bar i. Bars where any
    indicator is still NaN (warming up) are automatically 'hold'."""
    if i < 1 or not (bars.ready[i] and bars.ready[i - 1]):
        return "hold"
    action = strat.decide(bars.rows[i], bars.rows[i - 1], in_position, params)
    if action not in ("buy", "sell", "hold"):
        raise ValueError(f"decide() returned {action!r}; must be 'buy', 'sell' or 'hold'")
    return action


def lookahead_columns(candles: pd.DataFrame, strat, params: dict) -> list:
    """Return indicator columns that change when future bars are added.

    A legit indicator's value at bar t only depends on bars <= t, so
    computing it on the first half of the data or on all of it must give the
    same numbers for the first half. If not, it's peeking at the future
    (e.g. .shift(-1) or a centered rolling window) and backtests will lie.
    """
    cut = len(candles) // 2
    if cut < 2:
        return []
    full = with_indicators(candles, strat, params).iloc[:cut]
    part = with_indicators(candles.iloc[:cut], strat, params)
    bad = []
    for col in full.columns:
        if col not in part.columns:
            continue
        a, b = full[col], part[col]
        if pd.api.types.is_numeric_dtype(a):
            same = ((a - b).abs() <= 1e-9 * (1 + b.abs())) | (a.isna() & b.isna())
        else:
            same = (a == b) | (a.isna() & b.isna())
        if not same.all():
            bad.append(col)
    return bad
