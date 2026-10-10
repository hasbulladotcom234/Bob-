"""Multi-asset views of the store: one column per symbol, one row per bar.

This is the shape cross-asset research works in: "on each bar, rank every
coin by X" is one line on a panel.
"""
import pandas as pd


def load_panel(store, symbols, timeframe: str, field: str = "close", start=None, end=None,
               max_fill_bars: int = 24) -> pd.DataFrame:
    """Wide DataFrame of `field` (cleaned data) on a complete, regular time grid.

    Short holes (up to max_fill_bars, usually "nobody traded that hour") are
    filled: prices carry the last value forward, volume is 0. Longer holes
    are data outages and stay NaN, so no return is ever computed across
    them. Before a coin's first bar and after its last (e.g. delisted) the
    value is NaN too, so a dead coin can't masquerade as a flat one.
    """
    series = {}
    for sym in symbols:
        df = store.load(sym, timeframe, start, end)
        if not df.empty:
            series[sym] = df.set_index("timestamp")[field]
    if not series:
        return pd.DataFrame()
    panel = pd.DataFrame(series)
    freq = pd.Series(panel.index).diff().min()
    grid = pd.date_range(panel.index.min(), panel.index.max(), freq=freq)
    panel = panel.reindex(grid)
    has = panel.notna()
    alive = has.cummax() & has.iloc[::-1].cummax().iloc[::-1]       # between first and last bar
    short_hole = panel.ffill(limit=max_fill_bars).notna()
    if field == "volume":
        panel = panel.fillna(0).where(short_hole & alive)
    else:
        panel = panel.ffill(limit=max_fill_bars).where(alive)
    panel.index.name = "timestamp"
    return panel


def to_daily(close_panel: pd.DataFrame) -> pd.DataFrame:
    """Daily closes (the last price of each UTC day)."""
    return close_panel.resample("1D").last()


def returns(close_panel: pd.DataFrame) -> pd.DataFrame:
    """Simple per-bar returns. NaN where the coin wasn't trading yet."""
    return close_panel.pct_change(fill_method=None)


def benchmarks(close_panel: pd.DataFrame, btc: str = "BTC/USD") -> pd.DataFrame:
    """Growth of $1 in the two yardsticks every strategy must beat:
    - btc_buy_hold: just hold bitcoin
    - equal_weight: hold every coin in equal amounts, rebalanced every bar
      (an idealised, cost-free basket; coins join when their data starts).
      Pass daily closes: rebalancing every hour would harvest noise.
    """
    r = returns(close_panel)
    out = pd.DataFrame(index=close_panel.index)
    if btc in r:
        out["btc_buy_hold"] = (1 + r[btc].fillna(0)).cumprod()
    out["equal_weight"] = (1 + r.mean(axis=1, skipna=True).fillna(0)).cumprod()
    return out
