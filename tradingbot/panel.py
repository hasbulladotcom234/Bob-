"""Multi-asset views of the store: one column per symbol, one row per bar.

This is the shape cross-asset research works in: "on each bar, rank every
coin by X" is one line on a panel.
"""
import pandas as pd


def load_panel(store, symbols, timeframe: str, field: str = "close", start=None, end=None) -> pd.DataFrame:
    """Wide DataFrame of `field` on a complete, regular time grid.

    Bars missing because nothing traded are filled: close/open/high/low carry
    the last price forward, volume is 0. Before a coin's first stored bar the
    value stays NaN (it wasn't trading or we have no data), so it can't leak
    into results.
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
    if field == "volume":
        first_valid = panel.notna().cummax()
        panel = panel.fillna(0).where(first_valid)
    else:
        panel = panel.ffill()
    panel.index.name = "timestamp"
    return panel


def returns(close_panel: pd.DataFrame) -> pd.DataFrame:
    """Simple per-bar returns. NaN where the coin wasn't trading yet."""
    return close_panel.pct_change(fill_method=None)


def benchmarks(close_panel: pd.DataFrame, btc: str = "BTC/USD") -> pd.DataFrame:
    """Growth of $1 in the two yardsticks every strategy must beat:
    - btc_buy_hold: just hold bitcoin
    - equal_weight: hold every coin in equal amounts, rebalanced every bar
      (an idealised, cost-free basket; coins join when their data starts)
    """
    r = returns(close_panel)
    out = pd.DataFrame(index=close_panel.index)
    if btc in r:
        out["btc_buy_hold"] = (1 + r[btc].fillna(0)).cumprod()
    out["equal_weight"] = (1 + r.mean(axis=1, skipna=True).fillna(0)).cumprod()
    return out
