"""Removing bad prices ("bad prints") from candle data.

The raw data in the store is never modified. Cleaning happens when data is
loaded, so the rules can be improved later and re-applied to everything.

What counts as a bad print: a price far from BOTH the bars just before and
just after it. A real move stays (the following bars sit near the new
level); a bad print snaps back. "Far" adapts to how volatile the coin is at
that time, so a normal 10% hour for a meme coin in a frenzy isn't flagged,
but the same move on a quiet day is.

Three kinds are handled:
- bad close: the whole bar is wrong -> the bar is dropped (treated as missing)
- bad open:  only the opening price is off -> replaced by the previous close
- bad wick:  only the high or low is off (e.g. a stray trade at a silly
             price) -> clipped to the neighbouring bars' range, so backtests
             don't trigger stop-losses on prices nobody could trade at

This uses bars on both sides of each bar, i.e. it looks at the "future".
That's standard for cleaning history (a live trader would also have seen
the price snap back within an hour), but it means a strategy that would
have reacted to a bad print in real time looks slightly better in a
backtest than it would have live.
"""
import numpy as np
import pandas as pd

Z = 10.0            # how many "typical moves" away a price must be...
MIN_MOVE = 0.10     # ...and never less than 10%, to count as a bad print
NEIGHBOURS = 3      # bars on each side used as the reference level
VOL_WINDOW = 720    # bars used to measure how volatile the coin is right now (30 days of 1h bars)


def local_volatility(log_close: pd.Series) -> pd.Series:
    """Typical one-bar move around each bar: a rolling, outlier-resistant
    standard deviation (median absolute deviation x 1.4826)."""
    r = log_close.diff()
    min_p = max(VOL_WINDOW // 10, 5)
    center = r.rolling(VOL_WINDOW, center=True, min_periods=min_p).median()
    mad = (r - center).abs().rolling(VOL_WINDOW, center=True, min_periods=min_p).median()
    overall = (r - r.median()).abs().median()
    return (1.4826 * mad).fillna(1.4826 * overall).fillna(0.0)


def _side_medians(x: pd.Series, n: int = NEIGHBOURS):
    before = x.shift(1).rolling(n, min_periods=1).median()
    after = x.iloc[::-1].shift(1).rolling(n, min_periods=1).median().iloc[::-1]
    return before, after


def clean_bars(df: pd.DataFrame):
    """Return (cleaned candles, report). The report has counts and a table
    of every change made."""
    report = {"bad_closes": 0, "bad_opens": 0, "bad_wicks": 0, "changes": pd.DataFrame()}
    if len(df) < 2 * NEIGHBOURS + 1:
        return df.copy(), report

    df = df.reset_index(drop=True)
    changes = []

    # 1. bad closes: the bar is far from the level both before and after it
    c = np.log(df["close"])
    before, after = _side_medians(c)
    threshold = np.maximum(Z * local_volatility(c), MIN_MOVE)
    up = (c - before > threshold) & (c - after > threshold)
    down = (before - c > threshold) & (after - c > threshold)
    bad_close = (up | down).fillna(False)
    for i in np.flatnonzero(bad_close):
        changes.append({"timestamp": df.at[i, "timestamp"], "kind": "bad close (bar dropped)",
                        "raw": df.at[i, "close"], "fixed_to": np.nan, "level_before": np.exp(before[i]),
                        "level_after": np.exp(after[i])})
    out = df[~bad_close].reset_index(drop=True)
    report["bad_closes"] = int(bad_close.sum())

    # 2. bad opens: open far from both the previous close and this bar's close
    c = np.log(out["close"])
    o = np.log(out["open"])
    prev_c = c.shift(1)
    threshold = np.maximum(Z * local_volatility(c), MIN_MOVE)
    bad_open = (((o - prev_c > threshold) & (o - c > threshold)) |
                ((prev_c - o > threshold) & (c - o > threshold))).fillna(False)
    for i in np.flatnonzero(bad_open):
        changes.append({"timestamp": out.at[i, "timestamp"], "kind": "bad open",
                        "raw": out.at[i, "open"], "fixed_to": np.exp(prev_c[i]),
                        "level_before": np.exp(prev_c[i]), "level_after": out.at[i, "close"]})
    out.loc[bad_open, "open"] = np.exp(prev_c[bad_open])
    report["bad_opens"] = int(bad_open.sum())

    # 3. bad wicks: high/low far outside this bar's body AND the neighbours' range
    o, c = out["open"], out["close"]
    body_hi, body_lo = np.maximum(o, c), np.minimum(o, c)
    h, l = out["high"], out["low"]
    neigh_hi = pd.concat([h.shift(1), h.shift(-1)], axis=1).max(axis=1)
    neigh_lo = pd.concat([l.shift(1), l.shift(-1)], axis=1).min(axis=1)
    threshold = np.maximum(Z * local_volatility(np.log(c)), MIN_MOVE)
    bad_hi = ((np.log(h / body_hi) > threshold) & (np.log(h / neigh_hi) > threshold)).fillna(False)
    bad_lo = ((np.log(body_lo / l) > threshold) & (np.log(neigh_lo / l) > threshold)).fillna(False)
    new_hi = np.maximum(body_hi, neigh_hi)
    new_lo = np.minimum(body_lo, neigh_lo)
    for i in np.flatnonzero(bad_hi):
        changes.append({"timestamp": out.at[i, "timestamp"], "kind": "bad high", "raw": h[i],
                        "fixed_to": new_hi[i], "level_before": out["close"].shift(1)[i], "level_after": c[i]})
    for i in np.flatnonzero(bad_lo):
        changes.append({"timestamp": out.at[i, "timestamp"], "kind": "bad low", "raw": l[i],
                        "fixed_to": new_lo[i], "level_before": out["close"].shift(1)[i], "level_after": c[i]})
    out.loc[bad_hi, "high"] = new_hi[bad_hi]
    out.loc[bad_lo, "low"] = new_lo[bad_lo]
    report["bad_wicks"] = int(bad_hi.sum() + bad_lo.sum())

    if changes:
        report["changes"] = pd.DataFrame(changes).sort_values("timestamp").reset_index(drop=True)
    return out, report


def gaps(df: pd.DataFrame, timeframe_seconds: int, min_bars: int = 24) -> pd.DataFrame:
    """Stretches of at least `min_bars` missing bars, with the price on each side."""
    if len(df) < 2:
        return pd.DataFrame(columns=["from", "to", "missing_bars", "price_before", "price_after", "change_pct"])
    step = pd.Timedelta(seconds=timeframe_seconds)
    missing = (df["timestamp"].diff() / step - 1).fillna(0).astype(int)
    rows = []
    for i in np.flatnonzero(missing >= min_bars):
        before, after = df["close"].iloc[i - 1], df["close"].iloc[i]
        rows.append({"from": df["timestamp"].iloc[i - 1], "to": df["timestamp"].iloc[i],
                     "missing_bars": int(missing.iloc[i]), "price_before": before, "price_after": after,
                     "change_pct": 100 * (after / before - 1)})
    return pd.DataFrame(rows)
