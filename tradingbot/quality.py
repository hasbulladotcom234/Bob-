"""Data quality checks. Bad data produces fake signals, so check before trusting.

Missing bars are common for smaller coins on Alpaca: a bar only exists if
something traded in that interval. That's a property of the market, not an
error, but it matters (thin trading = wide spreads = expensive to trade).
"""
import time

import numpy as np
import pandas as pd

from .cleaning import clean_bars

LONG_GAP_BARS = 24        # holes longer than this are data outages, not quiet trading
DELISTED_AFTER_DAYS = 7   # no new bars for this long: probably delisted or halted


def check_bars(df: pd.DataFrame, timeframe_seconds: int, now: float = None) -> dict:
    """Quality metrics for one symbol's candles (sorted, bar start timestamps)."""
    now = time.time() if now is None else now
    if df.empty:
        return {"rows": 0, "status": "bad", "issues": "no data"}

    ts = df["timestamp"]
    step = pd.Timedelta(seconds=timeframe_seconds)
    expected = int((ts.iloc[-1] - ts.iloc[0]) / step) + 1
    gaps = (ts.diff() / step - 1).fillna(0)
    longest_gap = int(gaps.max())

    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    ohlc_bad = int(((h < np.maximum(o, c)) | (l > np.minimum(o, c)) | (h < l)).sum())
    prices = df[["open", "high", "low", "close"]]
    nonpositive = int((prices <= 0).any(axis=1).sum())
    blank = int(prices.isna().any(axis=1).sum())

    _, cleaning = clean_bars(df) if not (ohlc_bad or nonpositive or blank) else (None, {})
    bad_prints = cleaning.get("bad_closes", 0) + cleaning.get("bad_opens", 0)
    bad_wicks = cleaning.get("bad_wicks", 0)

    runs = (c != c.shift()).cumsum()
    longest_flat = int(c.groupby(runs).size().max())

    last_end = (ts.iloc[-1] - pd.Timestamp(0)).total_seconds() + timeframe_seconds
    age_hours = (now - last_end) / 3600

    out = {
        "rows": len(df),
        "first": ts.iloc[0],
        "last": ts.iloc[-1],
        "coverage_pct": 100 * ts.nunique() / expected,
        "longest_gap_bars": longest_gap,
        "longest_gap_start": ts.iloc[int(gaps.values.argmax()) - 1] if longest_gap else None,
        "duplicates": int(ts.duplicated().sum()),
        "ohlc_violations": ohlc_bad,
        "nonpositive_prices": nonpositive,
        "blank_prices": blank,
        "bad_prints": bad_prints,
        "bad_wicks": bad_wicks,
        "longest_flat_bars": longest_flat,
        "zero_volume_pct": 100 * (df["volume"] <= 0).mean(),
        "hours_since_last_bar": age_hours,
    }

    issues, status = [], "ok"
    if out["duplicates"] or ohlc_bad or nonpositive or blank:
        status = "bad"
        issues.append("corrupt bars")
    if longest_gap > LONG_GAP_BARS:
        issues.append(f"data hole of {longest_gap} bars from {out['longest_gap_start']:%Y-%m-%d}")
    elif out["coverage_pct"] < 95:
        issues.append(f"{100 - out['coverage_pct']:.0f}% of bars missing (thinly traded)")
    if bad_prints or bad_wicks:
        issues.append(f"{bad_prints} bad price(s), {bad_wicks} bad wick(s): fixed automatically on load")
    if age_hours > DELISTED_AFTER_DAYS * 24:
        issues.append(f"no new data since {ts.iloc[-1]:%Y-%m-%d} (delisted?)")
    elif age_hours > 2 * timeframe_seconds / 3600 + 1:
        issues.append(f"last bar {age_hours:.0f}h old (run data update)")
    if issues and status == "ok":
        status = "warn"
    out["status"] = status
    out["issues"] = "; ".join(issues)
    return out


def quality_report(store, timeframe: str, timeframe_seconds: int, symbols=None, now: float = None) -> pd.DataFrame:
    rows = []
    for sym in symbols or store.symbols(timeframe):
        rows.append({"symbol": sym, **check_bars(store.load(sym, timeframe, clean=False), timeframe_seconds, now)})
    return pd.DataFrame(rows)
