"""Removing bad prices ("bad prints") from candle data.

The raw data in the store is never modified. Cleaning happens when data is
loaded, so the rules can be improved later and re-applied to everything.

What counts as a bad print: a price far from BOTH the bars just before and
just after it. A real move stays (the following bars sit near the new
level); a bad print snaps back. "Far" adapts to how volatile the coin is at
that time, so a normal 10% hour for a meme coin in a frenzy isn't flagged,
but the same move on a quiet day is.

A real crash can also snap back (19 May 2021: crypto fell ~30% within hours
and partly recovered). Deleting those would hide real losses and make every
strategy look safer than it was. So a suspect is checked against the market
(Bitcoin; Ether when cleaning Bitcoin): if the market made a similar move
in the same hour, it's real and is kept. A bad print is specific to one coin.

Three kinds are handled:
- bad close: the whole bar is wrong -> the bar is dropped (treated as missing)
- bad open:  only the opening price is off -> replaced by the previous close
- bad wick:  only the high or low is off (e.g. a stray trade at a silly
             price) -> cut back to the bar's own open/close range, so
             backtests don't trigger stop-losses on prices nobody could trade at

Data holes (a day or more with no bars) split the data into separate
stretches that are cleaned independently: prices from before a hole are
never used to judge prices after it.

This uses bars on both sides of each bar, i.e. it looks at the "future".
That's standard for cleaning history (a live trader would also have seen
the price snap back within an hour), but it means a strategy that would
have reacted to a bad print in real time looks slightly better in a
backtest than it would have live.
"""
import numpy as np
import pandas as pd

Z = 10.0              # how many "typical moves" away a price must be...
MIN_MOVE = 0.10       # ...and never less than 10%, to count as a bad print
NEIGHBOURS = 3        # bars on each side used as the reference level
VOL_WINDOW = 720      # bars used to measure how volatile the coin is right now (30 days of 1h bars)
LONG_GAP_BARS = 24    # a hole this long splits the data into separately cleaned stretches
CONFIRM_SHARE = 0.25  # the market moving at least this share of the coin's move...
CONFIRM_MIN = 0.03    # ...or at least 3%, in the same direction and hour, means the move was real


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


def _spike_sizes(log_close: pd.Series):
    """How far each close sticks out above (up) or below (down) BOTH the level
    before and the level after it. Positive = sticks out; negative = doesn't."""
    before, after = _side_medians(log_close)
    up = np.minimum(log_close - before, log_close - after)
    down = np.minimum(before - log_close, after - log_close)
    return up, down, before, after


def segment_ids(timestamps: pd.Series) -> pd.Series:
    """Number the stretches of data between holes of LONG_GAP_BARS or more."""
    step = timestamps.diff().median()
    if pd.isna(step):
        return pd.Series(0, index=timestamps.index)
    return (timestamps.diff() > LONG_GAP_BARS * step).cumsum()


def market_moves(reference: pd.DataFrame) -> pd.DataFrame:
    """For the market proxy: how much each hour's close sticks out (up/down)
    and how long its wicks are, indexed by timestamp. Used to confirm that a
    coin's sudden move was a real market event."""
    ref = reference.sort_values("timestamp").reset_index(drop=True)
    parts = []
    for _, seg in ref.groupby(segment_ids(ref["timestamp"])):
        c = np.log(seg["close"])
        up, down, _, _ = _spike_sizes(c)
        body_hi = np.log(np.maximum(seg["open"], seg["close"]))
        body_lo = np.log(np.minimum(seg["open"], seg["close"]))
        parts.append(pd.DataFrame({"up": up, "down": down,
                                   "wick_up": np.log(seg["high"]) - body_hi,
                                   "wick_down": body_lo - np.log(seg["low"])}).set_index(seg["timestamp"]))
    return pd.concat(parts) if parts else pd.DataFrame(columns=["up", "down", "wick_up", "wick_down"])


def _confirmed(coin_size: pd.Series, market_size: pd.Series) -> pd.Series:
    return (market_size >= np.minimum(CONFIRM_SHARE * coin_size, CONFIRM_MIN)).fillna(False)


def _clean_segment(df: pd.DataFrame, market: pd.DataFrame, changes: list, kept: list) -> pd.DataFrame:
    if len(df) < 2 * NEIGHBOURS + 1:
        return df
    df = df.reset_index(drop=True)
    mkt = market.reindex(df["timestamp"]).reset_index(drop=True) if market is not None else None

    # 1. bad closes: the bar sticks out from the level both before and after it
    c = np.log(df["close"])
    up, down, before, after = _spike_sizes(c)
    threshold = np.maximum(Z * local_volatility(c), MIN_MOVE)
    suspect_up, suspect_down = (up > threshold).fillna(False), (down > threshold).fillna(False)
    real_up = suspect_up & _confirmed(up, mkt["up"]) if mkt is not None else suspect_up & False
    real_down = suspect_down & _confirmed(down, mkt["down"]) if mkt is not None else suspect_down & False
    bad_close = (suspect_up & ~real_up) | (suspect_down & ~real_down)
    for i in np.flatnonzero(real_up | real_down):
        kept.append({"timestamp": df.at[i, "timestamp"], "kind": "close", "price": df.at[i, "close"],
                     "level_before": np.exp(before[i]), "level_after": np.exp(after[i])})
    for i in np.flatnonzero(bad_close):
        changes.append({"timestamp": df.at[i, "timestamp"], "kind": "bad close (bar dropped)",
                        "raw": df.at[i, "close"], "fixed_to": np.nan, "level_before": np.exp(before[i]),
                        "level_after": np.exp(after[i]), "size_pct": 100 * max(up[i], down[i])})
    out = df[~bad_close].reset_index(drop=True)
    mkt = mkt[~bad_close].reset_index(drop=True) if mkt is not None else None

    # 2. bad opens: open far from both the previous close and this bar's close.
    #    The first bar of a stretch has no previous close, so use the next close instead.
    c = np.log(out["close"])
    o = np.log(out["open"])
    other = c.shift(1).fillna(c.shift(-1))
    threshold = np.maximum(Z * local_volatility(c), MIN_MOVE)
    bad_open = (((o - other > threshold) & (o - c > threshold)) |
                ((other - o > threshold) & (c - o > threshold))).fillna(False)
    replacement = np.exp(c.shift(1)).fillna(out["close"])
    for i in np.flatnonzero(bad_open):
        changes.append({"timestamp": out.at[i, "timestamp"], "kind": "bad open", "raw": out.at[i, "open"],
                        "fixed_to": replacement[i], "level_before": np.exp(other[i]),
                        "level_after": out.at[i, "close"], "size_pct": 100 * abs(o[i] - c[i])})
    out.loc[bad_open, "open"] = replacement[bad_open]

    # 3. bad wicks: high/low far beyond this bar's AND nearby bars' open/close
    #    range. Open/close are used as the yardstick (not the neighbours' highs
    #    and lows) because stray trades often repeat, and would hide each other.
    o, c = out["open"], out["close"]
    body_hi, body_lo = np.maximum(o, c), np.minimum(o, c)
    shifts = [k for k in range(-NEIGHBOURS, NEIGHBOURS + 1) if k]
    near_hi = np.maximum(body_hi, pd.concat([body_hi.shift(k) for k in shifts], axis=1).max(axis=1))
    near_lo = np.minimum(body_lo, pd.concat([body_lo.shift(k) for k in shifts], axis=1).min(axis=1))
    h, l = out["high"], out["low"]
    wick_up = np.log(h / near_hi)
    wick_down = np.log(near_lo / l)
    threshold = np.maximum(Z * local_volatility(np.log(c)), MIN_MOVE)
    suspect_hi, suspect_lo = (wick_up > threshold).fillna(False), (wick_down > threshold).fillna(False)
    real_hi = suspect_hi & _confirmed(wick_up, mkt["wick_up"]) if mkt is not None else suspect_hi & False
    real_lo = suspect_lo & _confirmed(wick_down, mkt["wick_down"]) if mkt is not None else suspect_lo & False
    bad_hi, bad_lo = suspect_hi & ~real_hi, suspect_lo & ~real_lo
    for i in np.flatnonzero(real_hi):
        kept.append({"timestamp": out.at[i, "timestamp"], "kind": "high", "price": h[i],
                     "level_before": body_hi[i], "level_after": body_hi[i]})
    for i in np.flatnonzero(real_lo):
        kept.append({"timestamp": out.at[i, "timestamp"], "kind": "low", "price": l[i],
                     "level_before": body_lo[i], "level_after": body_lo[i]})
    for i in np.flatnonzero(bad_hi):
        changes.append({"timestamp": out.at[i, "timestamp"], "kind": "bad high", "raw": h[i],
                        "fixed_to": body_hi[i], "level_before": body_hi[i], "level_after": c[i],
                        "size_pct": 100 * wick_up[i]})
    for i in np.flatnonzero(bad_lo):
        changes.append({"timestamp": out.at[i, "timestamp"], "kind": "bad low", "raw": l[i],
                        "fixed_to": body_lo[i], "level_before": body_lo[i], "level_after": c[i],
                        "size_pct": 100 * wick_down[i]})
    out.loc[bad_hi, "high"] = body_hi[bad_hi]
    out.loc[bad_lo, "low"] = body_lo[bad_lo]
    return out


def clean_bars(df: pd.DataFrame, reference: pd.DataFrame = None):
    """Return (cleaned candles, report).

    `reference` is the market proxy's raw candles (e.g. Bitcoin), used to
    keep real market-wide moves. Without it, nothing can be confirmed and
    every suspect is treated as a bad print.

    The report has counts, a table of every change ("changes", largest
    first) and a table of suspects kept because the market moved too ("kept").
    """
    report = {"bad_closes": 0, "bad_opens": 0, "bad_wicks": 0, "kept_by_market": 0,
              "changes": pd.DataFrame(), "kept": pd.DataFrame()}
    if df.empty:
        return df.copy(), report
    df = df.sort_values("timestamp").reset_index(drop=True)
    market = market_moves(reference) if reference is not None and not reference.empty else None

    changes, kept = [], []
    parts = [_clean_segment(seg, market, changes, kept) for _, seg in df.groupby(segment_ids(df["timestamp"]))]
    out = pd.concat(parts, ignore_index=True) if parts else df.copy()

    if changes:
        ch = pd.DataFrame(changes)
        report["bad_closes"] = int((ch["kind"] == "bad close (bar dropped)").sum())
        report["bad_opens"] = int((ch["kind"] == "bad open").sum())
        report["bad_wicks"] = int(ch["kind"].isin(["bad high", "bad low"]).sum())
        report["changes"] = ch.sort_values("size_pct", ascending=False).reset_index(drop=True)
    if kept:
        report["kept"] = pd.DataFrame(kept).sort_values("timestamp").reset_index(drop=True)
        report["kept_by_market"] = len(kept)
    return out, report


def gaps(df: pd.DataFrame, timeframe_seconds: int, min_bars: int = LONG_GAP_BARS) -> pd.DataFrame:
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
