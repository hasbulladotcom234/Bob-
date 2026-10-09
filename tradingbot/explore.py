"""Statistical facts about a market, to ground strategy ideas in data.

Every number comes with the question it answers. None of this is a
strategy; it's where ideas for one come from.
"""
import numpy as np
import pandas as pd

from .metrics import periods_per_year


def autocorr_table(returns: pd.Series, lags=(1, 2, 3, 6, 12, 24)) -> pd.DataFrame:
    """Correlation of each bar's return with the return `lag` bars earlier.
    Positive = moves tend to continue (momentum). Negative = moves tend to
    reverse (mean reversion). Inside +/- noise_band = indistinguishable from 0."""
    band = 1.96 / np.sqrt(len(returns))
    rows = []
    for lag in lags:
        r = returns.autocorr(lag)
        rows.append({"lag_bars": lag, "return_autocorr": r, "abs_return_autocorr": returns.abs().autocorr(lag),
                     "significant": abs(r) > band})
    return pd.DataFrame(rows), band


def seasonality(returns: pd.Series, timestamps: pd.Series, key: str) -> pd.DataFrame:
    """Average return by hour of day or day of week, with a t-stat.
    |t| > 2 is roughly 'unlikely to be pure luck', but with 24 hours to pick
    from, one or two will clear that bar by chance alone."""
    groups = timestamps.dt.hour if key == "hour" else timestamps.dt.day_name().str[:3]
    g = returns.groupby(groups.values)
    out = pd.DataFrame({"mean_return_pct": g.mean() * 100, "bars": g.size()})
    out["t_stat"] = g.mean() / (g.std() / np.sqrt(g.size()))
    if key == "day":
        out = out.reindex(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]).dropna()
    return out


def forward_returns_by_condition(df: pd.DataFrame, condition: pd.Series, horizon: int) -> pd.DataFrame:
    """Average return over the next `horizon` bars when `condition` is True vs False.
    This is the simplest test of any idea: does the signal predict anything?"""
    fwd = df["close"].shift(-horizon) / df["close"] - 1
    valid = fwd.notna() & condition.notna()
    rows = []
    for state in (True, False):
        f = fwd[valid & (condition == state)]
        rows.append({"condition": state, "bars": len(f), "avg_fwd_return_pct": f.mean() * 100,
                     "pct_positive": (f > 0).mean() * 100})
    return pd.DataFrame(rows)


def report(df: pd.DataFrame) -> str:
    returns = df["close"].pct_change().dropna()
    ppy = periods_per_year(df["timestamp"])
    lines = []
    add = lines.append

    total = df["close"].iloc[-1] / df["close"].iloc[0] - 1
    dd = (df["close"] / df["close"].cummax() - 1).min()
    add("== The basics: what does holding this coin feel like? ==")
    add(f"Period:              {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]} ({len(df)} bars)")
    add(f"Buy & hold return:   {total * 100:.2f}%")
    add(f"Worst drop from peak:{dd * 100:8.2f}%")
    add(f"Annualized vol:      {returns.std() * np.sqrt(ppy) * 100:.1f}%   (S&P 500 is typically ~15-20%)")

    add("\n== Return distribution: are big moves rarer than a bell curve says? ==")
    add(f"Mean bar return:     {returns.mean() * 100:.4f}%   std: {returns.std() * 100:.3f}%")
    add(f"Up bars:             {(returns > 0).mean() * 100:.1f}%")
    add(f"Skew:                {returns.skew():.2f}   (negative = crashes bigger than rallies)")
    add(f"Excess kurtosis:     {returns.kurt():.2f}   (0 = bell curve; higher = fatter tails)")
    big = (returns.abs() > 4 * returns.std()).mean() * 100
    add(f"Bars moving > 4 std: {big:.3f}%   (a bell curve predicts 0.006%)")
    add(f"Biggest bar up/down: {returns.max() * 100:.2f}% / {returns.min() * 100:.2f}%")

    ac, band = autocorr_table(returns)
    add("\n== Memory: do moves continue (momentum) or reverse (mean reversion)? ==")
    add(f"Noise band: +/-{band:.4f}. Values inside it are indistinguishable from zero.")
    add("abs_return_autocorr > 0 means calm follows calm and wild follows wild (volatility clustering).")
    add(ac.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if ppy > 365:  # intraday bars
        add("\n== Hour of day (UTC): is there a time when it tends to move? ==")
        add(seasonality(returns, df["timestamp"].iloc[1:], "hour").to_string(float_format=lambda x: f"{x:.4f}"))
    add("\n== Day of week ==")
    add(seasonality(returns, df["timestamp"].iloc[1:], "day").to_string(float_format=lambda x: f"{x:.4f}"))

    add("\n== Example signal test: after price is above its 50-bar average, what happens next? ==")
    above = df["close"] > df["close"].ewm(span=50, adjust=False).mean()
    for h in (1, 6, 24):
        t = forward_returns_by_condition(df, above, h)
        add(f"next {h} bars:")
        add(t.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    add("\nCopy forward_returns_by_condition() into your own tests: swap in any condition you believe in.")
    return "\n".join(lines)
