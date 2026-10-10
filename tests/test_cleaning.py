import numpy as np
import pandas as pd

from tradingbot.cleaning import clean_bars, gaps


def realistic_bars(n=20000, seed=0, start=0.00001):
    """Fat-tailed hourly returns (Student-t, 3 dof) with calm and wild
    regimes, like real crypto. Starts at a PEPE-like tiny price."""
    rng = np.random.default_rng(seed)
    vol = np.where((np.arange(n) // 2000) % 2 == 0, 0.006, 0.02)   # alternating calm / wild months
    r = rng.standard_t(3, n) * vol / np.sqrt(3)
    close = start * np.exp(np.cumsum(r))
    open_ = np.r_[start, close[:-1]]
    wick = np.abs(rng.normal(0, 0.3, n)) * vol
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
        "open": open_, "high": np.maximum(open_, close) * (1 + wick),
        "low": np.minimum(open_, close) * (1 - wick), "close": close,
        "volume": rng.uniform(1, 100, n),
    })


def test_clean_data_is_left_alone():
    df = realistic_bars()
    out, rep = clean_bars(df)
    assert rep["bad_closes"] + rep["bad_opens"] + rep["bad_wicks"] <= 2   # essentially no false alarms
    assert len(out) >= len(df) - 2


def test_real_jumps_survive():
    df = realistic_bars()
    # a real +40% jump that stays, and a pump that lasts 4 bars before fading
    df.loc[5000:, ["open", "high", "low", "close"]] *= 1.4
    df.loc[8000:8003, ["high", "close"]] *= 1.3
    df.loc[8001:8003, ["open", "low"]] *= 1.3
    out, rep = clean_bars(df)
    kept = set(out["timestamp"])
    assert df.at[5000, "timestamp"] in kept
    assert all(df.at[i, "timestamp"] in kept for i in range(8000, 8004))


def test_bad_prints_are_removed_or_fixed():
    df = realistic_bars()
    df.loc[3000, ["high", "close"]] = df.loc[3000, "close"] * 100      # 100x price for one bar
    df.loc[6000, ["low", "close"]] = df.loc[6000, "close"] / 1000     # near-zero price
    df.loc[9000:9001, ["high", "close"]] *= 20                          # two bad bars in a row
    df.loc[12000, "high"] = df.loc[12000, "close"] * 5                 # stray trade: silly high only
    df.loc[15000, "low"] = df.loc[15000, "close"] / 5                  # flash-crash wick
    df.loc[17000, "open"] = df.loc[17000, "open"] * 30                 # bad opening price
    df.loc[17000, "high"] = df.loc[17000, "open"]
    out, rep = clean_bars(df)

    kept = set(out["timestamp"])
    for i in (3000, 6000, 9000, 9001):
        assert df.at[i, "timestamp"] not in kept
    assert rep["bad_closes"] == 4
    assert rep["bad_wicks"] >= 2 and rep["bad_opens"] == 1

    # nothing silly survives: no one-bar move or wick over 10x
    r = np.log(out["close"]).diff().abs()
    assert r.max() < np.log(3)
    assert (out["high"] / out["low"]).max() < 3
    assert (out["high"] >= out[["open", "close"]].max(axis=1)).all()
    assert (out["low"] <= out[["open", "close"]].min(axis=1)).all()
    assert set(rep["changes"]["kind"]) >= {"bad close (bar dropped)", "bad high", "bad low", "bad open"}


def test_gaps_are_listed():
    df = realistic_bars(n=1000).drop(index=range(300, 700)).reset_index(drop=True)
    g = gaps(df, 3600)
    assert len(g) == 1 and g["missing_bars"].iloc[0] == 400
