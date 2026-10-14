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


def _crash(df, i, depth, recover):
    """Price drops by `depth` at bar i and recovers to `recover` of the way back, like 19 May 2021."""
    level = df.loc[i - 1, "close"]
    low = level * (1 - depth)
    df.loc[i, ["close"]] = low
    df.loc[i, "low"] = low * 0.99
    df.loc[i, "high"] = max(df.loc[i, "open"], low)
    df.loc[i + 1, "open"] = low
    df.loc[i + 1, "low"] = min(df.loc[i + 1, "low"], low)
    return df


def test_market_wide_crash_is_kept_but_lone_spike_is_removed():
    coin = realistic_bars(n=5000, seed=1, start=40.0)
    btc = realistic_bars(n=5000, seed=2, start=40000.0)
    _crash(coin, 2000, 0.25, 1)      # coin -25% for one hour, back next hour...
    _crash(btc, 2000, 0.12, 1)       # ...and Bitcoin -12% in the same hour: a real crash
    _crash(coin, 3000, 0.25, 1)      # coin alone -25% and back: a bad print
    out, rep = clean_bars(coin, reference=btc)
    kept = set(out["timestamp"])
    assert coin.at[2000, "timestamp"] in kept
    assert coin.at[3000, "timestamp"] not in kept
    assert rep["kept_by_market"] >= 1 and coin.at[2000, "timestamp"] in set(rep["kept"]["timestamp"])
    # without a market reference the real crash can't be confirmed, so it's (wrongly) removed
    out_no_ref, _ = clean_bars(coin)
    assert coin.at[2000, "timestamp"] not in set(out_no_ref["timestamp"])


def test_repeated_stray_highs_do_not_hide_each_other():
    """PEPE pattern: the same silly high (about 6x the price) on many bars, often back to back."""
    df = realistic_bars(n=3000, seed=4, start=0.000009)
    stray = 0.000054
    hit = [500, 501, 502, 900, 1400, 1401, 2200]
    df.loc[hit, "high"] = stray
    out, rep = clean_bars(df)
    assert (out["high"] < 0.00003).all()
    assert rep["bad_wicks"] == len(hit)


def test_stretches_after_a_hole_are_cleaned_on_their_own():
    """SOL pattern: a 400-day hole, price 8x higher after it, and the first bar after
    the hole carries a stale open/low from before the hole."""
    before = realistic_bars(n=2000, seed=5, start=20.0)
    after = realistic_bars(n=2000, seed=6, start=158.0)
    after["timestamp"] = after["timestamp"] + pd.Timedelta(days=500)
    after.loc[0, ["open", "low"]] = 18.1      # stale prices from before the hole
    df = pd.concat([before, after], ignore_index=True)
    out, rep = clean_bars(df)
    first_after = out[out["timestamp"] == after.at[0, "timestamp"]].iloc[0]
    assert first_after["open"] > 100 and first_after["low"] > 100   # fixed using its own stretch
    assert len(out) >= len(df) - 3                                   # the hole itself isn't "a bad print"


def test_backtest_closes_position_before_a_data_hole():
    import types
    from tradingbot.backtester import run_backtest
    from tradingbot.config import Config
    before = realistic_bars(n=300, seed=7, start=20.0)
    after = realistic_bars(n=300, seed=8, start=160.0)
    after["timestamp"] = after["timestamp"] + pd.Timedelta(days=400)
    df = pd.concat([before, after], ignore_index=True)
    strat = types.SimpleNamespace(PARAMS={"stop_loss_pct": 0.9, "take_profit_pct": 100.0},
                                  indicators=lambda c, p: c, decide=lambda bar, prev, pos, p: "buy")
    result = run_backtest(df, Config(), strat)
    first = result.trades.iloc[0]
    assert first["reason"] == "data_hole"
    assert first["exit_time"] == before["timestamp"].iloc[-1]
    assert first["exit_price"] < 40          # no fake 8x gain from jumping across the hole
