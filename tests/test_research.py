import types

import numpy as np
import pandas as pd
import pytest

import my_signals
from tradingbot.research import (cross_sectional_ic, forward_returns, group_backtest, ic_stats,
                                 load_research_data, lookahead_check, run_research)

FIELDS = ("open", "high", "low", "close", "volume")


def fake_market(days=1500, coins=20, reversal=0.0, seed=0):
    """Daily prices for `coins` coins. With reversal > 0, each coin's return
    tends to undo part of its previous day's move (a pattern a reversal
    signal should find). Includes a shared market move, like real crypto."""
    rng = np.random.default_rng(seed)
    market = rng.normal(0, 0.03, days)
    own = np.zeros((days, coins))
    noise = rng.normal(0, 0.04, (days, coins))
    for t in range(1, days):
        own[t] = -reversal * own[t - 1] + noise[t]
    r = market[:, None] + own
    close = pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=pd.date_range("2021-01-01", periods=days, freq="D"),
                         columns=[f"C{i}/USD" for i in range(coins)])
    return types.SimpleNamespace(open=close.shift(1), high=close * 1.01, low=close * 0.99, close=close,
                                 volume=close * 0 + 1000)


def test_finds_a_real_reversal_effect():
    d = fake_market(reversal=0.3)
    r = run_research(my_signals.reversal_1d, d, rebalance=1, placebo_runs=0)
    assert r.ic_by_horizon.loc[1, "mean_ic"] > 0.1
    assert r.ic_by_horizon.loc[1, "t_stat"] > 5
    stats = r.groups.stats
    assert stats.loc["group_3", "cagr_pct"] > stats.loc["group_2", "cagr_pct"] > stats.loc["group_1", "cagr_pct"]
    assert not r.uses_future


def test_finds_nothing_in_pure_noise():
    d = fake_market(reversal=0.0, seed=3)
    for fn in (my_signals.reversal_1d, my_signals.momentum_7d):
        r = run_research(fn, d, placebo_runs=0)
        assert abs(r.ic_by_horizon.loc[1, "t_stat"]) < 3
        assert abs(r.ic_by_horizon.loc[1, "mean_ic"]) < 0.03


def test_flags_signals_that_peek_at_the_future():
    d = fake_market()
    cheat = lambda d: d.close.shift(-1) / d.close - 1        # tomorrow's return: impossible to know
    cheat.__name__ = "cheat"
    assert lookahead_check(cheat, d)
    assert not lookahead_check(my_signals.momentum_7d, d)


def test_groups_trade_on_the_next_day_only():
    """If a group could use the same day's price it was sorted on, a signal equal to
    TODAY's return would 'earn' it. It must earn nothing special; only knowing
    tomorrow's return (cheating) would."""
    d = fake_market(seed=5)
    today = d.close.pct_change(fill_method=None)
    honest = group_backtest(today, d.close, rebalance=1, fee=0)
    cheat = group_backtest(today.shift(-1), d.close, rebalance=1, fee=0)
    assert abs(honest.daily["top_minus_bottom"].mean()) < 0.003
    assert cheat.daily["top_minus_bottom"].mean() > 0.03


def test_overlapping_horizons_do_not_inflate_confidence():
    ic = pd.Series(np.random.default_rng(1).normal(0.02, 0.2, 1000))
    assert ic_stats(ic, 7)["t_stat"] == pytest.approx(ic_stats(ic, 1)["t_stat"] / np.sqrt(7))


def test_fees_and_turnover():
    d = fake_market(seed=6)
    constant = d.close * 0 + np.arange(d.close.shape[1])          # same ranking every day
    gb = group_backtest(constant, d.close, rebalance=7, fee=0.01)
    assert gb.turnover == 0 and gb.costs_per_year_pct == 0
    noisy = pd.DataFrame(np.random.default_rng(2).normal(size=d.close.shape), index=d.close.index, columns=d.close.columns)
    gb2 = group_backtest(noisy, d.close, rebalance=7, fee=0.01)
    assert gb2.turnover > 0.4 and gb2.costs_per_year_pct > 10


def test_holdout_is_never_loaded(tmp_path):
    from tests.test_cleaning import realistic_bars
    from tradingbot.store import BarStore
    store = BarStore(root=str(tmp_path), use_parquet=False)
    for i, sym in enumerate(["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "DOGE/USD", "LTC/USD"]):
        store.save(sym, "1h", realistic_bars(n=24 * 200, seed=i, start=100.0))
    d = load_research_data(store, store.symbols("1h"), "1h", end_before="2024-05-01")
    assert d.close.index.max() < pd.Timestamp("2024-05-01")
    assert list(d.close.columns) == store.symbols("1h")
    fwd = forward_returns(d.close, 7)
    assert fwd.iloc[-7:].isna().all().all()          # no forward return reaches into the holdout


def test_placebo_separates_real_effects_from_luck():
    from tradingbot.research import placebo_test
    real = fake_market(reversal=0.3, seed=8)
    p = placebo_test(my_signals.reversal_1d(real), real.close, rebalance=1, runs=50)
    assert p["share_as_good"] < 0.05 and p["actual_pct"] > p["placebo_p95_pct"]
    noise = fake_market(seed=9)
    p2 = placebo_test(my_signals.momentum_7d(noise), noise.close, runs=50)
    assert p2["share_as_good"] > 0.05
