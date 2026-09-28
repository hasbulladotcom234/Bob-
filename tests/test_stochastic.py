import numpy as np
import pytest

from tests.conftest import make_synthetic_ohlcv
from tradingbot.stochastic import (
    bracket_edge, ewma_drift_vol, expected_exit_bars, take_profit_probability,
)


def _barriers(sl, tp):
    return np.log1p(tp), -np.log1p(-sl)


def test_driftless_probability_is_barrier_ratio():
    a, b = _barriers(0.03, 0.06)
    assert take_profit_probability(0.0, 0.01, 0.03, 0.06) == pytest.approx(b / (a + b))


def test_probability_is_continuous_through_zero_drift():
    p0 = take_profit_probability(0.0, 0.01, 0.03, 0.06)
    assert take_profit_probability(1e-10, 0.01, 0.03, 0.06) == pytest.approx(p0, abs=1e-6)
    assert take_profit_probability(-1e-10, 0.01, 0.03, 0.06) == pytest.approx(p0, abs=1e-6)


def test_probability_increases_with_drift_and_saturates():
    drifts = np.linspace(-0.01, 0.01, 41)
    p = take_profit_probability(drifts, 0.005, 0.03, 0.06)
    assert np.all(np.diff(p) >= 0)
    assert p[0] == pytest.approx(0.0, abs=1e-9)
    assert p[-1] == pytest.approx(1.0, abs=1e-9)
    assert np.all(np.isfinite(p))


def test_probability_matches_monte_carlo():
    nu, sigma, sl, tp = 2e-4, 0.006, 0.03, 0.06
    a, b = _barriers(sl, tp)
    rng = np.random.default_rng(0)
    n, dt = 4000, 0.05
    x = np.zeros(n)
    done = np.zeros(n, bool)
    up = np.zeros(n, bool)
    while not done.all():
        live = ~done
        x[live] += nu * dt + sigma * np.sqrt(dt) * rng.standard_normal(live.sum())
        hit_up = live & (x >= a)
        up |= hit_up
        done |= hit_up | (live & (x <= -b))
    assert up.mean() == pytest.approx(take_profit_probability(nu, sigma, sl, tp), abs=0.03)


def test_missing_volatility_gives_nan():
    p = take_profit_probability([0.0, 0.0], [np.nan, 0.0], 0.03, 0.06)
    assert np.isnan(p).all()


def test_expected_exit_bars_driftless_formula():
    a, b = _barriers(0.03, 0.06)
    assert expected_exit_bars(0.0, 0.01, 0.03, 0.06) == pytest.approx(a * b / 0.01 ** 2)
    assert expected_exit_bars(1e-6, 0.01, 0.03, 0.06) == pytest.approx(a * b / 0.01 ** 2, rel=1e-3)


def test_bracket_edge_accounts_for_costs():
    assert bracket_edge(0.5, 0.03, 0.06) == pytest.approx(0.015)
    assert bracket_edge(0.5, 0.03, 0.06, round_trip_cost=0.01) == pytest.approx(0.005)


def test_ewma_volatility_recovers_true_volatility():
    df = make_synthetic_ohlcv(n=5000, trend=0.0, vol=0.01)
    _, sigma = ewma_drift_vol(df["close"], lookback=2000)
    assert sigma.iloc[-1] == pytest.approx(0.01, rel=0.15)


def test_full_shrinkage_zeroes_drift():
    df = make_synthetic_ohlcv(n=500, trend=0.001)
    nu, _ = ewma_drift_vol(df["close"], lookback=100, shrinkage=0.0)
    assert (nu.dropna() == 0).all()
