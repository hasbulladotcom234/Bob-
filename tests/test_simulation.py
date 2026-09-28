from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from tradingbot.config import Config
from tradingbot.montecarlo import run_monte_carlo
from tradingbot.simulation import MODELS, ModelParams, calibrate, paths_to_frames, simulate_paths

TRUE = ModelParams(model="bates", drift=5e-5, theta=4e-5, kappa=0.02, xi=0.001, rho=-0.4,
                   v0=4e-5, jump_intensity=0.01, jump_mean=-0.005, jump_std=0.03)


def _frame(params, n_bars, seed):
    rng = np.random.default_rng(seed)
    paths = simulate_paths(params, n_bars, 1, 30_000.0, rng)
    return paths_to_frames(paths, pd.Timestamp("2024-01-01"), pd.Timedelta("1h"), rng)[0]


@pytest.fixture(scope="module")
def long_bates_history():
    return _frame(TRUE, 20_000, seed=1)


def test_simulated_bars_are_valid_ohlc():
    df = _frame(TRUE, 500, seed=0)
    assert (df[["open", "high", "low", "close"]] > 0).all().all()
    assert (df["high"] >= df[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (df["open"].iloc[1:].to_numpy() == pytest.approx(df["close"].iloc[:-1].to_numpy()))


def test_simulation_is_reproducible_with_seed():
    a = _frame(TRUE, 200, seed=7)
    b = _frame(TRUE, 200, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_gbm_simulation_matches_parameters():
    gbm = ModelParams(model="gbm", drift=1e-4, theta=1e-4, v0=1e-4)
    rng = np.random.default_rng(2)
    paths = simulate_paths(gbm, 50, 4000, 100.0, rng, substeps=4)
    r = np.diff(np.log(paths["close"]), axis=1).ravel()
    assert r.mean() == pytest.approx(1e-4, abs=2e-5)
    assert r.var() == pytest.approx(1e-4, rel=0.05)


def test_calibration_recovers_bates_parameters(long_bates_history):
    fit = calibrate(long_bates_history, "bates")
    assert fit.theta == pytest.approx(TRUE.theta, rel=0.2)
    assert fit.kappa == pytest.approx(TRUE.kappa, rel=0.6)
    assert fit.xi == pytest.approx(TRUE.xi, rel=0.6)
    assert 0 < fit.jump_intensity < 2 * TRUE.jump_intensity
    assert fit.jump_mean < 0
    assert fit.rho < 0


@pytest.mark.parametrize("model", MODELS)
def test_calibrate_every_model(model, trending_ohlcv):
    fit = calibrate(trending_ohlcv, model)
    assert fit.model == model
    assert fit.theta > 0
    if model in ("gbm", "merton"):
        assert not fit.stochastic_vol
    if model in ("gbm", "heston"):
        assert fit.jump_intensity == 0


def test_calibrate_rejects_short_history(trending_ohlcv):
    with pytest.raises(ValueError):
        calibrate(trending_ohlcv.head(20))


def test_monte_carlo_summary(trending_ohlcv):
    mc = run_monte_carlo(trending_ohlcv, Config(), model="bates", n_paths=20, seed=0)
    assert len(mc.paths) == 20
    s = mc.summary
    assert 0 <= s["prob_loss_pct"] <= 100
    assert s["cvar_5_pct"] <= s["var_5_pct"] <= s["median_return_pct"]
    assert s["worst_max_drawdown_pct"] <= s["median_max_drawdown_pct"] <= 0


def test_monte_carlo_zero_drift_strips_trend(trending_ohlcv):
    mc = run_monte_carlo(trending_ohlcv, Config(), model="gbm", n_paths=5, drift="zero", seed=0)
    assert mc.model_params.drift == 0.0
    with pytest.raises(ValueError):
        run_monte_carlo(trending_ohlcv, Config(), n_paths=1, drift="up")


def test_monte_carlo_accepts_given_params(trending_ohlcv):
    params = replace(TRUE, drift=0.0)
    mc = run_monte_carlo(trending_ohlcv, Config(), n_paths=3, n_bars=300, seed=0, model_params=params)
    assert mc.model_params == params
