"""Stochastic market models, calibrated to real price history, used to
generate realistic synthetic OHLCV for Monte Carlo stress tests.

All four models are special cases of the Bates model (Heston stochastic
volatility plus Merton log-normal jumps). Time is measured in bars.

    d(log S) = (m - v/2) dt + sqrt(v) dW1 + J dN
    dv       = kappa (theta - v) dt + xi sqrt(v) dW2,    d<W1, W2> = rho dt

    N ~ Poisson(lambda), J ~ Normal(jump_mean, jump_std^2)

    gbm     constant variance, no jumps
    merton  constant variance, jumps
    heston  stochastic variance, no jumps
    bates   stochastic variance, jumps

`drift` is the expected log return per bar. m is chosen so the diffusion,
variance and jump terms together reproduce it.

Calibration is moment-based, not full maximum likelihood: jumps are
flagged as returns more than `jump_threshold` local robust standard
deviations from the median. The variance process is fitted as an AR(1) on
block realized variance, which is the exact discretization of the CIR mean
reversion. It is quick and gets the shape right (fat tails, volatility
clustering, leverage effect). It is not precise.
"""
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .stochastic import log_returns

MODELS = ("gbm", "merton", "heston", "bates")


@dataclass
class ModelParams:
    model: str = "bates"
    drift: float = 0.0          # expected log return per bar
    theta: float = 1e-4         # long-run variance per bar (sigma^2)
    kappa: float = 0.0          # variance mean-reversion speed per bar (0 = constant variance)
    xi: float = 0.0             # volatility of variance
    rho: float = 0.0            # correlation between price and variance shocks
    v0: float = 1e-4            # starting variance
    jump_intensity: float = 0.0  # expected jumps per bar
    jump_mean: float = 0.0      # mean log jump size
    jump_std: float = 0.0       # std of log jump size

    @property
    def stochastic_vol(self) -> bool:
        return self.kappa > 0 and self.xi > 0

    def describe(self, bars_per_year: float) -> dict:
        d = asdict(self)
        d["annual_drift_pct"] = self.drift * bars_per_year * 100
        d["annual_vol_pct"] = np.sqrt(self.theta * bars_per_year) * 100
        d["jumps_per_year"] = self.jump_intensity * bars_per_year
        d["vol_half_life_bars"] = np.log(2) / self.kappa if self.kappa > 0 else float("inf")
        return d


def _flag_jumps(r: pd.Series, threshold: float, window: int = 100) -> pd.Series:
    centered = r - r.median()
    local_mad = centered.abs().rolling(window, center=True, min_periods=20).median()
    local_mad = local_mad.fillna(centered.abs().median())
    return centered.abs() > threshold * 1.4826 * local_mad


def calibrate(df: pd.DataFrame, model: str = "bates", jump_threshold: float = 4.0,
              block: int = 24) -> ModelParams:
    """Fit ModelParams to the close prices in df."""
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}")
    r = log_returns(df["close"]).dropna()
    if len(r) < 3 * block:
        raise ValueError(f"need at least {3 * block + 1} bars to calibrate, got {len(r) + 1}")

    params = ModelParams(model=model, drift=float(r.mean()))

    diffusive = r
    if model in ("merton", "bates"):
        jumps = _flag_jumps(r, jump_threshold)
        n_jumps = int(jumps.sum())
        # Genuine jumps are rare; if many bars are flagged the data is just
        # heavy-tailed everywhere and the diffusion should absorb it.
        if 2 <= n_jumps <= 0.05 * len(r):
            params.jump_intensity = n_jumps / len(r)
            params.jump_mean = float(r[jumps].mean())
            params.jump_std = float(r[jumps].std())
            diffusive = r[~jumps]

    theta = float(diffusive.var())
    params.theta = params.v0 = theta

    if model in ("heston", "bates"):
        _fit_variance_process(params, diffusive, block)

    return params


def _fit_variance_process(params: ModelParams, r: pd.Series, block: int) -> None:
    n_blocks = len(r) // block
    blocks = r.iloc[: n_blocks * block].to_numpy().reshape(n_blocks, block)
    v = (blocks ** 2).mean(axis=1)            # realized variance per bar, per block
    block_ret = blocks.sum(axis=1)

    x, y = v[:-1], v[1:]
    beta, alpha = np.polyfit(x, y, 1)
    if not 0 < beta < 1:
        return  # no detectable mean reversion; keep constant variance

    theta = alpha / (1 - beta)
    kappa = -np.log(beta) / block
    resid = y - (alpha + beta * x)
    # Remove the sampling noise of realized variance (about 2 theta^2 / block for
    # Gaussian returns) so it isn't mistaken for volatility-of-volatility.
    true_resid_var = max(resid.var() - 2 * theta ** 2 / block, 0.0)
    # Exact CIR conditional variance over one block, at v = theta.
    denom = theta * (1 - beta ** 2) / (2 * kappa)
    xi = np.sqrt(true_resid_var / denom) if denom > 0 else 0.0

    dv = np.diff(v)
    rho = np.corrcoef(block_ret[1:], dv)[0, 1] if dv.std() > 0 else 0.0

    params.theta = float(theta)
    params.kappa = float(kappa)
    params.xi = float(min(xi, 5.0))
    params.rho = float(np.clip(np.nan_to_num(rho), -0.95, 0.95))
    params.v0 = float(v[-1])


def simulate_paths(params: ModelParams, n_bars: int, n_paths: int, start_price: float,
                   rng: np.random.Generator, substeps: int = 16):
    """Simulate OHLC for n_paths independent paths, n_bars each. Each bar is
    split into `substeps` Euler steps (full truncation for the variance) so
    highs and lows come from the simulated intrabar path.

    Returns a dict of (n_paths, n_bars) arrays: open, high, low, close.
    """
    dt = 1.0 / substeps
    jump_drift = params.jump_intensity * params.jump_mean
    # m - v/2 averages to `drift` when v sits at theta and jumps are included.
    m = params.drift - jump_drift + params.theta / 2

    log_s = np.full(n_paths, np.log(start_price))
    v = np.full(n_paths, params.v0)
    out = {k: np.empty((n_paths, n_bars)) for k in ("open", "high", "low", "close")}
    sqrt_one_minus_rho2 = np.sqrt(1 - params.rho ** 2)

    for t in range(n_bars):
        bar_open = log_s.copy()
        hi = log_s.copy()
        lo = log_s.copy()
        for _ in range(substeps):
            z1 = rng.standard_normal(n_paths)
            vp = np.maximum(v, 0.0)
            log_s = log_s + (m - vp / 2) * dt + np.sqrt(vp * dt) * z1
            if params.stochastic_vol:
                z2 = params.rho * z1 + sqrt_one_minus_rho2 * rng.standard_normal(n_paths)
                v = v + params.kappa * (params.theta - vp) * dt + params.xi * np.sqrt(vp * dt) * z2
            if params.jump_intensity > 0:
                n = rng.poisson(params.jump_intensity * dt, n_paths)
                hit = n > 0
                if hit.any():
                    log_s[hit] += params.jump_mean * n[hit] + params.jump_std * np.sqrt(n[hit]) * rng.standard_normal(hit.sum())
            np.maximum(hi, log_s, out=hi)
            np.minimum(lo, log_s, out=lo)
        out["open"][:, t] = bar_open
        out["high"][:, t] = hi
        out["low"][:, t] = lo
        out["close"][:, t] = log_s

    return {k: np.exp(a) for k, a in out.items()}


def paths_to_frames(paths: dict, start_time: pd.Timestamp, freq: pd.Timedelta,
                    rng: np.random.Generator) -> list:
    """Turn simulate_paths output into one OHLCV DataFrame per path.
    Volume is synthetic (scales with the bar's range) and unused by the
    strategy."""
    n_paths, n_bars = paths["close"].shape
    timestamps = pd.date_range(start_time, periods=n_bars, freq=freq)
    frames = []
    for i in range(n_paths):
        bar_range = np.log(paths["high"][i] / paths["low"][i])
        volume = 1000 * (1 + bar_range / (bar_range.mean() + 1e-12)) * rng.lognormal(0, 0.3, n_bars)
        frames.append(pd.DataFrame({
            "timestamp": timestamps,
            "open": paths["open"][i], "high": paths["high"][i],
            "low": paths["low"][i], "close": paths["close"][i],
            "volume": volume,
        }))
    return frames
