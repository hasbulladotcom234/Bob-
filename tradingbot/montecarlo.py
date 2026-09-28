"""Monte Carlo stress test: calibrate a stochastic market model to real
history, simulate many alternative price histories from it, and backtest
the strategy on every one.

A single backtest is one draw from a distribution of outcomes. Running the
strategy across hundreds of plausible histories shows that distribution:
how often it loses money, how bad the bad cases get (VaR / CVaR), and
whether the real-history result was typical or lucky.
"""
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .backtester import run_backtest
from .config import Config
from .simulation import ModelParams, calibrate, paths_to_frames, simulate_paths


@dataclass
class MonteCarloResult:
    model_params: ModelParams
    paths: pd.DataFrame     # one row of backtest metrics per simulated path
    summary: dict


def summarize(paths: pd.DataFrame, tail: float = 0.05) -> dict:
    returns = paths["total_return_pct"]
    var = returns.quantile(tail)
    summary = {
        "paths": len(paths),
        "mean_return_pct": returns.mean(),
        "median_return_pct": returns.median(),
        "prob_loss_pct": (returns < 0).mean() * 100,
        f"var_{int(tail * 100)}_pct": var,
        f"cvar_{int(tail * 100)}_pct": returns[returns <= var].mean(),
        "median_max_drawdown_pct": paths["max_drawdown_pct"].median(),
        "worst_max_drawdown_pct": paths["max_drawdown_pct"].min(),
        "mean_trades": paths["num_trades"].mean(),
        "median_sharpe": paths["sharpe"].median(),
    }
    return {k: float(v) if k != "paths" else v for k, v in summary.items()}


def run_monte_carlo(df: pd.DataFrame, config: Config, model: str = "bates",
                    n_paths: int = 200, n_bars: int = None, drift: str = "historical",
                    seed: int = None, model_params: ModelParams = None) -> MonteCarloResult:
    """Calibrate `model` to df (unless model_params is given), simulate
    n_paths histories of n_bars (default: len(df)) starting from df's first
    close, and backtest config's strategy on each.

    drift="zero" removes the calibrated trend so the test asks whether the
    strategy has an edge beyond riding whatever the market did in the
    calibration window.
    """
    if drift not in ("historical", "zero"):
        raise ValueError("drift must be 'historical' or 'zero'")
    params = model_params or calibrate(df, model)
    if drift == "zero":
        params = replace(params, drift=0.0)

    n_bars = n_bars or len(df)
    rng = np.random.default_rng(seed)
    sim = simulate_paths(params, n_bars, n_paths, float(df["close"].iloc[0]), rng)

    timestamps = pd.to_datetime(df["timestamp"])
    freq = timestamps.diff().median() if len(timestamps) > 1 else pd.Timedelta("1h")
    frames = paths_to_frames(sim, timestamps.iloc[0], freq, rng)

    rows = []
    for frame in frames:
        result = run_backtest(frame, config)
        rows.append({
            "total_return_pct": result.total_return_pct,
            "max_drawdown_pct": result.max_drawdown_pct,
            "win_rate_pct": result.win_rate_pct,
            "num_trades": result.num_trades,
            "sharpe": result.sharpe,
            "buy_and_hold_pct": (frame["close"].iloc[-1] / frame["close"].iloc[0] - 1) * 100,
        })
    paths = pd.DataFrame(rows)
    return MonteCarloResult(model_params=params, paths=paths, summary=summarize(paths))
