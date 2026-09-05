"""Parameter search ("training") for the strategy, with a train/test split
so we don't just curve-fit to one dataset and call it done.

This is grid search over a handful of strategy parameters, scored on
walk-forward out-of-sample data. It is not machine learning; it is the
honest version of "let the model train itself" for a rule-based strategy --
there is no model whose weights are being fit here, only a small parameter
space being searched.
"""
import copy
import itertools
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from .backtester import run_backtest
from .config import Config, StrategyParams


@dataclass
class OptimizationResult:
    best_params: StrategyParams
    train_score: float
    test_score: float
    leaderboard: pd.DataFrame


def _score(result) -> float:
    """Reward risk-adjusted return, penalize deep drawdowns, ignore
    parameter sets with almost no trades (statistically meaningless)."""
    if result.num_trades < 5:
        return -999.0
    return result.sharpe - abs(result.max_drawdown_pct) / 100.0


def grid_search(
    df: pd.DataFrame,
    base_config: Config,
    fast_ema_choices: Iterable[int] = (8, 12, 20),
    slow_ema_choices: Iterable[int] = (26, 50, 100),
    rsi_period_choices: Iterable[int] = (14, 21),
    stop_loss_choices: Iterable[float] = (0.02, 0.03, 0.05),
    take_profit_choices: Iterable[float] = (0.04, 0.06, 0.10),
    train_fraction: float = 0.7,
) -> OptimizationResult:
    split_idx = int(len(df) * train_fraction)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    rows = []
    best = None

    for fast, slow, rsi_p, sl, tp in itertools.product(
        fast_ema_choices, slow_ema_choices, rsi_period_choices, stop_loss_choices, take_profit_choices
    ):
        if fast >= slow:
            continue

        params = StrategyParams(
            fast_ema=fast, slow_ema=slow, rsi_period=rsi_p,
            stop_loss_pct=sl, take_profit_pct=tp,
        )
        cfg = copy.deepcopy(base_config)
        cfg.strategy = params

        train_result = run_backtest(train_df, cfg)
        train_score = _score(train_result)

        rows.append({
            "fast_ema": fast, "slow_ema": slow, "rsi_period": rsi_p,
            "stop_loss_pct": sl, "take_profit_pct": tp,
            "train_score": train_score,
            "train_return_pct": train_result.total_return_pct,
            "train_max_drawdown_pct": train_result.max_drawdown_pct,
            "train_trades": train_result.num_trades,
        })

        if best is None or train_score > best[0]:
            best = (train_score, params)

    leaderboard = pd.DataFrame(rows).sort_values("train_score", ascending=False).reset_index(drop=True)

    if best is None:
        raise RuntimeError("No valid parameter combination produced enough trades to score.")

    best_score, best_params = best
    cfg = copy.deepcopy(base_config)
    cfg.strategy = best_params
    test_result = run_backtest(test_df, cfg)
    test_score = _score(test_result)

    return OptimizationResult(
        best_params=best_params,
        train_score=best_score,
        test_score=test_score,
        leaderboard=leaderboard,
    )
