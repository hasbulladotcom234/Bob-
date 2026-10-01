"""Parameter search over your strategy's PARAM_GRID, with a train/test split.

Every combination is backtested on the first part of the data (train); only
the winner is then run once on the last part (test), which the search never
saw. If the test score is far worse than train, the "best" parameters were
fitting noise. The more combinations you try, the luckier the winner looks
on train, so keep PARAM_GRID small.
"""
import itertools
from dataclasses import dataclass

import pandas as pd

from .backtester import run_backtest
from .config import Config
from .strategy import load_strategy


@dataclass
class OptimizationResult:
    best_params: dict
    train_score: float
    test_score: float
    test_return_pct: float
    test_buy_hold_pct: float
    combos_tried: int
    leaderboard: pd.DataFrame


def score(result) -> float:
    """Risk-adjusted return minus a drawdown penalty. Parameter sets with
    almost no trades are statistically meaningless, so they're ruled out."""
    if result.num_trades < 5:
        return -999.0
    return result.sharpe - abs(result.max_drawdown_pct) / 100.0


def grid_search(df: pd.DataFrame, config: Config, strat=None, grid: dict = None,
                train_fraction: float = 0.7) -> OptimizationResult:
    strat = strat or load_strategy(config.strategy_module)
    grid = grid if grid is not None else getattr(strat, "PARAM_GRID", {})
    if not grid:
        raise RuntimeError("Your strategy has no PARAM_GRID to search.")

    split = int(len(df) * train_fraction)
    train_df = df.iloc[:split].reset_index(drop=True)
    test_df = df.iloc[split:].reset_index(drop=True)

    keys = list(grid)
    rows, best = [], None
    for values in itertools.product(*(grid[k] for k in keys)):
        params = dict(strat.PARAMS, **dict(zip(keys, values)))
        result = run_backtest(train_df, config, strat, params, check_lookahead=False)
        s = score(result)
        rows.append({**dict(zip(keys, values)), "train_score": s,
                     "train_return_pct": result.total_return_pct,
                     "train_max_drawdown_pct": result.max_drawdown_pct,
                     "train_trades": result.num_trades})
        if best is None or s > best[0]:
            best = (s, params)

    best_score, best_params = best
    if best_score == -999.0:
        raise RuntimeError("No parameter combination made enough trades to score.")

    test = run_backtest(test_df, config, strat, best_params, check_lookahead=False)
    return OptimizationResult(
        best_params=best_params,
        train_score=best_score,
        test_score=score(test),
        test_return_pct=test.total_return_pct,
        test_buy_hold_pct=test.buy_hold_return_pct,
        combos_tried=len(rows),
        leaderboard=pd.DataFrame(rows).sort_values("train_score", ascending=False).reset_index(drop=True),
    )
