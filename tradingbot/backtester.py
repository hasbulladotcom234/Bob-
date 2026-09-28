"""Vectorized-ish event-driven backtester for the trend-following strategy.

Simulates one position at a time (long-only), applying fees, slippage,
stop-loss/take-profit exits, and the same RiskManager used live.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config
from .data import bars_per_year
from .risk import RiskManager
from .strategy import generate_signals


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: pd.DataFrame
    total_return_pct: float
    max_drawdown_pct: float
    win_rate_pct: float
    num_trades: int
    sharpe: float


def run_backtest(df: pd.DataFrame, config: Config) -> BacktestResult:
    """df must have columns: timestamp, open, high, low, close, volume,
    sorted ascending by timestamp."""
    data = generate_signals(df, config.strategy, config.round_trip_cost())
    risk = RiskManager(params=config.risk)

    equity = config.starting_balance
    cash = config.starting_balance
    position_qty = 0.0
    entry_price = None
    stop_price = None
    take_price = None

    equity_curve = []
    trade_log = []

    fee = config.risk.taker_fee_pct
    slip = config.risk.slippage_pct

    days = pd.to_datetime(data["timestamp"]).dt.date if "timestamp" in data else [None] * len(data)
    columns = zip(data["close"], data["high"], data["low"], data["long_entry"], data["long_exit"], days)

    for price, high, low, long_entry, long_exit, day in columns:
        # mark-to-market equity
        equity = cash + position_qty * price
        halted = risk.check_daily_circuit_breaker(day, equity) if day else False

        # manage open position: stop-loss / take-profit / signal exit
        if position_qty > 0:
            hit_stop = low <= stop_price
            hit_take = high >= take_price
            signal_exit = long_exit

            if hit_stop or hit_take or signal_exit:
                exit_price = stop_price if hit_stop else (take_price if hit_take else price)
                exit_price *= (1 - slip)
                proceeds = position_qty * exit_price * (1 - fee)
                pnl = proceeds - (position_qty * entry_price)
                cash += proceeds
                trade_log.append({
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "qty": position_qty,
                    "pnl": pnl,
                    "reason": "stop_loss" if hit_stop else ("take_profit" if hit_take else "signal_exit"),
                })
                position_qty = 0.0
                entry_price = stop_price = take_price = None

        # consider new entry
        if position_qty == 0 and long_entry and not halted:
            candidate_entry = price * (1 + slip)
            candidate_stop = risk.stop_loss_price(candidate_entry, config.strategy.stop_loss_pct)
            qty = risk.position_size(equity, candidate_entry, candidate_stop)
            cost = qty * candidate_entry * (1 + fee)
            if qty > 0 and cost <= cash:
                position_qty = qty
                entry_price = candidate_entry
                stop_price = candidate_stop
                take_price = risk.take_profit_price(candidate_entry, config.strategy.take_profit_pct)
                cash -= cost

        equity = cash + position_qty * price
        equity_curve.append(equity)

    equity_series = pd.Series(equity_curve, index=data.index)
    trades_df = pd.DataFrame(trade_log)

    total_return_pct = (equity_series.iloc[-1] / config.starting_balance - 1) * 100 if len(equity_series) else 0.0

    running_max = equity_series.cummax()
    drawdown = (equity_series - running_max) / running_max
    max_drawdown_pct = drawdown.min() * 100 if len(drawdown) else 0.0

    if not trades_df.empty:
        win_rate_pct = (trades_df["pnl"] > 0).mean() * 100
    else:
        win_rate_pct = 0.0

    returns = equity_series.pct_change().dropna()
    sharpe = (returns.mean() / returns.std() * np.sqrt(bars_per_year(config.timeframe))) if returns.std() not in (0, None) and len(returns) > 1 else 0.0
    if pd.isna(sharpe):
        sharpe = 0.0

    return BacktestResult(
        equity_curve=equity_series,
        trades=trades_df,
        total_return_pct=total_return_pct,
        max_drawdown_pct=max_drawdown_pct,
        win_rate_pct=win_rate_pct,
        num_trades=len(trades_df),
        sharpe=sharpe,
    )
