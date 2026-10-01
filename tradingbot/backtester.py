"""Event-driven backtester: replays history bar by bar and runs your
strategy exactly the way the live bot would.

Timing rules (the main defence against fooling yourself):
- decide() sees a bar only after it has closed.
- Orders fill at the NEXT bar's open, plus slippage and fees.
- Stop-loss / take-profit are checked against each bar's low/high. If a bar
  touches both, we assume the stop hit first (the pessimistic choice). If
  price gaps through a level, the fill is at the open, not the level.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config
from .risk import RiskManager
from .strategy import Bars, decide, load_strategy, lookahead_columns, with_indicators


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: pd.DataFrame
    total_return_pct: float
    buy_hold_return_pct: float   # what simply holding the coin did over the same period
    max_drawdown_pct: float
    win_rate_pct: float
    num_trades: int
    sharpe: float
    exposure_pct: float          # share of bars spent holding a position
    lookahead_warning: list      # indicator columns that peek at future data


def periods_per_year(timestamps: pd.Series) -> float:
    """Crypto trades 24/7, so a year has 365 days of bars."""
    if len(timestamps) < 2:
        return 0.0
    step = pd.Series(timestamps).diff().median().total_seconds()
    return 365 * 24 * 3600 / step if step else 0.0


def run_backtest(df: pd.DataFrame, config: Config, strat=None, params: dict = None,
                 check_lookahead: bool = True) -> BacktestResult:
    """df must have columns timestamp, open, high, low, close, volume, oldest first."""
    strat = strat or load_strategy(config.strategy_module)
    params = dict(strat.PARAMS, **(params or {}))
    data = with_indicators(df, strat, params)
    bars = Bars(data)
    leaks = lookahead_columns(df, strat, params) if check_lookahead else []

    risk = RiskManager(params=config.risk)
    fee, slip = config.risk.taker_fee_pct, config.risk.slippage_pct

    cash = config.starting_balance
    qty = 0.0
    entry = stop = take = None
    entry_time = None
    pending = "hold"       # decision from the previous bar's close, executed at this bar's open
    halted = False
    equity_curve, trades, in_market = [], [], []

    def close_position(px, when, reason):
        nonlocal cash, qty, entry, stop, take
        fill = px * (1 - slip)
        proceeds = qty * fill * (1 - fee)
        cost = qty * entry * (1 + fee)
        cash += proceeds
        trades.append({"entry_time": entry_time, "exit_time": when, "entry_price": entry,
                       "exit_price": fill, "qty": qty, "pnl": proceeds - cost,
                       "return_pct": (proceeds / cost - 1) * 100, "reason": reason})
        qty, entry, stop, take = 0.0, None, None, None

    for i, row in enumerate(bars.rows):
        when = row.timestamp

        # 1. at the open: execute what was decided at the last close
        if pending == "sell" and qty > 0:
            close_position(row.open, when, "signal")
        elif pending == "buy" and qty == 0 and not halted:
            fill = row.open * (1 + slip)
            s = risk.stop_loss_price(fill, params["stop_loss_pct"])
            size = risk.position_size(cash, fill, s)
            size = min(size, cash / (fill * (1 + fee)))
            if size > 0:
                cash -= size * fill * (1 + fee)
                qty, entry, stop, entry_time = size, fill, s, when
                take = risk.take_profit_price(fill, params["take_profit_pct"])

        # 2. during the bar: protective exits
        if qty > 0:
            if row.low <= stop:
                close_position(min(stop, row.open), when, "stop_loss")
            elif row.high >= take:
                close_position(max(take, row.open), when, "take_profit")

        # 3. at the close: mark to market, then ask the strategy
        equity = cash + qty * row.close
        halted = risk.check_daily_circuit_breaker(pd.Timestamp(when).date(), equity)
        pending = decide(bars, i, qty > 0, strat, params)
        equity_curve.append(equity)
        in_market.append(qty > 0)

    if qty > 0:  # mark the open position closed at the final price so stats include it
        close_position(data.iloc[-1]["close"], data.iloc[-1]["timestamp"], "end_of_data")
        equity_curve[-1] = cash

    equity_series = pd.Series(equity_curve, index=data["timestamp"])
    trades_df = pd.DataFrame(trades)

    total_return = (equity_series.iloc[-1] / config.starting_balance - 1) * 100
    buy_hold = (data["close"].iloc[-1] / data["open"].iloc[0] - 1) * 100
    drawdown = equity_series / equity_series.cummax() - 1
    win_rate = (trades_df["pnl"] > 0).mean() * 100 if len(trades_df) else 0.0

    returns = equity_series.pct_change().dropna()
    std = returns.std()
    sharpe = returns.mean() / std * np.sqrt(periods_per_year(data["timestamp"])) if std and std > 0 else 0.0

    return BacktestResult(
        equity_curve=equity_series,
        trades=trades_df,
        total_return_pct=float(total_return),
        buy_hold_return_pct=float(buy_hold),
        max_drawdown_pct=float(drawdown.min() * 100),
        win_rate_pct=float(win_rate),
        num_trades=len(trades_df),
        sharpe=float(0.0 if pd.isna(sharpe) else sharpe),
        exposure_pct=float(np.mean(in_market) * 100) if in_market else 0.0,
        lookahead_warning=leaks,
    )
