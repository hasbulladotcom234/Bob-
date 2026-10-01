import types

import pandas as pd

from tradingbot.backtester import periods_per_year, run_backtest
from tradingbot.config import Config


def test_backtest_runs_and_returns_metrics(trending_ohlcv):
    result = run_backtest(trending_ohlcv, Config())
    assert result.equity_curve.iloc[0] > 0
    assert isinstance(result.total_return_pct, float)
    assert isinstance(result.buy_hold_return_pct, float)
    assert result.num_trades >= 0
    assert result.lookahead_warning == []


def test_equity_never_negative(trending_ohlcv):
    result = run_backtest(trending_ohlcv, Config())
    assert (result.equity_curve > 0).all()


def test_max_drawdown_is_non_positive(choppy_ohlcv):
    assert run_backtest(choppy_ohlcv, Config()).max_drawdown_pct <= 0


def test_orders_fill_at_next_bar_open(trending_ohlcv):
    """A buy decided at bar 10's close must fill at bar 11's open."""
    df = trending_ohlcv.copy()
    df["open"] = df["close"].shift(1).fillna(df["close"]) * 1.01  # make open distinct from prev close
    strat = types.SimpleNamespace(
        PARAMS={"stop_loss_pct": 0.5, "take_profit_pct": 10.0},
        indicators=lambda c, p: c.assign(n=range(len(c))),
        decide=lambda bar, prev, pos, p: "buy" if bar.n == 10 else ("sell" if bar.n == 20 else "hold"),
    )
    config = Config()
    config.risk.slippage_pct = 0.0
    result = run_backtest(df, config, strat)
    trade = result.trades.iloc[0]
    assert trade["entry_time"] == df["timestamp"].iloc[11]
    assert trade["entry_price"] == df["open"].iloc[11]
    assert trade["exit_time"] == df["timestamp"].iloc[21]


def test_periods_per_year_matches_timeframe():
    hourly = pd.Series(pd.date_range("2024-01-01", periods=10, freq="h"))
    daily = pd.Series(pd.date_range("2024-01-01", periods=10, freq="D"))
    assert periods_per_year(hourly) == 365 * 24
    assert periods_per_year(daily) == 365
