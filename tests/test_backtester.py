from tradingbot.backtester import run_backtest
from tradingbot.config import Config


def test_backtest_runs_and_returns_metrics(trending_ohlcv):
    config = Config()
    result = run_backtest(trending_ohlcv, config)
    assert result.equity_curve.iloc[0] > 0
    assert isinstance(result.total_return_pct, float)
    assert isinstance(result.max_drawdown_pct, float)
    assert result.num_trades >= 0


def test_backtest_never_spends_more_cash_than_available(trending_ohlcv):
    config = Config()
    result = run_backtest(trending_ohlcv, config)
    # equity should never go negative given position sizing caps
    assert (result.equity_curve > 0).all()


def test_max_drawdown_is_non_positive(choppy_ohlcv):
    config = Config()
    result = run_backtest(choppy_ohlcv, config)
    assert result.max_drawdown_pct <= 0
