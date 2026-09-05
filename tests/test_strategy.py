from tradingbot.config import StrategyParams
from tradingbot.strategy import generate_signals


def test_generate_signals_has_expected_columns(trending_ohlcv):
    data = generate_signals(trending_ohlcv, StrategyParams())
    for col in ("ema_fast", "ema_slow", "rsi", "long_entry", "long_exit"):
        assert col in data.columns


def test_no_simultaneous_entry_and_exit(trending_ohlcv):
    data = generate_signals(trending_ohlcv, StrategyParams())
    assert not ((data["long_entry"]) & (data["long_exit"])).any()


def test_entries_require_bullish_rsi(trending_ohlcv):
    params = StrategyParams(rsi_bull_threshold=50.0)
    data = generate_signals(trending_ohlcv, params)
    entries = data[data["long_entry"]]
    assert (entries["rsi"] > 50.0).all()
