import types

import my_strategy
from tradingbot.strategy import Bars, decide, load_strategy, lookahead_columns, with_indicators


def test_my_strategy_loads_and_adds_indicators(trending_ohlcv):
    strat = load_strategy("my_strategy")
    data = with_indicators(trending_ohlcv, strat, strat.PARAMS)
    for col in ("ema_fast", "ema_slow", "rsi"):
        assert col in data.columns


def test_decisions_are_valid_and_respect_position(trending_ohlcv):
    bars = Bars(with_indicators(trending_ohlcv, my_strategy, my_strategy.PARAMS))
    for i in range(len(bars)):
        assert decide(bars, i, False, my_strategy, my_strategy.PARAMS) in ("buy", "hold")
        assert decide(bars, i, True, my_strategy, my_strategy.PARAMS) in ("sell", "hold")


def test_example_strategy_has_no_lookahead(trending_ohlcv):
    assert lookahead_columns(trending_ohlcv, my_strategy, my_strategy.PARAMS) == []


def test_lookahead_detector_catches_peeking(trending_ohlcv):
    cheat = types.SimpleNamespace(
        PARAMS={},
        indicators=lambda c, p: c.assign(next_close=c["close"].shift(-1)),
        decide=lambda bar, prev, pos, p: "hold",
    )
    assert lookahead_columns(trending_ohlcv, cheat, {}) == ["next_close"]
