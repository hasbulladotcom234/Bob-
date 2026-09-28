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


def test_stochastic_filter_only_removes_entries(choppy_ohlcv):
    unfiltered = generate_signals(choppy_ohlcv, StrategyParams(use_stochastic_filter=False))
    filtered = generate_signals(choppy_ohlcv, StrategyParams(use_stochastic_filter=True), round_trip_cost=0.01)
    assert not (filtered["long_entry"] & ~unfiltered["long_entry"]).any()
    assert (filtered.loc[filtered["long_entry"], "edge"] > 0).all()


def test_stochastic_columns_are_sane(trending_ohlcv):
    data = generate_signals(trending_ohlcv, StrategyParams()).dropna(subset=["p_take"])
    assert data["p_take"].between(0, 1).all()
    assert (data["vol"] > 0).all()
    assert (data["expected_bars"] > 0).all()


def test_higher_costs_never_add_entries(trending_ohlcv):
    cheap = generate_signals(trending_ohlcv, StrategyParams(), round_trip_cost=0.0)
    pricey = generate_signals(trending_ohlcv, StrategyParams(), round_trip_cost=0.02)
    assert pricey["long_entry"].sum() <= cheap["long_entry"].sum()
