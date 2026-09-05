import numpy as np
import pandas as pd
import pytest


def make_synthetic_ohlcv(n=500, seed=42, start_price=100.0, trend=0.0005, vol=0.01):
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=trend, scale=vol, size=n)
    close = start_price * np.cumprod(1 + returns)
    open_ = np.roll(close, 1)
    open_[0] = start_price
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.002, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.002, n))
    volume = rng.uniform(100, 1000, n)
    timestamp = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame({
        "timestamp": timestamp, "open": open_, "high": high,
        "low": low, "close": close, "volume": volume,
    })


@pytest.fixture
def trending_ohlcv():
    return make_synthetic_ohlcv(trend=0.001)


@pytest.fixture
def choppy_ohlcv():
    return make_synthetic_ohlcv(trend=0.0, vol=0.02)
