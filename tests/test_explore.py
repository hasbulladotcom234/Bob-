import numpy as np
import pandas as pd

from tests.conftest import make_synthetic_ohlcv
from tradingbot.explore import autocorr_table, forward_returns_by_condition, report


def test_report_runs(trending_ohlcv):
    text = report(trending_ohlcv)
    assert "Buy & hold" in text and "Hour of day" in text


def test_autocorr_detects_momentum():
    rng = np.random.default_rng(0)
    r = np.zeros(5000)
    for i in range(1, 5000):
        r[i] = 0.3 * r[i - 1] + rng.normal(0, 0.01)
    table, band = autocorr_table(pd.Series(r), lags=(1,))
    assert table["return_autocorr"].iloc[0] > 0.25 and table["significant"].iloc[0]


def test_forward_returns_by_condition():
    df = make_synthetic_ohlcv(n=100)
    cond = pd.Series([True] * 50 + [False] * 50)
    t = forward_returns_by_condition(df, cond, 1)
    assert t["bars"].sum() == 99
