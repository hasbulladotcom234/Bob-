import ccxt

from tradingbot import data

HOUR_MS = 3600 * 1000


class FakeExchange:
    """Serves hourly candles from a fixed history, at most `page` per call,
    like a real exchange's per-request cap."""

    def __init__(self, n=1000, page=300, keep_last=None):
        self.history = [[i * HOUR_MS, 1, 2, 0.5, 1.5, 10] for i in range(n)]
        if keep_last:
            self.history = self.history[-keep_last:]
        self.page = page
        self.calls = 0

    def milliseconds(self):
        return 1000 * HOUR_MS

    def fetch_ohlcv(self, symbol, timeframe, since, limit):
        self.calls += 1
        rows = [c for c in self.history if c[0] >= since]
        return rows[: min(limit, self.page)]


def _patch(monkeypatch, fake):
    monkeypatch.setattr(ccxt, "fakex", lambda config: fake, raising=False)


def test_fetch_pages_through_exchange_cap(monkeypatch):
    fake = FakeExchange(page=300)
    _patch(monkeypatch, fake)
    df = data.fetch_ohlcv("fakex", "BTC/USD", "1h", limit=1000)
    assert len(df) == 1000
    assert df["timestamp"].is_monotonic_increasing
    assert df["timestamp"].is_unique
    assert fake.calls == 4


def test_fetch_stops_when_history_runs_out(monkeypatch):
    fake = FakeExchange(keep_last=720)
    _patch(monkeypatch, fake)
    df = data.fetch_ohlcv("fakex", "BTC/USD", "1h", limit=1000)
    assert len(df) == 720


def test_bars_per_year():
    assert data.bars_per_year("1h") == 365 * 24
    assert data.bars_per_year("1d") == 365
