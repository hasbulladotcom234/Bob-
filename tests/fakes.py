"""Test doubles that mimic real exchange APIs closely enough to exercise
paging, timestamps and error handling without network access."""
import ccxt
import numpy as np


def make_bars(start_ms, n, tf_ms=3600_000, seed=0, start_price=100.0, skip=()):
    """Synthetic candles as [ts, o, h, l, c, v] lists; indices in `skip` are left out (no trades)."""
    rng = np.random.default_rng(seed)
    close = start_price * np.cumprod(1 + rng.normal(0, 0.01, n))
    rows = []
    for i in range(n):
        if i in skip:
            continue
        o = float(close[i - 1]) if i else float(start_price)
        c = float(close[i])  # plain floats, like parsed JSON (ccxt drops numpy types)
        rows.append([start_ms + i * tf_ms, o, max(o, c) * 1.001, min(o, c) * 0.999, c, 10.0 + i % 7])
    return rows


class FakeAlpacaData:
    """A real ccxt.alpaca instance whose public crypto-bars endpoint is served
    from memory, with Alpaca's response shape and page-token paging."""

    def __init__(self, bars_by_symbol, page_size=500, now_ms=None, fail_first=0):
        self.exchange = ccxt.alpaca()
        self.bars = bars_by_symbol
        self.page_size = page_size
        self.requests = []
        self.fail_first = fail_first
        self.exchange.marketPublicGetV1beta3CryptoLocBars = self._bars
        if now_ms is not None:
            self.exchange.milliseconds = lambda: now_ms

    def _bars(self, request):
        self.requests.append(dict(request))
        if self.fail_first > 0:
            self.fail_first -= 1
            raise ccxt.RateLimitExceeded("429 too many requests")
        sym = request["symbols"]
        if sym not in self.bars:
            return {"bars": {}, "next_page_token": None}
        start = self.exchange.parse8601(request["start"])
        end = self.exchange.parse8601(request["end"]) if "end" in request else float("inf")
        rows = [b for b in self.bars[sym] if start <= b[0] <= end]
        offset = int(request.get("page_token", 0))
        page = rows[offset:offset + self.page_size]
        token = str(offset + self.page_size) if offset + self.page_size < len(rows) else None
        return {
            "bars": {sym: [{"t": self.exchange.iso8601(b[0]), "o": b[1], "h": b[2], "l": b[3],
                            "c": b[4], "v": b[5], "n": 3, "vw": b[4]} for b in page]},
            "next_page_token": token,
        }
