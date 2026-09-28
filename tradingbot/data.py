"""Market data fetching via ccxt (public endpoints, no API key required)."""
import ccxt
import pandas as pd

# Largest page every supported exchange accepts (Coinbase caps at 300).
PAGE_SIZE = 300


def timeframe_seconds(timeframe: str) -> int:
    return ccxt.Exchange.parse_timeframe(timeframe)


def bars_per_year(timeframe: str) -> float:
    return 365 * 24 * 3600 / timeframe_seconds(timeframe)


def fetch_ohlcv(exchange_id: str, symbol: str, timeframe: str, limit: int = 500,
                 since: int = None) -> pd.DataFrame:
    """Fetch up to `limit` candles, paging through the exchange's per-request
    cap. Starts at `since` (ms) if given, otherwise `limit` bars before now.

    Some exchanges keep only a short history (Kraken serves the most recent
    720 candles per timeframe), in which case fewer than `limit` come back.
    """
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({"enableRateLimit": True})
    tf_ms = timeframe_seconds(timeframe) * 1000
    cursor = since if since is not None else exchange.milliseconds() - limit * tf_ms

    candles = {}
    while len(candles) < limit:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor,
                                     limit=min(PAGE_SIZE, limit - len(candles)))
        new = [c for c in batch if c[0] not in candles and c[0] >= cursor]
        if not new:
            break
        candles.update((c[0], c) for c in new)
        cursor = max(c[0] for c in new) + tf_ms

    raw = [candles[ts] for ts in sorted(candles)][:limit]
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df
