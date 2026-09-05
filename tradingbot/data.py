"""Market data fetching via ccxt (public endpoints, no API key required)."""
import ccxt
import pandas as pd


def fetch_ohlcv(exchange_id: str, symbol: str, timeframe: str, limit: int = 500,
                 since: int = None) -> pd.DataFrame:
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({"enableRateLimit": True})
    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, since=since)
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df
