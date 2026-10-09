"""Local market-data store.

Candles are downloaded once and kept as Parquet files:

    data/<exchange>/<timeframe>/<BASE-QUOTE>.parquet

Updates only fetch what's new, so research is fast, works offline, and two
runs on the same data give the same answer. Only finished bars are stored.
Timestamps are bar *start* times in UTC.
"""
import os
from dataclasses import dataclass

import pandas as pd

from .data import COLUMNS, closed_bars, fetch_range

DEFAULT_ROOT = os.getenv("DATA_DIR", "data")
DEFAULT_SINCE = os.getenv("DATA_SINCE", "2021-01-01")
REFETCH_BARS = 3  # re-download the last few stored bars in case the exchange revised them


@dataclass
class UpdateResult:
    symbol: str
    new_rows: int
    total_rows: int
    first: pd.Timestamp = None
    last: pd.Timestamp = None


class BarStore:
    def __init__(self, root: str = DEFAULT_ROOT, exchange_id: str = "alpaca"):
        self.root = root
        self.exchange_id = exchange_id

    def path(self, symbol: str, timeframe: str) -> str:
        return os.path.join(self.root, self.exchange_id, timeframe, symbol.replace("/", "-") + ".parquet")

    def symbols(self, timeframe: str) -> list:
        folder = os.path.join(self.root, self.exchange_id, timeframe)
        if not os.path.isdir(folder):
            return []
        return sorted(f[:-len(".parquet")].replace("-", "/") for f in os.listdir(folder) if f.endswith(".parquet"))

    def load(self, symbol: str, timeframe: str, start=None, end=None) -> pd.DataFrame:
        """Stored candles for one symbol, optionally between start and end (inclusive)."""
        path = self.path(symbol, timeframe)
        if not os.path.exists(path):
            return pd.DataFrame(columns=COLUMNS)
        df = pd.read_parquet(path)
        if start is not None:
            df = df[df["timestamp"] >= pd.Timestamp(start)]
        if end is not None:
            df = df[df["timestamp"] <= pd.Timestamp(end)]
        return df.reset_index(drop=True)

    def save(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        path = self.path(symbol, timeframe)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        df[COLUMNS].to_parquet(tmp, index=False)
        os.replace(tmp, path)  # atomic: a crash mid-write never leaves a half-written file

    def update(self, exchange, symbol: str, timeframe: str, since: str = DEFAULT_SINCE,
               now: float = None) -> UpdateResult:
        """Download new bars for one symbol and merge them into the store."""
        tf_seconds = exchange.parse_timeframe(timeframe)
        existing = self.load(symbol, timeframe)
        if existing.empty:
            start = pd.Timestamp(since)
        else:
            start = existing["timestamp"].iloc[-1] - pd.Timedelta(seconds=REFETCH_BARS * tf_seconds)
        start_ms = int((start - pd.Timestamp(0)).total_seconds() * 1000)

        fresh = closed_bars(fetch_range(exchange, symbol, timeframe, start_ms), tf_seconds, now)
        if fresh.empty and existing.empty:
            return UpdateResult(symbol, 0, 0)

        merged = pd.concat([existing, fresh]) if not existing.empty else fresh
        merged = merged.drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)
        new_rows = len(merged) - len(existing)
        self.save(symbol, timeframe, merged)
        return UpdateResult(symbol, new_rows, len(merged), merged["timestamp"].iloc[0], merged["timestamp"].iloc[-1])

    def summary(self, timeframe: str) -> pd.DataFrame:
        rows = []
        for sym in self.symbols(timeframe):
            df = self.load(sym, timeframe)
            rows.append({"symbol": sym, "rows": len(df),
                         "first": df["timestamp"].iloc[0] if len(df) else None,
                         "last": df["timestamp"].iloc[-1] if len(df) else None})
        return pd.DataFrame(rows)
