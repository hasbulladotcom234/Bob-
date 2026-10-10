"""Local market-data store.

Candles are downloaded once and kept as one file per symbol:

    data/<exchange>/<timeframe>/<BASE-QUOTE>.parquet   (if pyarrow is installed)
    data/<exchange>/<timeframe>/<BASE-QUOTE>.csv       (otherwise)

Parquet is smaller and faster; CSV needs nothing extra installed. Either is
read back transparently, and saving converts a symbol to the preferred format.

Updates only fetch what's new, so research is fast, works offline, and two
runs on the same data give the same answer. Only finished bars are stored.
Timestamps are bar *start* times in UTC.
"""
import os
from dataclasses import dataclass

import pandas as pd

from .cleaning import clean_bars
from .data import COLUMNS, closed_bars, fetch_range

DEFAULT_ROOT = os.getenv("DATA_DIR", "data")
DEFAULT_SINCE = os.getenv("DATA_SINCE", "2021-01-01")
REFETCH_BARS = 3  # re-download the last few stored bars in case the exchange revised them
FORMATS = (".parquet", ".csv")


def parquet_available() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


@dataclass
class UpdateResult:
    symbol: str
    new_rows: int
    total_rows: int
    first: pd.Timestamp = None
    last: pd.Timestamp = None


class BarStore:
    def __init__(self, root: str = DEFAULT_ROOT, exchange_id: str = "alpaca", use_parquet: bool = None):
        self.root = root
        self.exchange_id = exchange_id
        self.use_parquet = parquet_available() if use_parquet is None else use_parquet

    def path(self, symbol: str, timeframe: str, ext: str = None) -> str:
        ext = ext or (".parquet" if self.use_parquet else ".csv")
        return os.path.join(self.root, self.exchange_id, timeframe, symbol.replace("/", "-") + ext)

    def _existing_path(self, symbol: str, timeframe: str):
        preferred = (".parquet", ".csv") if self.use_parquet else (".csv", ".parquet")
        for ext in preferred:
            path = self.path(symbol, timeframe, ext)
            if os.path.exists(path) and (ext == ".csv" or self.use_parquet):
                return path
        return None

    def symbols(self, timeframe: str) -> list:
        folder = os.path.join(self.root, self.exchange_id, timeframe)
        if not os.path.isdir(folder):
            return []
        names = set()
        for f in os.listdir(folder):
            for ext in FORMATS:
                if f.endswith(ext):
                    names.add(f[:-len(ext)].replace("-", "/"))
        return sorted(names)

    def load(self, symbol: str, timeframe: str, start=None, end=None, clean: bool = True) -> pd.DataFrame:
        """Stored candles for one symbol, optionally between start and end
        (inclusive). With clean=True (the default) bad prints are removed;
        see tradingbot/cleaning.py. The file on disk is never changed."""
        path = self._existing_path(symbol, timeframe)
        if path is None:
            if os.path.exists(self.path(symbol, timeframe, ".parquet")):
                raise RuntimeError(f"{symbol} is stored as Parquet but pyarrow isn't installed. "
                                   f"Install it (pip install pyarrow) or delete the file to re-download.")
            return pd.DataFrame(columns=COLUMNS)
        if path.endswith(".parquet"):
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path, parse_dates=["timestamp"], float_precision="round_trip")
            df[COLUMNS[1:]] = df[COLUMNS[1:]].astype(float)
        if clean:
            df = clean_bars(df)[0]
        if start is not None:
            df = df[df["timestamp"] >= pd.Timestamp(start)]
        if end is not None:
            df = df[df["timestamp"] <= pd.Timestamp(end)]
        return df.reset_index(drop=True)

    def save(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        path = self.path(symbol, timeframe)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        if self.use_parquet:
            df[COLUMNS].to_parquet(tmp, index=False)
        else:
            df[COLUMNS].to_csv(tmp, index=False)  # full float precision, round-trips exactly
        os.replace(tmp, path)  # atomic: a crash mid-write never leaves a half-written file
        for ext in FORMATS:  # drop a copy in the other format so there's one source of truth
            other = self.path(symbol, timeframe, ext)
            if other != path and os.path.exists(other):
                os.remove(other)

    def update(self, exchange, symbol: str, timeframe: str, since: str = DEFAULT_SINCE,
               now: float = None) -> UpdateResult:
        """Download new bars for one symbol and merge them into the store."""
        tf_seconds = exchange.parse_timeframe(timeframe)
        existing = self.load(symbol, timeframe, clean=False)
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
