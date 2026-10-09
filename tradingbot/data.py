"""Market data fetching via ccxt: exchange setup, ranged downloads with
paging and retries, and closed-bar filtering.

For research, don't call these directly; use tradingbot.store, which keeps a
local copy and only downloads what's new.
"""
import logging
import time

import ccxt
import pandas as pd

log = logging.getLogger("tradingbot")

COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
ALPACA_PAGE_LIMIT = 10000   # max bars Alpaca returns per request
GENERIC_PAGE_LIMIT = 1000   # safe page size for other exchanges


def make_exchange(exchange_id: str, api_key: str = "", api_secret: str = "",
                  sandbox: bool = False) -> ccxt.Exchange:
    """Build a ccxt exchange client. With sandbox=True it points at the
    exchange's paper/testnet endpoints (refusing if the exchange has none),
    so it can never silently fall back to the real-money API."""
    exchange_class = getattr(ccxt, exchange_id)
    options = {"enableRateLimit": True}
    if api_key and api_secret:
        options.update({"apiKey": api_key, "secret": api_secret})
    exchange = exchange_class(options)
    if sandbox:
        if not exchange.urls.get("test"):
            raise RuntimeError(f"{exchange_id} has no sandbox/paper endpoint in ccxt; use MODE=paper instead.")
        exchange.set_sandbox_mode(True)
    return exchange


def exchange_from_config(config) -> ccxt.Exchange:
    # Pass keys along when we have them (needed for trading, and by some
    # exchanges for data). Sandbox keys only work against sandbox endpoints.
    return make_exchange(config.exchange_id, config.api_key, config.api_secret,
                         sandbox=config.is_sandbox())


def with_retries(fn, *args, attempts: int = 5, base_delay: float = 1.0, sleep=None, **kwargs):
    """Call fn, retrying network hiccups and rate limits with exponential
    backoff (1s, 2s, 4s, ...). Errors that retrying can't fix (bad keys, unknown
    symbol) are raised immediately."""
    for attempt in range(attempts):
        try:
            return fn(*args, **kwargs)
        except ccxt.NetworkError as err:  # includes RateLimitExceeded, RequestTimeout
            if attempt == attempts - 1:
                raise
            delay = base_delay * 2 ** attempt
            log.warning("%s: %s; retrying in %.0fs", type(err).__name__, err, delay)
            (sleep or time.sleep)(delay)


def _to_frame(raw) -> pd.DataFrame:
    df = pd.DataFrame(raw, columns=COLUMNS)
    df = df.drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    for col in COLUMNS[1:]:
        df[col] = df[col].astype(float)
    return df


def _is_keyless_alpaca(exchange) -> bool:
    # Alpaca's crypto price data is public, but ccxt's normal fetch_ohlcv first
    # loads the asset list, which needs API keys. Without keys we call the
    # public bars endpoint directly.
    return getattr(exchange, "id", None) == "alpaca" and not getattr(exchange, "apiKey", None)


def _alpaca_range(exchange, symbol, timeframe, start_ms, end_ms):
    request = {"loc": "us", "symbols": symbol, "start": exchange.iso8601(start_ms),
               "limit": ALPACA_PAGE_LIMIT, "timeframe": exchange.timeframes.get(timeframe, timeframe)}
    if end_ms is not None:
        request["end"] = exchange.iso8601(end_ms)
    rows = []
    while True:
        response = with_retries(exchange.marketPublicGetV1beta3CryptoLocBars, dict(request))
        page = (response.get("bars") or {}).get(symbol) or []
        rows.extend(exchange.parse_ohlcv(bar) for bar in page)
        token = response.get("next_page_token")
        if not page or not token:
            return rows
        request["page_token"] = token


def _generic_range(exchange, symbol, timeframe, start_ms, end_ms):
    tf_ms = exchange.parse_timeframe(timeframe) * 1000
    end_ms = exchange.milliseconds() if end_ms is None else end_ms
    since, rows = start_ms, []
    while since < end_ms:
        chunk = with_retries(exchange.fetch_ohlcv, symbol, timeframe=timeframe, since=since,
                             limit=GENERIC_PAGE_LIMIT)
        if not chunk:
            break
        rows.extend(chunk)
        next_since = chunk[-1][0] + tf_ms
        if next_since <= since:
            break
        since = next_since
    return [r for r in rows if r[0] <= end_ms]


def fetch_range(exchange, symbol: str, timeframe: str, start_ms: int, end_ms: int = None) -> pd.DataFrame:
    """All candles from start_ms to end_ms (default: now), paging through as
    many requests as needed. Empty DataFrame if the symbol has no data."""
    if isinstance(exchange, str):
        exchange = make_exchange(exchange)
    fetch = _alpaca_range if _is_keyless_alpaca(exchange) else _generic_range
    return _to_frame(fetch(exchange, symbol, timeframe, start_ms, end_ms))


def fetch_history(exchange, symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
    """The most recent `bars` candles."""
    if isinstance(exchange, str):
        exchange = make_exchange(exchange)
    tf_ms = exchange.parse_timeframe(timeframe) * 1000
    start = exchange.milliseconds() - bars * tf_ms
    return fetch_range(exchange, symbol, timeframe, start).tail(bars).reset_index(drop=True)


def fetch_ohlcv(exchange, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
    """Recent candles for the live loop. `exchange` is a ccxt exchange or an id string."""
    if isinstance(exchange, str):
        exchange = make_exchange(exchange)
    if _is_keyless_alpaca(exchange):
        return fetch_history(exchange, symbol, timeframe, limit)
    return _to_frame(with_retries(exchange.fetch_ohlcv, symbol, timeframe=timeframe, limit=limit))


def closed_bars(df: pd.DataFrame, timeframe_seconds: int, now: float = None) -> pd.DataFrame:
    """Drop candles that haven't finished yet. Exchanges return the
    still-forming bar last; deciding on it means acting on a 'close' that can
    still change, so signals would flicker on and off within the hour."""
    if df.empty:
        return df
    now = time.time() if now is None else now
    start_seconds = (df["timestamp"] - pd.Timestamp(0)).dt.total_seconds()
    return df[start_seconds + timeframe_seconds <= now].reset_index(drop=True)


def research_exchange(config) -> ccxt.Exchange:
    """Exchange client for downloading history. For Alpaca this is always the
    keyless public data endpoint (works in every mode, nothing to configure)."""
    if config.exchange_id == "alpaca":
        return make_exchange("alpaca")
    return make_exchange(config.exchange_id, config.api_key, config.api_secret)
