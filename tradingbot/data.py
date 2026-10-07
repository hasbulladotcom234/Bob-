"""Market data fetching via ccxt."""
import time

import ccxt
import pandas as pd

COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


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


def _to_frame(raw) -> pd.DataFrame:
    df = pd.DataFrame(raw, columns=COLUMNS)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def _alpaca_public_ohlcv(exchange, symbol, timeframe, since, limit):
    """Alpaca's crypto price data is public, but ccxt's normal fetch_ohlcv first
    loads the asset list, which needs API keys. Call the public bars endpoint
    directly so exploring and backtesting work without any keys."""
    tf_ms = exchange.parse_timeframe(timeframe) * 1000
    if since is None:  # Alpaca defaults to "start of today" otherwise
        since = exchange.milliseconds() - limit * tf_ms
    request = {"loc": "us", "symbols": symbol, "start": exchange.iso8601(since), "limit": limit,
               "timeframe": exchange.timeframes.get(timeframe, timeframe)}
    rows = []
    while len(rows) < limit:
        response = exchange.marketPublicGetV1beta3CryptoLocBars(request)
        page = (response.get("bars") or {}).get(symbol) or []
        rows.extend(exchange.parse_ohlcv(bar) for bar in page)
        token = response.get("next_page_token")
        if not page or not token:
            break
        request["page_token"] = token
    return rows[:limit]


def _raw_ohlcv(exchange, symbol, timeframe, since, limit):
    if getattr(exchange, "id", None) == "alpaca" and not getattr(exchange, "apiKey", None):
        return _alpaca_public_ohlcv(exchange, symbol, timeframe, since, limit)
    return exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, since=since)


def fetch_ohlcv(exchange, symbol: str, timeframe: str, limit: int = 500,
                since: int = None) -> pd.DataFrame:
    """One request's worth of candles. `exchange` is a ccxt exchange or an id string."""
    if isinstance(exchange, str):
        exchange = make_exchange(exchange)
    return _to_frame(_raw_ohlcv(exchange, symbol, timeframe, since, limit))


def fetch_history(exchange, symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
    """The most recent `bars` candles, paging through as many requests as needed
    (exchanges cap how many candles one request returns)."""
    if isinstance(exchange, str):
        exchange = make_exchange(exchange)
    tf_ms = exchange.parse_timeframe(timeframe) * 1000
    now = exchange.milliseconds()
    since = now - bars * tf_ms
    raw = []
    while since < now:
        chunk = _raw_ohlcv(exchange, symbol, timeframe, since, 1000)
        if not chunk:
            break
        raw.extend(chunk)
        next_since = chunk[-1][0] + tf_ms
        if next_since <= since:
            break
        since = next_since
    return _to_frame(raw).tail(bars).reset_index(drop=True)


def closed_bars(df: pd.DataFrame, timeframe_seconds: int, now: float = None) -> pd.DataFrame:
    """Drop the newest candle if it hasn't finished yet. Exchanges return the
    still-forming bar last; deciding on it means acting on a 'close' that can
    still change, so signals would flicker on and off within the hour."""
    if df.empty:
        return df
    now = time.time() if now is None else now
    bar_end = df["timestamp"].iloc[-1].timestamp() + timeframe_seconds
    return df.iloc[:-1] if bar_end > now else df
