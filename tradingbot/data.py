"""Market data fetching via ccxt."""
import ccxt
import pandas as pd


def make_exchange(exchange_id: str, api_key: str = "", api_secret: str = "",
                  sandbox: bool = False) -> ccxt.Exchange:
    """Build a ccxt exchange client. With sandbox=True it points at the
    exchange's paper/testnet endpoints (ccxt raises if the exchange has none),
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
    # Some exchanges (e.g. Alpaca) need keys even to load markets, so pass them
    # along for data too. Sandbox keys only work against sandbox endpoints.
    return make_exchange(config.exchange_id, config.api_key, config.api_secret,
                         sandbox=config.is_sandbox())


def fetch_ohlcv(exchange, symbol: str, timeframe: str, limit: int = 500,
                 since: int = None) -> pd.DataFrame:
    """`exchange` is a ccxt exchange instance or an exchange id string."""
    if isinstance(exchange, str):
        exchange = make_exchange(exchange)
    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, since=since)
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df
