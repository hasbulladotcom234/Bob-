import json
import types

import ccxt

from tests.conftest import make_synthetic_ohlcv
from tradingbot.bot import Bot
from tradingbot.brokers import PaperBroker
from tradingbot.config import Config
from tradingbot.data import closed_bars


class FakeExchange:
    """Serves synthetic candles; `n` controls how many bars exist so far."""

    def __init__(self, df):
        self.df, self.n = df, 50

    def parse_timeframe(self, tf):
        return ccxt.Exchange.parse_timeframe(tf)

    def fetch_ohlcv(self, symbol, timeframe, limit, since=None):
        rows = self.df.iloc[:self.n]
        return [[int(r.timestamp.timestamp() * 1000), r.open, r.high, r.low, r.close, r.volume]
                for r in rows.itertuples()]

    def fetch_ticker(self, symbol):
        return {"last": float(self.df["close"].iloc[self.n - 1])}


def make_bot(tmp_path, decide):
    config = Config()
    config.state_file = str(tmp_path / "state.json")
    config.trade_log = str(tmp_path / "trades.csv")
    strat = types.SimpleNamespace(PARAMS={"stop_loss_pct": 0.5, "take_profit_pct": 10.0},
                                  indicators=lambda c, p: c, decide=decide)
    ex = FakeExchange(make_synthetic_ohlcv(n=100))
    return Bot(config, exchange=ex, broker=PaperBroker(config), strat=strat), ex, config


def test_closed_bars_drops_unfinished_candle():
    df = make_synthetic_ohlcv(n=5)
    last_start = df["timestamp"].iloc[-1].timestamp()
    assert len(closed_bars(df, 3600, now=last_start + 10)) == 4
    assert len(closed_bars(df, 3600, now=last_start + 3600)) == 5


def test_bot_buys_once_per_bar_and_persists(tmp_path):
    bot, ex, config = make_bot(tmp_path, lambda bar, prev, pos, p: "hold" if pos else "buy")
    far_future = 1e12
    bot.step(now=far_future)
    assert bot.position is not None
    qty = bot.position["qty"]
    bot.step(now=far_future)  # same bar again: no second buy
    assert bot.position["qty"] == qty

    # restart: position and paper cash survive
    bot2, _, _ = make_bot(tmp_path, lambda bar, prev, pos, p: "hold")
    assert bot2.position == bot.position
    assert bot2.broker.cash == bot.broker.cash
    lines = open(config.trade_log).read().splitlines()
    assert len(lines) == 2 and ",buy," in lines[1]


def test_bot_sells_on_signal(tmp_path):
    bot, ex, config = make_bot(tmp_path, lambda bar, prev, pos, p: "sell" if pos else "buy")
    bot.step(now=1e12)
    ex.n += 1
    bot.step(now=1e12)
    assert bot.position is None
    assert json.load(open(config.state_file))["position"] is None


class FakeAlpaca:
    """Market orders fill instantly; the buy fee is taken out of the coin received."""

    def __init__(self, fee=0.0025):
        self.usd, self.btc, self.fee = 10_000.0, 0.0, fee

    def fetch_balance(self):
        return {"free": {"USD": self.usd, "BTC": self.btc}}

    def create_market_buy_order(self, symbol, qty):
        self.usd -= qty * 100.0
        self.btc += qty * (1 - self.fee)
        return {"id": "1", "status": "closed", "average": 100.0, "filled": qty}

    def create_market_sell_order(self, symbol, qty):
        assert qty <= self.btc + 1e-12, "tried to sell more than held"
        self.btc -= qty
        self.usd += qty * 100.0 * (1 - self.fee)
        return {"id": "2", "status": "closed", "average": 100.0, "filled": qty}


def test_exchange_broker_sells_only_what_is_held():
    from tradingbot.brokers import ExchangeBroker
    broker = ExchangeBroker.__new__(ExchangeBroker)
    broker.config, broker.exchange = Config(), FakeAlpaca()
    broker.config.symbol = "BTC/USD"
    broker.base, broker.quote = "BTC", "USD"
    held, fill = broker.buy(10.0, 100.0)
    assert held == 10.0 * (1 - 0.0025)
    sold, _ = broker.sell(10.0, 100.0)  # asks for the ordered qty; must sell only what's held
    assert sold == held
    assert broker.exchange.btc == 0


def test_alpaca_data_works_without_keys():
    from tradingbot.data import fetch_history
    ex = ccxt.alpaca()
    calls = []

    def fake_bars(request):
        calls.append(dict(request))
        start = ex.parse8601(request["start"])
        n = 3 if "page_token" not in request else 2
        offset = 0 if "page_token" not in request else 3
        bars = [{"t": ex.iso8601(start + (offset + i) * 3600_000), "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 10}
                for i in range(n)]
        return {"bars": {"BTC/USD": bars}, "next_page_token": None if offset else "tok"}

    ex.marketPublicGetV1beta3CryptoLocBars = fake_bars
    ex.load_markets = lambda *a, **k: (_ for _ in ()).throw(AssertionError("needs keys"))
    df = fetch_history(ex, "BTC/USD", "1h", bars=5)
    assert len(df) == 5
    assert calls[0]["timeframe"] == "1H" and calls[1]["page_token"] == "tok"
