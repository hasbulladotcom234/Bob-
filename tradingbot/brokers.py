"""Brokers turn "buy this much" / "sell this much" into fills. The bot loop
tracks the position itself (and saves it to disk), so a broker only needs to
report what actually got filled.

- PaperBroker: simulated fills in memory, no account needed.
- ExchangeBroker: real orders via ccxt, either on the exchange's paper/testnet
  account (MODE=sandbox) or with real money (MODE=live).
"""
import logging
import time
from abc import ABC, abstractmethod

from .config import Config
from .data import exchange_from_config

log = logging.getLogger("tradingbot")


class Broker(ABC):
    @abstractmethod
    def get_equity(self, last_price: float) -> float:
        ...

    @abstractmethod
    def buy(self, qty: float, price: float) -> tuple:
        """Market buy. Returns (qty now held from this order, average fill price)."""

    @abstractmethod
    def sell(self, qty: float, price: float) -> tuple:
        """Market sell. Returns (qty sold, average fill price)."""

    def save_state(self) -> dict:
        return {}

    def load_state(self, state: dict) -> None:
        pass


class PaperBroker(Broker):
    """Simulates fills at the given price plus slippage and fees."""

    def __init__(self, config: Config):
        self.config = config
        self.cash = config.starting_balance
        self.coin = 0.0

    def get_equity(self, last_price: float) -> float:
        return self.cash + self.coin * last_price

    def buy(self, qty, price):
        r = self.config.risk
        fill = price * (1 + r.slippage_pct)
        qty = min(qty, self.cash / (fill * (1 + r.taker_fee_pct)))
        if qty <= 0:
            return 0.0, fill
        self.cash -= qty * fill * (1 + r.taker_fee_pct)
        self.coin += qty
        return qty, fill

    def sell(self, qty, price):
        r = self.config.risk
        fill = price * (1 - r.slippage_pct)
        qty = min(qty, self.coin)
        self.cash += qty * fill * (1 - r.taker_fee_pct)
        self.coin -= qty
        return qty, fill

    def save_state(self):
        return {"paper_cash": self.cash, "paper_coin": self.coin}

    def load_state(self, state):
        self.cash = state.get("paper_cash", self.cash)
        self.coin = state.get("paper_coin", self.coin)


class ExchangeBroker(Broker):
    """Real orders via ccxt. Sandbox mode trades the exchange's paper account
    (fake money); live mode requires Config.validate_live_trading_allowed()."""

    def __init__(self, config: Config):
        self.config = config
        if config.is_sandbox():
            config.validate_sandbox_allowed()
        else:
            config.validate_live_trading_allowed()
        self.exchange = exchange_from_config(config)
        self.base, self.quote = config.symbol.split("/")

    def _free(self, asset: str) -> float:
        return float(self.exchange.fetch_balance().get("free", {}).get(asset) or 0.0)

    def get_equity(self, last_price):
        free = self.exchange.fetch_balance().get("free", {})
        return float(free.get(self.quote) or 0.0) + float(free.get(self.base) or 0.0) * last_price

    def coin_balance(self) -> float:
        return self._free(self.base)

    def _wait_for_fill(self, order: dict, timeout: float = 10.0) -> dict:
        deadline = time.time() + timeout
        while order.get("status") not in ("closed", "canceled", "rejected", "expired") and time.time() < deadline:
            time.sleep(1)
            order = self.exchange.fetch_order(order["id"], self.config.symbol)
        return order

    def buy(self, qty, price):
        if qty <= 0:
            return 0.0, price
        before = self.coin_balance()
        order = self._wait_for_fill(self.exchange.create_market_buy_order(self.config.symbol, qty))
        fill = order.get("average") or order.get("price") or price
        # Some exchanges (Alpaca crypto) take the fee out of the coin you buy,
        # so what you hold is less than what you ordered. Use the real balance.
        held = self.coin_balance() - before
        if held <= 0:
            held = order.get("filled") or 0.0
        log.info("Order %s: bought %.8f %s at %.2f (status %s)",
                 order.get("id"), held, self.base, fill, order.get("status"))
        return held, fill

    def sell(self, qty, price):
        qty = min(qty, self.coin_balance())  # never try to sell more than we actually hold
        if qty <= 0:
            return 0.0, price
        order = self._wait_for_fill(self.exchange.create_market_sell_order(self.config.symbol, qty))
        fill = order.get("average") or order.get("price") or price
        log.info("Order %s: sold %.8f %s at %.2f (status %s)",
                 order.get("id"), qty, self.base, fill, order.get("status"))
        return qty, fill


def make_broker(config: Config) -> Broker:
    if config.is_live() or config.is_sandbox():
        return ExchangeBroker(config)
    return PaperBroker(config)
