"""Broker implementations: a simulated paper broker and a real ccxt-backed
live broker, behind a common interface so the bot loop doesn't care which
one it's talking to.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import ccxt

from .config import Config


@dataclass
class Position:
    qty: float = 0.0
    entry_price: float = 0.0
    stop_price: float = 0.0
    take_price: float = 0.0


class Broker(ABC):
    @abstractmethod
    def get_equity(self, last_price: float) -> float:
        ...

    @abstractmethod
    def get_position(self) -> Position:
        ...

    @abstractmethod
    def buy(self, qty: float, price: float, stop_price: float, take_price: float) -> None:
        ...

    @abstractmethod
    def sell_all(self, price: float) -> None:
        ...


class PaperBroker(Broker):
    """Simulates fills against the last traded price, applying fees and
    slippage, with an in-memory cash balance. No real money moves."""

    def __init__(self, config: Config):
        self.config = config
        self.cash = config.starting_balance
        self.position = Position()

    def get_equity(self, last_price: float) -> float:
        return self.cash + self.position.qty * last_price

    def get_position(self) -> Position:
        return self.position

    def buy(self, qty: float, price: float, stop_price: float, take_price: float) -> None:
        fill_price = price * (1 + self.config.risk.slippage_pct)
        cost = qty * fill_price * (1 + self.config.risk.taker_fee_pct)
        if cost > self.cash or qty <= 0:
            return
        self.cash -= cost
        self.position = Position(qty=qty, entry_price=fill_price, stop_price=stop_price, take_price=take_price)

    def sell_all(self, price: float) -> None:
        if self.position.qty <= 0:
            return
        fill_price = price * (1 - self.config.risk.slippage_pct)
        proceeds = self.position.qty * fill_price * (1 - self.config.risk.taker_fee_pct)
        self.cash += proceeds
        self.position = Position()


class LiveBroker(Broker):
    """Places real orders on a real exchange via ccxt. Only usable when
    Config.validate_live_trading_allowed() has passed."""

    def __init__(self, config: Config):
        self.config = config
        config.validate_live_trading_allowed()
        exchange_class = getattr(ccxt, config.exchange_id)
        self.exchange = exchange_class({
            "apiKey": config.api_key,
            "secret": config.api_secret,
            "enableRateLimit": True,
        })
        self.position = Position()

    def get_equity(self, last_price: float) -> float:
        balance = self.exchange.fetch_balance()
        quote = self.config.symbol.split("/")[1]
        base = self.config.symbol.split("/")[0]
        cash = balance.get("free", {}).get(quote, 0.0) or 0.0
        base_qty = balance.get("free", {}).get(base, 0.0) or 0.0
        return cash + base_qty * last_price

    def get_position(self) -> Position:
        return self.position

    def buy(self, qty: float, price: float, stop_price: float, take_price: float) -> None:
        if qty <= 0:
            return
        order = self.exchange.create_market_buy_order(self.config.symbol, qty)
        fill_price = order.get("average") or order.get("price") or price
        self.position = Position(qty=qty, entry_price=fill_price, stop_price=stop_price, take_price=take_price)

    def sell_all(self, price: float) -> None:
        if self.position.qty <= 0:
            return
        self.exchange.create_market_sell_order(self.config.symbol, self.position.qty)
        self.position = Position()


def make_broker(config: Config) -> Broker:
    if config.is_live():
        return LiveBroker(config)
    return PaperBroker(config)
