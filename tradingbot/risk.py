"""Position sizing and account-level risk controls.

These rules exist to survive being wrong, since no strategy is right every
time. They cap what a single bad trade or a bad day can cost.
"""
from dataclasses import dataclass, field
from datetime import date

from .config import RiskParams


@dataclass
class RiskManager:
    params: RiskParams
    _day: date = field(default=None, repr=False)
    _day_start_equity: float = field(default=None, repr=False)
    halted_today: bool = False

    def position_size(self, equity: float, entry_price: float, stop_price: float) -> float:
        """Return quantity to buy, sized so that hitting the stop loses at most
        risk_per_trade_pct of equity, capped by max_position_pct of equity."""
        stop_distance = abs(entry_price - stop_price)
        if stop_distance <= 0:
            return 0.0

        risk_amount = equity * self.params.risk_per_trade_pct
        qty_by_risk = risk_amount / stop_distance

        max_notional = equity * self.params.max_position_pct
        qty_by_cap = max_notional / entry_price

        return max(0.0, min(qty_by_risk, qty_by_cap))

    def stop_loss_price(self, entry_price: float, stop_loss_pct: float) -> float:
        return entry_price * (1 - stop_loss_pct)

    def take_profit_price(self, entry_price: float, take_profit_pct: float) -> float:
        return entry_price * (1 + take_profit_pct)

    def check_daily_circuit_breaker(self, today: date, current_equity: float) -> bool:
        """Call once per bar/tick. Returns True if trading should be halted
        for the rest of the day because losses exceeded max_daily_loss_pct."""
        if self._day != today:
            self._day = today
            self._day_start_equity = current_equity
            self.halted_today = False

        if self._day_start_equity:
            drawdown = (self._day_start_equity - current_equity) / self._day_start_equity
            if drawdown >= self.params.max_daily_loss_pct:
                self.halted_today = True

        return self.halted_today
