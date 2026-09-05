from datetime import date

from tradingbot.config import RiskParams
from tradingbot.risk import RiskManager


def test_position_size_respects_risk_per_trade():
    risk = RiskManager(params=RiskParams(risk_per_trade_pct=0.01, max_position_pct=1.0))
    equity = 10_000
    entry = 100.0
    stop = 95.0  # 5% stop distance
    qty = risk.position_size(equity, entry, stop)
    max_loss = qty * (entry - stop)
    assert max_loss <= equity * 0.01 + 1e-6


def test_position_size_respects_max_position_cap():
    risk = RiskManager(params=RiskParams(risk_per_trade_pct=1.0, max_position_pct=0.1))
    equity = 10_000
    entry = 100.0
    stop = 99.0  # tiny stop distance would otherwise imply a huge position
    qty = risk.position_size(equity, entry, stop)
    notional = qty * entry
    assert notional <= equity * 0.1 + 1e-6


def test_zero_stop_distance_gives_zero_size():
    risk = RiskManager(params=RiskParams())
    assert risk.position_size(10_000, 100.0, 100.0) == 0.0


def test_daily_circuit_breaker_trips_on_large_drawdown():
    risk = RiskManager(params=RiskParams(max_daily_loss_pct=0.05))
    today = date(2024, 1, 1)
    assert risk.check_daily_circuit_breaker(today, 10_000) is False
    assert risk.check_daily_circuit_breaker(today, 9_400) is True  # -6% > 5% limit


def test_circuit_breaker_resets_on_new_day():
    risk = RiskManager(params=RiskParams(max_daily_loss_pct=0.05))
    day1 = date(2024, 1, 1)
    day2 = date(2024, 1, 2)
    risk.check_daily_circuit_breaker(day1, 10_000)
    assert risk.check_daily_circuit_breaker(day1, 9_000) is True
    assert risk.check_daily_circuit_breaker(day2, 9_000) is False
