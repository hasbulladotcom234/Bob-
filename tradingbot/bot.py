"""Main bot loop: fetch latest data, compute signals, apply risk rules,
place orders through whichever broker (paper or live) the config selects.
"""
import logging
import time
from datetime import datetime, timezone

from .brokers import make_broker
from .config import Config
from .data import exchange_from_config, fetch_ohlcv
from .risk import RiskManager
from .strategy import generate_signals

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("tradingbot")


def run_forever(config: Config) -> None:
    if config.is_live():
        config.validate_live_trading_allowed()
        log.warning("LIVE TRADING ENABLED. Real funds are at risk on %s %s.",
                    config.exchange_id, config.symbol)
    elif config.is_sandbox():
        config.validate_sandbox_allowed()
        log.info("Running in SANDBOX mode: orders go to the %s paper/testnet account (fake money).",
                 config.exchange_id)
    else:
        log.info("Running in PAPER mode (simulated funds, no real orders).")

    broker = make_broker(config)
    risk = RiskManager(params=config.risk)
    exchange = exchange_from_config(config)

    while True:
        try:
            df = fetch_ohlcv(exchange, config.symbol, config.timeframe, limit=200)
            data = generate_signals(df, config.strategy)
            last = data.iloc[-1]
            price = last["close"]
            today = datetime.now(timezone.utc).date()

            equity = broker.get_equity(price)
            halted = risk.check_daily_circuit_breaker(today, equity)
            position = broker.get_position()

            if halted:
                log.warning("Daily loss limit hit, trading halted for today. Equity=%.2f", equity)
            elif position.qty > 0:
                hit_stop = last["low"] <= position.stop_price
                hit_take = last["high"] >= position.take_price
                if hit_stop or hit_take or last["long_exit"]:
                    reason = "stop_loss" if hit_stop else ("take_profit" if hit_take else "signal_exit")
                    log.info("Exiting position (%s) at %.2f", reason, price)
                    broker.sell_all(price)
            elif last["long_entry"]:
                stop_price = risk.stop_loss_price(price, config.strategy.stop_loss_pct)
                take_price = risk.take_profit_price(price, config.strategy.take_profit_pct)
                qty = risk.position_size(equity, price, stop_price)
                if qty > 0:
                    log.info("Entering long: qty=%.6f price=%.2f stop=%.2f take=%.2f",
                              qty, price, stop_price, take_price)
                    broker.buy(qty, price, stop_price, take_price)

            log.info("Equity: %.2f | Position qty: %.6f", equity, broker.get_position().qty)

        except Exception:
            log.exception("Error in bot loop, will retry after poll interval")

        time.sleep(config.poll_interval_seconds)


if __name__ == "__main__":
    run_forever(Config())
