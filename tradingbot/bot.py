"""Live loop: every poll it checks the stop-loss / take-profit against the
current price, and once per newly closed bar it asks your strategy what to do.

State (open position, last bar seen) is saved to config.state_file so a
restart picks up where it left off. Every fill is appended to
config.trade_log (CSV) so you can analyse what the bot actually did.
"""
import csv
import json
import logging
import os
import time
from datetime import datetime, timezone


from .brokers import ExchangeBroker, make_broker
from .config import Config
from .data import closed_bars, exchange_from_config, fetch_ohlcv
from .risk import RiskManager
from .strategy import Bars, decide, load_strategy, with_indicators

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("tradingbot")

LIVE_BARS = 500  # candles fetched each poll; enough for indicators to warm up


class Bot:
    def __init__(self, config: Config, exchange=None, broker=None, strat=None):
        self.config = config
        self.exchange = exchange or exchange_from_config(config)
        self.broker = broker or make_broker(config)
        self.strat = strat or load_strategy(config.strategy_module)
        self.params = dict(self.strat.PARAMS)
        self.risk = RiskManager(params=config.risk)
        self.tf_seconds = self.exchange.parse_timeframe(config.timeframe)
        self.position = None   # dict(qty, entry, stop, take, entry_time) while holding
        self.last_bar = None   # timestamp (ISO string) of the last bar we decided on
        self._load_state()

    # ---- persistence -------------------------------------------------------
    def _load_state(self):
        if not os.path.exists(self.config.state_file):
            return
        with open(self.config.state_file) as f:
            state = json.load(f)
        self.position = state.get("position")
        self.last_bar = state.get("last_bar")
        self.broker.load_state(state.get("broker", {}))
        log.info("Loaded state from %s: position=%s", self.config.state_file, self.position)

    def _save_state(self):
        state = {"position": self.position, "last_bar": self.last_bar, "broker": self.broker.save_state()}
        tmp = self.config.state_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, self.config.state_file)

    def _journal(self, action, qty, price, reason, equity):
        new = not os.path.exists(self.config.trade_log)
        with open(self.config.trade_log, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["time", "mode", "symbol", "action", "qty", "price", "reason", "equity"])
            w.writerow([datetime.now(timezone.utc).isoformat(timespec="seconds"), self.config.mode,
                        self.config.symbol, action, f"{qty:.8f}", f"{price:.2f}", reason, f"{equity:.2f}"])

    def reconcile(self):
        """On a real account, make sure our saved position matches reality."""
        if not isinstance(self.broker, ExchangeBroker):
            return
        held = self.broker.coin_balance()
        if self.position and held <= 0:
            log.warning("State says we hold %s but the account has none; clearing position.",
                        self.config.symbol)
            self.position = None
        elif self.position and held < self.position["qty"]:
            self.position["qty"] = held
        elif not self.position and held > 0:
            log.warning("Account holds %.8f %s that this bot didn't open; leaving it alone.",
                        held, self.broker.base)
        self._save_state()

    # ---- trading -----------------------------------------------------------
    def _exit(self, price, reason, equity):
        qty, fill = self.broker.sell(self.position["qty"], price)
        log.info("SELL %.8f at %.2f (%s)", qty, fill, reason)
        self._journal("sell", qty, fill, reason, equity)
        self.position = None

    def _enter(self, price, equity):
        stop = self.risk.stop_loss_price(price, self.params["stop_loss_pct"])
        qty = self.risk.position_size(equity, price, stop)
        if qty <= 0:
            return
        held, fill = self.broker.buy(qty, price)
        if held <= 0:
            log.warning("Buy order for %.8f returned no fill", qty)
            return
        self.position = {
            "qty": held, "entry": fill,
            "stop": self.risk.stop_loss_price(fill, self.params["stop_loss_pct"]),
            "take": self.risk.take_profit_price(fill, self.params["take_profit_pct"]),
            "entry_time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        log.info("BUY %.8f at %.2f stop=%.2f take=%.2f", held, fill, self.position["stop"], self.position["take"])
        self._journal("buy", held, fill, "signal", equity)

    def step(self, now: float = None, dry_run: bool = False):
        """One pass of the loop. dry_run=True only reports what it would do:
        no orders, no state changes."""
        candles = closed_bars(fetch_ohlcv(self.exchange, self.config.symbol, self.config.timeframe,
                                          limit=LIVE_BARS), self.tf_seconds, now)
        if candles.empty:
            return
        try:
            price = float(self.exchange.fetch_ticker(self.config.symbol)["last"])
        except Exception:
            price = float(candles["close"].iloc[-1])

        equity = self.broker.get_equity(price)
        today = datetime.now(timezone.utc).date()
        halted = self.risk.check_daily_circuit_breaker(today, equity)

        if dry_run:
            data = with_indicators(candles, self.strat, self.params)
            action = decide(Bars(data), len(data) - 1, self.position is not None, self.strat, self.params)
            log.info("Price %.2f | Equity %.2f | %s | latest bar %s -> strategy says %s",
                     price, equity, f"holding {self.position['qty']:.8f}" if self.position else "flat",
                     candles["timestamp"].iloc[-1], action)
            return action

        # protective exits, checked every poll
        if self.position:
            if price <= self.position["stop"]:
                self._exit(price, "stop_loss", equity)
            elif price >= self.position["take"]:
                self._exit(price, "take_profit", equity)

        # strategy decisions, once per newly closed bar
        bar_time = candles["timestamp"].iloc[-1].isoformat()
        if bar_time != self.last_bar:
            data = with_indicators(candles, self.strat, self.params)
            action = decide(Bars(data), len(data) - 1, self.position is not None, self.strat, self.params)
            log.info("Bar %s closed at %.2f -> %s", bar_time, data["close"].iloc[-1], action)
            self.last_bar = bar_time
            if action == "sell" and self.position:
                self._exit(price, "signal", equity)
            elif action == "buy" and not self.position:
                if halted:
                    log.warning("Buy signal ignored: daily loss limit hit")
                else:
                    self._enter(price, equity)

        self._save_state()
        log.info("Price %.2f | Equity %.2f | %s", price, self.broker.get_equity(price),
                 f"holding {self.position['qty']:.8f}" if self.position else "flat")


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

    bot = Bot(config)
    bot.reconcile()
    while True:
        try:
            bot.step()
        except Exception:
            log.exception("Error in bot loop, will retry after poll interval")
        time.sleep(config.poll_interval_seconds)


if __name__ == "__main__":
    run_forever(Config())
