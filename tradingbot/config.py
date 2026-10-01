"""Bot configuration, loaded from environment variables (.env supported).

Nothing here is a secret by default. API keys are only read from the
environment and are never hardcoded or logged.

Strategy settings (moving average lengths, stop-loss %, ...) live in
my_strategy.py. This file holds the plumbing and account-level risk rules.
"""
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass
class RiskParams:
    risk_per_trade_pct: float = 0.01   # fraction of equity lost if a trade hits its stop
    max_position_pct: float = 0.25     # never put more than this fraction of equity in one trade
    max_daily_loss_pct: float = 0.05   # circuit breaker: no new trades for the day past this drawdown
    taker_fee_pct: float = float(os.getenv("TAKER_FEE_PCT", "0.0025"))  # Alpaca crypto entry tier: 0.25%
    slippage_pct: float = 0.0005       # assumed slippage on market fills


@dataclass
class Config:
    exchange_id: str = os.getenv("EXCHANGE_ID", "binance")
    symbol: str = os.getenv("SYMBOL", "BTC/USDT")
    timeframe: str = os.getenv("TIMEFRAME", "1h")
    # "paper"   = local simulation, no account needed
    # "sandbox" = real orders on the exchange's paper/testnet account (e.g. Alpaca paper)
    # "live"    = real orders with real money
    mode: str = os.getenv("MODE", "paper")
    starting_balance: float = float(os.getenv("STARTING_BALANCE", "10000"))
    poll_interval_seconds: int = int(os.getenv("POLL_INTERVAL_SECONDS", "60"))
    strategy_module: str = os.getenv("STRATEGY", "my_strategy")
    history_bars: int = int(os.getenv("HISTORY_BARS", "5000"))  # bars used by backtest/optimize

    # The bot remembers its position here across restarts, and logs every trade.
    state_file: str = os.getenv("STATE_FILE", "state.json")
    trade_log: str = os.getenv("TRADE_LOG", "trades.csv")

    api_key: str = os.getenv("EXCHANGE_API_KEY", "")
    api_secret: str = os.getenv("EXCHANGE_API_SECRET", "")

    # Live trading is refused unless this is explicitly set to "yes" by a human,
    # in addition to mode=live and valid API keys being present.
    live_trading_confirmed: str = os.getenv("I_UNDERSTAND_LIVE_TRADING_RISK", "")

    risk: RiskParams = field(default_factory=RiskParams)

    def is_live(self) -> bool:
        return self.mode.strip().lower() == "live"

    def is_sandbox(self) -> bool:
        return self.mode.strip().lower() == "sandbox"

    def validate_sandbox_allowed(self) -> None:
        if self.is_sandbox() and (not self.api_key or not self.api_secret):
            raise RuntimeError(
                "MODE=sandbox needs EXCHANGE_API_KEY / EXCHANGE_API_SECRET set to your "
                "exchange's paper-account keys."
            )

    def validate_live_trading_allowed(self) -> None:
        if not self.is_live():
            return
        if self.live_trading_confirmed.strip().lower() != "yes":
            raise RuntimeError(
                "Refusing to start live trading: set I_UNDERSTAND_LIVE_TRADING_RISK=yes "
                "in your environment only after you have backtested and paper-traded the "
                "strategy and understand you can lose real money."
            )
        if not self.api_key or not self.api_secret:
            raise RuntimeError(
                "Refusing to start live trading: EXCHANGE_API_KEY / EXCHANGE_API_SECRET "
                "are not set."
            )
