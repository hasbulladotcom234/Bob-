"""Bot configuration, loaded from environment variables (.env supported).

Nothing here is a secret by default. API keys are only read from the
environment and are never hardcoded or logged.
"""
import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class StrategyParams:
    fast_ema: int = 12
    slow_ema: int = 26
    rsi_period: int = 14
    rsi_bull_threshold: float = 50.0
    stop_loss_pct: float = 0.03      # exit if price falls 3% below entry
    take_profit_pct: float = 0.06    # exit if price rises 6% above entry

    # Stochastic entry filter (see stochastic.py): model log price as Brownian
    # motion with drift and skip entries whose stop/target bracket has negative
    # expected value after costs.
    use_stochastic_filter: bool = True
    stoch_lookback: int = 100        # EWMA span (bars) for drift/volatility estimates
    drift_shrinkage: float = 0.5     # scale drift estimate toward zero (it is very noisy)
    min_edge_pct: float = 0.0        # required expected return per trade, after costs


@dataclass
class RiskParams:
    risk_per_trade_pct: float = 0.01   # fraction of equity risked per trade
    max_position_pct: float = 0.25     # never put more than this fraction of equity in one trade
    max_daily_loss_pct: float = 0.05   # circuit breaker: halt trading for the day past this drawdown
    taker_fee_pct: Optional[float] = None  # exchange taker fee; None = TAKER_FEE_PCT env or exchange default
    slippage_pct: float = 0.0005       # assumed slippage on market fills


# Entry-level taker fees for market orders. Check your own fee tier; these
# change and drop with volume.
DEFAULT_TAKER_FEES = {
    "kraken": 0.004,
    "coinbase": 0.012,
    "binance": 0.001,
    "binanceus": 0.006,
}


@dataclass
class Config:
    exchange_id: str = os.getenv("EXCHANGE_ID", "kraken")
    symbol: str = os.getenv("SYMBOL", "BTC/USD")
    timeframe: str = os.getenv("TIMEFRAME", "1h")
    mode: str = os.getenv("MODE", "paper")  # "paper" or "live"
    starting_balance: float = float(os.getenv("STARTING_BALANCE", "10000"))
    poll_interval_seconds: int = int(os.getenv("POLL_INTERVAL_SECONDS", "60"))

    api_key: str = os.getenv("EXCHANGE_API_KEY", "")
    api_secret: str = os.getenv("EXCHANGE_API_SECRET", "")

    # Live trading is refused unless this is explicitly set to "yes" by a human,
    # in addition to mode=live and valid API keys being present.
    live_trading_confirmed: str = os.getenv("I_UNDERSTAND_LIVE_TRADING_RISK", "")

    strategy: StrategyParams = field(default_factory=StrategyParams)
    risk: RiskParams = field(default_factory=RiskParams)

    def __post_init__(self):
        if self.risk.taker_fee_pct is None:
            env_fee = os.getenv("TAKER_FEE_PCT")
            self.risk.taker_fee_pct = (
                float(env_fee) if env_fee else DEFAULT_TAKER_FEES.get(self.exchange_id, 0.001)
            )

    def round_trip_cost(self) -> float:
        """Fees plus slippage paid to get into and back out of a position."""
        return 2 * (self.risk.taker_fee_pct + self.risk.slippage_pct)

    def is_live(self) -> bool:
        return self.mode.strip().lower() == "live"

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
