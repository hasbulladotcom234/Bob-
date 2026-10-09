"""Which assets to research and trade."""
import os

# Alpaca's crypto/USD pairs. If one isn't listed on Alpaca, it simply
# downloads no data and is reported as such.
DEFAULT_CRYPTO = [
    "BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "DOGE/USD", "LTC/USD", "BCH/USD",
    "LINK/USD", "AVAX/USD", "DOT/USD", "UNI/USD", "AAVE/USD", "SHIB/USD", "PEPE/USD",
    "CRV/USD", "GRT/USD", "MKR/USD", "SUSHI/USD", "BAT/USD", "XTZ/USD", "YFI/USD",
]

# Pegged to a dollar or to gold: nothing to research.
EXCLUDED_BASES = {"USDC", "USDT", "DAI", "USDG", "PAXG"}


def default_universe() -> list:
    """UNIVERSE env var (comma-separated) if set, else DEFAULT_CRYPTO."""
    env = os.getenv("UNIVERSE", "").strip()
    return [s.strip() for s in env.split(",") if s.strip()] if env else list(DEFAULT_CRYPTO)

