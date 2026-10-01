"""YOUR STRATEGY. This is the only file you need to edit.

The plumbing calls two functions:

1. indicators(candles, p)
   Gets a pandas DataFrame of candles (columns: timestamp, open, high, low,
   close, volume), one row per bar, oldest first. Add any columns you want
   (moving averages, volatility, z-scores...) and return it.

2. decide(bar, prev, in_position, p)
   Called once per finished bar. `bar` is the newest row, `prev` the one
   before it, so you can write `bar.close`, `bar.ema_fast`, `prev.rsi`, etc.
   (Use the dot form; column names must be valid Python names.)
   `in_position` is True if you currently hold the coin.
   Return "buy", "sell" or "hold".

`p` is the PARAMS dict below. Put every number you might want to tune in
there instead of hard-coding it, so `python main.py optimize` can test
other values from PARAM_GRID.

How your decisions are executed (same in backtest and live):
- You decide when a bar closes; the order fills at the next bar's open.
  You can never trade on a price you only learn afterwards.
- Long only: "buy" opens a position, "sell" closes it. "buy" while already
  holding (or "sell" while flat) is ignored.
- Every position also gets a safety stop-loss and take-profit from
  p["stop_loss_pct"] and p["take_profit_pct"], checked continuously.
- Position size comes from the risk rules in tradingbot/config.py
  (RiskParams), not from here.

The example below is a simple trend follower: buy when the fast moving
average crosses above the slow one and RSI is above a threshold, sell on
the opposite cross. Replace it with your own ideas.
"""
from tradingbot.indicators import ema, rsi

PARAMS = {
    "fast_ema": 12,
    "slow_ema": 26,
    "rsi_period": 14,
    "rsi_min": 50,
    "stop_loss_pct": 0.03,    # exit if price falls 3% below entry
    "take_profit_pct": 0.06,  # exit if price rises 6% above entry
}

# Values `python main.py optimize` will try. Every combination is tested,
# so keep this small: more combinations = more chance of a lucky fluke.
PARAM_GRID = {
    "fast_ema": [8, 12, 20],
    "slow_ema": [26, 50, 100],
    "stop_loss_pct": [0.02, 0.03, 0.05],
    "take_profit_pct": [0.04, 0.06, 0.10],
}


def indicators(candles, p):
    candles["ema_fast"] = ema(candles["close"], p["fast_ema"])
    candles["ema_slow"] = ema(candles["close"], p["slow_ema"])
    candles["rsi"] = rsi(candles["close"], p["rsi_period"])
    return candles


def decide(bar, prev, in_position, p):
    crossed_up = prev.ema_fast <= prev.ema_slow and bar.ema_fast > bar.ema_slow
    crossed_down = prev.ema_fast >= prev.ema_slow and bar.ema_fast < bar.ema_slow

    if not in_position and crossed_up and bar.rsi > p["rsi_min"]:
        return "buy"
    if in_position and crossed_down:
        return "sell"
    return "hold"
