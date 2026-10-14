"""YOUR SIGNALS: ideas to test with  python main.py research NAME

A signal gives every coin a score on every day. The research tool checks
whether coins with higher scores went on to do better than coins with lower
scores, across all coins and all days.

Each signal is a function that gets `d`, which holds one table per price:
    d.open, d.high, d.low, d.close, d.volume
Each table has one row per day and one column per coin. Maths on a table
happens for every coin at once:
    d.close.shift(7)             the close 7 days earlier, for every coin
    d.close.pct_change()         each day's return
    d.close.rolling(30).std()    a rolling 30-day calculation
Return a table of the same shape: each coin's score on each day.

Rules:
- Only use the past. A day's score may use prices up to that day's close,
  never later (no .shift(-1)). The tool checks and warns you.
- Write a one-line docstring: the idea, and WHY it should work.
- Add every idea you test to RESEARCH_LOG.md, including the ones that fail.
"""


def momentum_7d(d):
    """Coins that rose most over the past week keep rising (trends persist)."""
    return d.close / d.close.shift(7) - 1


def reversal_1d(d):
    """Yesterday's biggest losers bounce back today (people overreact, then it corrects)."""
    return -(d.close / d.close.shift(1) - 1)
