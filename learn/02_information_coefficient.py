# Lesson 2: the information coefficient (IC), by hand.
# Run it from the project folder:   python learn/02_information_coefficient.py

import numpy as np
import pandas as pd

# ---- Part 1: one day, five coins -------------------------------------------
# Your signal gave each coin a score (here: last week's return).
# Then we look at what each coin actually did over the NEXT week.
day = pd.DataFrame({
    "score":     [0.30, 0.10, -0.05, -0.20, 0.02],
    "next_week": [0.08, 0.01, -0.03, 0.02, -0.01],
}, index=["SOL", "ETH", "BTC", "DOGE", "XRP"])

# Rank both columns: 1 = lowest, 5 = highest.
# Ranks, not raw numbers, so one crazy coin can't dominate (same idea as the median in cleaning).
day["score_rank"] = day["score"].rank()
day["next_rank"] = day["next_week"].rank()
print(day)

# IC = correlation between the two rankings. +1 = perfect order, 0 = no link, -1 = backwards.
ic = day["score_rank"].corr(day["next_rank"])
print(f"\nIC for this one day: {ic:+.2f}")
# Check it yourself: DOGE had the lowest score but did 2nd best -> that pulls the IC down.

# ---- Part 2: every day, all coins (real data) -------------------------------
from tradingbot.config import Config
from tradingbot.research import load_research_data
from tradingbot.store import BarStore

config = Config()
store = BarStore()
d = load_research_data(store, store.symbols("1h"), "1h", end_before=config.holdout_start)
print(f"\nLoaded {d.close.shape[1]} coins x {d.close.shape[0]} days")

score = d.close / d.close.shift(7) - 1            # momentum: last week's return, every coin, every day
next_week = d.close.shift(-7) / d.close - 1       # what happened over the following week

ics = []
for date in d.close.index:
    both = pd.DataFrame({"s": score.loc[date], "n": next_week.loc[date]}).dropna()
    if len(both) >= 5:                            # need a few coins to rank
        ics.append(both["s"].rank().corr(both["n"].rank()))
ics = pd.Series(ics)

# Is the average IC far enough from zero to not be luck?
# Next-week returns of neighbouring days overlap (they share 6 days), so only
# about len/7 of the days are independent pieces of evidence.
t_stat = ics.mean() / ics.std() * np.sqrt(len(ics) / 7)
print(f"Days measured: {len(ics)}   mean IC: {ics.mean():+.3f}   t-stat: {t_stat:+.2f}")
print("Compare with the 7-day row of:  python main.py research momentum_7d")
