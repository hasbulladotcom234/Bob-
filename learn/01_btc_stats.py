# Lesson 1: Bitcoin's performance, computed by hand.
# Run it from the project folder:   python learn/01_btc_stats.py
# Then compare with BTC/USD in:     python main.py data benchmarks

import pandas as pd      # tables of data (like Excel in code)
import numpy as np       # maths (square roots etc.)

# 1. Load the prices that `data update` saved. One row per hour.
df = pd.read_csv("data/alpaca/1h/BTC-USD.csv", parse_dates=["timestamp"])
print(df.head())         # show the first 5 rows so you can see what's in there

# 2. Keep just the closing price, labelled by time.
close = df.set_index("timestamp")["close"]

# 3. Hourly is very noisy, so use one price per day: the last price of each day.
daily = close.resample("1D").last().dropna()

# 4. Total return: (last price / first price) - 1
total_return = daily.iloc[-1] / daily.iloc[0] - 1

# 5. Daily returns: how much the price moved each day, e.g. 0.02 = +2%
returns = daily.pct_change().dropna()

# 6. Volatility: how much daily returns jump around (standard deviation),
#    scaled up to a year. Crypto trades every day, so a year = 365 days.
#    Why sqrt(365)? Variance adds up over independent days, so std grows with sqrt(time).
annual_vol = returns.std() * np.sqrt(365)

# 7. Sharpe ratio: average return per unit of risk, on a yearly scale.
sharpe = returns.mean() / returns.std() * np.sqrt(365)

# 8. Max drawdown: the worst fall from a previous peak.
#    cummax() = "highest price seen so far" on each day.
peak = daily.cummax()
drawdown = daily / peak - 1          # 0 at a new high, -0.5 when 50% below the peak
max_drawdown = drawdown.min()

print()
print(f"From {daily.index[0]:%Y-%m-%d} to {daily.index[-1]:%Y-%m-%d}")
print(f"Total return:   {total_return * 100:8.1f}%")
print(f"Annual vol:     {annual_vol * 100:8.1f}%")
print(f"Sharpe:         {sharpe:8.2f}")
print(f"Max drawdown:   {max_drawdown * 100:8.1f}%  (on {drawdown.idxmin():%Y-%m-%d})")
