# Roadmap

The goal is a trading system built the way a professional quant team would
build one: rigorous research, honest testing, real risk management and
solid engineering. Jane Street's actual edge (market making, low latency,
huge data and capital) can't be copied from a laptop, but their *process*
can, and the process is what this project is meant to show.

**Split of work.** Claude builds the infrastructure (data, research tools,
backtester, execution, servers, monitoring). You do the research: form
hypotheses, design signals, judge results statistically, and keep
`RESEARCH_LOG.md`.

Each phase takes a few days. Finish a phase before starting the next one.

---

## Phase 0: Foundation (done)
- Alpaca paper trading, one coin, one strategy file (`my_strategy.py`)
- Backtester with next-bar fills, fees, lookahead check
- Setup wizard, status check, trade journal, restart-safe state

## Phase 1: Data
Nothing downstream is better than its data.
- Download years of history once and store it locally (Parquet), with
  incremental updates instead of re-downloading every run
- Every coin Alpaca offers, not just BTC
- Data quality checks: missing bars, duplicate bars, bad prints, stale prices
- Benchmark series (e.g. BTC buy & hold, equal-weight basket)

## Phase 2: Research framework
Where most of a quant's time goes.
- Compute any signal for every asset at once (vectorised)
- Predictive-power report for a signal: information coefficient (rank
  correlation with future returns), how fast it decays across horizons,
  turnover, and whether it holds across assets and over time
- Research notebooks wired to the local data
- You: first real signals, each with a hypothesis written down first

## Phase 3: Honest backtesting
Make it hard to fool yourself.
- Walk-forward testing: re-fit on a rolling window, trade the next unseen
  window, repeat
- A locked holdout period that is only used once, for the final verdict
- Multiple-testing correction (deflated Sharpe ratio) based on how many
  ideas and parameter sets were tried
- Bootstrap confidence intervals on Sharpe and drawdown
- Realistic costs: fees, bid-ask spread, slippage that grows with order size

## Phase 4: Portfolio and risk
Move from "one coin, in or out" to a portfolio.
- Hold many assets at once, weighted by signal strength
- Combine several signals into one forecast
- Volatility targeting: size positions so portfolio risk stays roughly constant
- Position, concentration and correlation limits; drawdown controls
- Rebalancing that accounts for trading costs

## Phase 5: Execution
- Limit orders where sensible, not only market orders
- Measure implementation shortfall: the price when you decided vs the price you got
- Compare paper fills with what the backtest assumed, and feed that back
  into the cost model

## Phase 6: Production
- Run 24/7 on a small cloud server
- Dashboard: equity, positions, signals, live vs backtest
- Alerts (phone/Discord) on errors, disconnects and drawdowns
- Daily PnL and risk report
- Tests run automatically on every push

## Phase 7: Advanced research (only after 1-6)
- Machine learning on features, with purged cross-validation (the version
  that doesn't leak future data)
- Regime detection (trending vs choppy, calm vs volatile)
- More data: funding rates, order book, cross-exchange prices
- US stocks and ETFs on Alpaca: thousands of assets and shorting, which
  allows proper cross-sectional strategies

---

## Ground rules
- Every idea gets a written hypothesis before any backtest.
- The holdout period stays untouched until a strategy is final.
- A strategy has to beat buy & hold and the equal-weight basket after
  costs, out of sample.
- Most ideas will fail. That's normal; record them in the research log anyway.
