# tradingbot

A crypto trend-following/momentum trading bot: backtester, walk-forward
parameter optimizer, stochastic-calculus trade filter, Monte Carlo stress
testing on calibrated jump-diffusion / stochastic-volatility models,
paper-trading mode, and a live-trading mode gated behind explicit safety
checks. Trades on Kraken by default (any ccxt exchange works).

## Read this first

There is no bot that reliably buys the exact bottom and sells the exact
top. Nobody has that. What this does instead: trade a defined, testable
rule (EMA crossover + RSI momentum filter), size positions by risk rather
than gut feel, cut losers with a stop-loss, and halt for the day if losses
get too deep. That is what a real trading system looks like. Past backtest
performance does not guarantee future results. Only trade money you can
afford to lose. This is not financial advice.

The stochastic calculus doesn't change that. It can't see the future. It
gives two things: a way to measure whether a trade's odds cover its costs,
and a way to simulate thousands of realistic alternative markets. The
simulations show how often the strategy loses, instead of relying on one
lucky or unlucky backtest.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then edit as needed
```

## Configuration (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `EXCHANGE_ID` | `kraken` | Any [ccxt](https://github.com/ccxt/ccxt) exchange id (`kraken`, `coinbase`, `binance`, ...) |
| `SYMBOL` | `BTC/USD` | Trading pair |
| `TIMEFRAME` | `1h` | Candle timeframe |
| `MODE` | `paper` | `paper` or `live` |
| `STARTING_BALANCE` | `10000` | Simulated starting cash (paper mode only) |
| `POLL_INTERVAL_SECONDS` | `60` | How often the live loop checks for new signals |
| `TAKER_FEE_PCT` | per exchange | Taker fee as a fraction. Defaults: Kraken 0.004, Coinbase 0.012, Binance 0.001 (entry tiers, check yours) |
| `EXCHANGE_API_KEY` / `EXCHANGE_API_SECRET` | _(empty)_ | Only needed for live mode |
| `I_UNDERSTAND_LIVE_TRADING_RISK` | _(empty)_ | Must be set to `yes` to unlock live mode |

Kraken only serves the most recent 720 candles per timeframe, so
`backtest` / `optimize` get about 30 days of 1h data there. For longer
history, use `TIMEFRAME=4h`, or backtest on an exchange with deeper
history (`EXCHANGE_ID=coinbase`, which pages back further). The Monte
Carlo command can also simulate longer histories than the data you have.

## Workflow

1. **Backtest** the default strategy on recent history:
   ```bash
   python main.py backtest
   ```
2. **Optimize** ("train") parameters with a train/test split so results
   aren't just curve-fit to one dataset:
   ```bash
   python main.py optimize
   ```
   This grid-searches EMA/RSI/stop-loss/take-profit combinations, scores
   them by risk-adjusted return on the training slice, and reports the
   same score on an untouched test slice. If the test score is much worse
   than train, the parameters are overfit — don't trust them.
3. **Calibrate** the stochastic market models to recent history and see
   what the filter thinks of the market right now:
   ```bash
   python main.py calibrate
   ```
4. **Monte Carlo stress test**: simulate hundreds of alternative price
   histories from the calibrated model and backtest on every one:
   ```bash
   python main.py montecarlo                    # Bates model, 200 paths
   python main.py montecarlo --drift zero       # strip the trend: is there real edge?
   python main.py montecarlo --model merton --paths 500 --seed 1
   ```
   Reports mean/median return, chance of losing money, 5% VaR and CVaR,
   drawdowns, and where the real-history backtest ranks among the
   simulations. If the real backtest beats 95% of simulated paths, it was
   probably luck, not skill. If `--drift zero` loses money, the strategy
   is only riding the trend of the calibration window.
5. Take the best params from step 2 and set them in `tradingbot/config.py`
   (`StrategyParams` defaults), or wire them into `.env` if you prefer.
6. **Paper trade** to see it run against live market data with fake money:
   ```bash
   MODE=paper python main.py run
   ```
7. Only after you're satisfied with paper results, **go live**:
   ```bash
   EXCHANGE_API_KEY=... EXCHANGE_API_SECRET=... \
   I_UNDERSTAND_LIVE_TRADING_RISK=yes MODE=live python main.py run
   ```
   The bot refuses to place real orders unless both the confirmation
   variable and valid API keys are present.

## The stochastic calculus

### Entry filter (`stochastic.py`)

Over a short window, log price is modelled as Brownian motion with drift,
`d(log S) = nu dt + sigma dW`. That is the Ito form of geometric Brownian
motion. The drift `nu` and volatility `sigma` are estimated per bar with an
EWMA (`stoch_lookback`). Drift estimates are extremely noisy, so they are
shrunk toward zero (`drift_shrinkage`).

A stop-loss / take-profit bracket is a two-barrier first-passage problem.
Solving `nu f' + (sigma^2 / 2) f'' = 0` gives the closed-form probability
that the take-profit is hit before the stop:

```
P(take) = (1 - e^(k b)) / (e^(-k a) - e^(k b)),   k = 2 nu / sigma^2
a = ln(1 + take_profit),  b = -ln(1 - stop_loss)
```

From that, the expected return of the trade after fees and slippage is
`P * tp - (1 - P) * sl - costs`. Each EMA crossover is only taken if that
edge exceeds `min_edge_pct`. The expected time to exit is computed too.
The bot logs `P(take-profit first)` and `edge` on every loop. Disable the
filter with `use_stochastic_filter=False` in `StrategyParams`.

### Market simulator (`simulation.py`, `montecarlo.py`)

Real crypto prices have fat tails, sudden jumps, and volatility that comes
in clusters. Plain random-walk simulations miss all three. The simulator
implements the **Bates model**, which combines Heston stochastic
volatility with Merton jumps:

```
d(log S) = (m - v/2) dt + sqrt(v) dW1 + J dN        N ~ Poisson(lambda), J ~ Normal
dv       = kappa (theta - v) dt + xi sqrt(v) dW2     corr(dW1, dW2) = rho
```

The simpler special cases are also available: `gbm`, `merton` and
`heston`. Parameters are calibrated from price history:

- jumps are returns more than 4 local robust standard deviations from the
  median;
- mean reversion of variance comes from an AR(1) fit on block realized
  variance, the exact discretization of the CIR variance process;
- vol-of-vol is corrected for realized-variance sampling noise;
- `rho` measures the leverage effect.

Each bar is simulated in 16 Euler sub-steps, so highs and lows (and
therefore stop-loss hits) come from a real intrabar path. Calibration is
moment-based, so it gets the shape right but not every decimal place.

## Risk controls (always on, paper or live)

- **Position sizing**: each trade risks at most `risk_per_trade_pct` of
  equity (default 1%), capped at `max_position_pct` of equity in any one
  trade (default 25%).
- **Stop-loss / take-profit**: every position has both, checked every bar.
- **Daily circuit breaker**: trading halts for the rest of the day if
  losses exceed `max_daily_loss_pct` of the day's starting equity
  (default 5%).

Tune these in `tradingbot/config.py` (`RiskParams`).

## Project layout

```
tradingbot/
  config.py      # settings, env vars, live-trading safety gate
  indicators.py  # EMA, RSI
  strategy.py    # signal generation (EMA/RSI + stochastic edge filter)
  stochastic.py  # drift/vol estimation, first-passage probabilities, trade edge
  simulation.py  # GBM / Merton / Heston / Bates models: calibration and path simulation
  montecarlo.py  # backtest across simulated histories, VaR/CVaR summary
  risk.py        # position sizing, stop/take levels, daily circuit breaker
  backtester.py  # event-driven backtest over historical OHLCV
  optimizer.py   # walk-forward grid search over strategy params
  brokers.py     # PaperBroker (simulated) and LiveBroker (real ccxt orders)
  data.py        # OHLCV fetching via ccxt, paging past per-request limits
  bot.py         # main run loop
main.py          # CLI: backtest / optimize / calibrate / montecarlo / run
tests/           # pytest unit tests (synthetic data, no network calls)
```

## Testing

```bash
python -m pytest
```

Tests use synthetic price data and never hit the network.
