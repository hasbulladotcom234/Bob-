# tradingbot

A crypto trend-following/momentum trading bot: backtester, walk-forward
parameter optimizer, paper-trading mode, and a live-trading mode gated
behind explicit safety checks.

## Read this first

There is no bot that reliably buys the exact bottom and sells the exact
top. Nobody has that. What this does instead: trade a defined, testable
rule (EMA crossover + RSI momentum filter), size positions by risk rather
than gut feel, cut losers with a stop-loss, and halt for the day if losses
get too deep. That is what a real trading system looks like. Past backtest
performance does not guarantee future results. Only trade money you can
afford to lose. This is not financial advice.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then edit as needed
```

## Configuration (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `EXCHANGE_ID` | `binance` | Any [ccxt](https://github.com/ccxt/ccxt) exchange id |
| `SYMBOL` | `BTC/USDT` | Trading pair |
| `TIMEFRAME` | `1h` | Candle timeframe |
| `MODE` | `paper` | `paper` or `live` |
| `STARTING_BALANCE` | `10000` | Simulated starting cash (paper mode only) |
| `POLL_INTERVAL_SECONDS` | `60` | How often the live loop checks for new signals |
| `EXCHANGE_API_KEY` / `EXCHANGE_API_SECRET` | _(empty)_ | Only needed for live mode |
| `I_UNDERSTAND_LIVE_TRADING_RISK` | _(empty)_ | Must be set to `yes` to unlock live mode |

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
3. Take the best params from step 2 and set them in `tradingbot/config.py`
   (`StrategyParams` defaults), or wire them into `.env` if you prefer.
4. **Paper trade** to see it run against live market data with fake money:
   ```bash
   MODE=paper python main.py run
   ```
5. Only after you're satisfied with paper results, **go live**:
   ```bash
   EXCHANGE_API_KEY=... EXCHANGE_API_SECRET=... \
   I_UNDERSTAND_LIVE_TRADING_RISK=yes MODE=live python main.py run
   ```
   The bot refuses to place real orders unless both the confirmation
   variable and valid API keys are present.

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
  strategy.py    # signal generation
  risk.py        # position sizing, stop/take levels, daily circuit breaker
  backtester.py  # event-driven backtest over historical OHLCV
  optimizer.py   # walk-forward grid search over strategy params
  brokers.py     # PaperBroker (simulated) and LiveBroker (real ccxt orders)
  data.py        # OHLCV fetching via ccxt
  bot.py         # main run loop
main.py          # CLI: backtest / optimize / run
tests/           # pytest unit tests (synthetic data, no network calls)
```

## Testing

```bash
python -m pytest
```

Tests use synthetic price data and never hit the network.
