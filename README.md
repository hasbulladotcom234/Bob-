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

## Quick start: paper trading in about 2 minutes

1. Download the code (on GitHub: **Code → Download ZIP**) and unzip it.
   You need [Python](https://www.python.org/downloads/) installed (on
   Windows, tick "Add to PATH" in the installer).
2. Double-click **`start.bat`** (Windows) or **`start.command`** (Mac; if it
   won't open, run `bash start.command` in Terminal).
3. The first time, it asks for your Alpaca **paper** API key and secret,
   checks them with Alpaca, and saves them. Then the bot starts trading on
   your paper account. Next time, double-clicking just starts the bot.

Other commands (run in a terminal in the project folder):

```bash
python main.py status     # balance, position and what the strategy says now (never trades)
python main.py explore    # statistical facts about the market (no keys needed)
python main.py backtest   # test my_strategy.py on history (no keys needed)
python main.py optimize   # tune my_strategy.py's parameters (no keys needed)
python main.py setup      # re-enter your paper keys
```

## How the project is split

- **`my_strategy.py` is yours.** It holds the trading idea: which
  indicators to compute (`indicators`), when to buy or sell (`decide`), the
  numbers you might tune (`PARAMS`), and the values the optimizer should try
  (`PARAM_GRID`). It's the only file you need to touch to try a new idea.
- **`tradingbot/` is the plumbing.** It fetches data, backtests, optimizes,
  sizes positions, places orders on Alpaca, remembers the position across
  restarts, and keeps a trade journal. The same `decide()` runs in the
  backtest and live, so they can't drift apart.

Timing rules, identical in backtest and live: `decide()` only sees bars that
have closed, orders fill at the next bar's open (in live, the current price
right after the close), and stop-loss / take-profit are checked continuously.
The backtest also warns you if an indicator uses future data.

## Configuration (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `EXCHANGE_ID` | `alpaca` | Any [ccxt](https://github.com/ccxt/ccxt) exchange id |
| `SYMBOL` | `BTC/USD` | Trading pair |
| `TIMEFRAME` | `1h` | Candle timeframe |
| `MODE` | `paper` | `paper` (local simulation), `sandbox` (exchange paper account, e.g. Alpaca paper), or `live` |
| `STARTING_BALANCE` | `10000` | Simulated starting cash (backtests and `paper` mode) |
| `POLL_INTERVAL_SECONDS` | `60` | How often the live loop checks prices |
| `HISTORY_BARS` | `5000` | How many bars `backtest` / `optimize` download |
| `TAKER_FEE_PCT` | `0.0025` | Fee per fill used in backtests/paper (Alpaca crypto: 0.25% at the lowest tier) |
| `STRATEGY` | `my_strategy` | Strategy file to load (without `.py`), to keep several ideas side by side |
| `STATE_FILE` / `TRADE_LOG` | `state.json` / `trades.csv` | Where the bot saves its position and logs every fill |
| `EXCHANGE_API_KEY` / `EXCHANGE_API_SECRET` | _(empty)_ | Needed for `sandbox` and `live` only |
| `I_UNDERSTAND_LIVE_TRADING_RISK` | _(empty)_ | Must be set to `yes` to unlock live mode |

## Workflow

1. **Edit `my_strategy.py`** with your idea.
2. **Backtest** it:
   ```bash
   python main.py backtest
   ```
   Compare the strategy return against buy & hold. `backtest_trades.csv`
   and `backtest_equity.csv` are saved so you can dig into individual trades.
3. **Optimize** over `PARAM_GRID`:
   ```bash
   python main.py optimize
   ```
   Every combination is scored on the first 70% of the data; only the
   winner is run on the last 30%. If the test score is much worse than
   train, the parameters are overfit, so don't trust them. Copy parameters
   you do trust into `PARAMS`.
4. **Paper trade** on Alpaca's paper account (see below):
   ```bash
   MODE=sandbox python main.py run
   ```
   Every fill goes to `trades.csv`. Compare it with what the backtest
   predicted for the same period.
5. Only after you're satisfied with paper results, **go live**:
   ```bash
   I_UNDERSTAND_LIVE_TRADING_RISK=yes MODE=live python main.py run
   ```
   with live (not paper) API keys. The bot refuses to place real orders
   unless both the confirmation variable and keys are present.

## Alpaca paper trading

`MODE=sandbox` sends real orders to an exchange's paper/testnet account, so
you see real fills and your positions show up in the exchange's app, but no
real money moves. For Alpaca:

1. Log in at [alpaca.markets](https://alpaca.markets), switch to your
   **Paper** account (top-left account switcher), and generate API keys on
   the Home page. Paper keys are different from live keys.
2. Put them in `.env`:
   ```
   EXCHANGE_ID=alpaca
   SYMBOL=BTC/USD
   MODE=sandbox
   EXCHANGE_API_KEY=your-paper-key-id
   EXCHANGE_API_SECRET=your-paper-secret
   ```
3. Run `python main.py run`. Trades appear in the Alpaca app under your
   paper account.

Sandbox mode always routes to the exchange's paper endpoints
(`paper-api.alpaca.markets` for Alpaca) and refuses to start if the
exchange has none, so it can't accidentally hit a real-money API. The bot
trades crypto only (Alpaca supports pairs like `BTC/USD`, `ETH/USD`).

The bot assumes it's the only thing trading that coin on the account. On
startup it checks its saved position against the real balance. If you sell
manually in the app, it notices and stops tracking that position. It leaves
coins it didn't buy alone.

## Risk controls (always on, paper or live)

- **Position sizing**: each trade risks at most `risk_per_trade_pct` of
  equity (default 1%), capped at `max_position_pct` of equity in any one
  trade (default 25%).
- **Stop-loss / take-profit**: every position has both, set in
  `my_strategy.py` (`stop_loss_pct`, `take_profit_pct`), checked every
  bar in backtests and every poll live.
- **Daily circuit breaker**: no new trades for the rest of the day if
  losses exceed `max_daily_loss_pct` of the day's starting equity
  (default 5%).

Tune these in `tradingbot/config.py` (`RiskParams`).

## Project layout

```
my_strategy.py   # YOUR strategy: indicators, buy/sell rules, parameters
main.py          # CLI: backtest / optimize / run
tradingbot/
  config.py      # settings, env vars, risk limits, live-trading safety gate
  indicators.py  # EMA, RSI (add your own helpers here)
  strategy.py    # loads my_strategy.py, lookahead check
  risk.py        # position sizing, stop/take levels, daily circuit breaker
  backtester.py  # bar-by-bar backtest with next-open fills
  optimizer.py   # grid search over PARAM_GRID with a train/test split
  brokers.py     # PaperBroker (simulated) and ExchangeBroker (ccxt orders, sandbox or live)
  data.py        # ccxt exchange setup, history paging, closed-bar filtering
  bot.py         # live loop, saved state, trade journal
tests/           # pytest unit tests (synthetic data, no network calls)
```

## Testing

```bash
python -m pytest
```

Tests use synthetic price data and never hit the network.
