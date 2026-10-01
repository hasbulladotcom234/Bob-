"""CLI entry point.

Usage:
    python main.py backtest        # backtest current strategy params on historical data
    python main.py optimize        # grid-search + walk-forward validate strategy params
    python main.py run             # run the bot loop (paper by default; see .env / README)
"""
import sys

from tradingbot.backtester import run_backtest
from tradingbot.bot import run_forever
from tradingbot.config import Config
from tradingbot.data import exchange_from_config, fetch_ohlcv
from tradingbot.optimizer import grid_search


def cmd_backtest():
    config = Config()
    df = fetch_ohlcv(exchange_from_config(config), config.symbol, config.timeframe, limit=1000)
    result = run_backtest(df, config)
    print(f"Total return:     {result.total_return_pct:.2f}%")
    print(f"Max drawdown:     {result.max_drawdown_pct:.2f}%")
    print(f"Win rate:         {result.win_rate_pct:.2f}%")
    print(f"Number of trades: {result.num_trades}")
    print(f"Sharpe (approx):  {result.sharpe:.2f}")


def cmd_optimize():
    config = Config()
    df = fetch_ohlcv(exchange_from_config(config), config.symbol, config.timeframe, limit=1500)
    result = grid_search(df, config)
    print("Best params found on training data:")
    print(result.best_params)
    print(f"Train score: {result.train_score:.3f}")
    print(f"Test score (out-of-sample): {result.test_score:.3f}")
    print("\nTop 10 parameter sets:")
    print(result.leaderboard.head(10).to_string(index=False))


def cmd_run():
    run_forever(Config())


COMMANDS = {"backtest": cmd_backtest, "optimize": cmd_optimize, "run": cmd_run}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(1)
    COMMANDS[sys.argv[1]]()
