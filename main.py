"""CLI entry point.

Usage:
    python main.py backtest        # test my_strategy.py on historical data
    python main.py optimize        # search my_strategy.py's PARAM_GRID with a train/test split
    python main.py run             # run the bot (MODE=paper / sandbox / live, see .env / README)
"""
import sys

from tradingbot.backtester import run_backtest
from tradingbot.bot import run_forever
from tradingbot.config import Config
from tradingbot.data import exchange_from_config, fetch_history
from tradingbot.optimizer import grid_search


def load_history(config):
    df = fetch_history(exchange_from_config(config), config.symbol, config.timeframe, config.history_bars)
    print(f"{len(df)} {config.timeframe} bars of {config.symbol}: "
          f"{df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}\n")
    return df


def cmd_backtest():
    config = Config()
    result = run_backtest(load_history(config), config)
    if result.lookahead_warning:
        print(f"WARNING: indicator columns {result.lookahead_warning} use future data. "
              f"These results are not trustworthy.\n")
    print(f"Strategy return:   {result.total_return_pct:8.2f}%")
    print(f"Buy & hold return: {result.buy_hold_return_pct:8.2f}%")
    print(f"Max drawdown:      {result.max_drawdown_pct:8.2f}%")
    print(f"Sharpe (annual):   {result.sharpe:8.2f}")
    print(f"Trades:            {result.num_trades:8d}")
    print(f"Win rate:          {result.win_rate_pct:8.2f}%")
    print(f"Time in market:    {result.exposure_pct:8.2f}%")
    if result.num_trades:
        print("\nExit reasons:", result.trades["reason"].value_counts().to_dict())
        result.trades.to_csv("backtest_trades.csv", index=False)
        result.equity_curve.rename("equity").to_csv("backtest_equity.csv")
        print("Saved backtest_trades.csv and backtest_equity.csv for analysis.")


def cmd_optimize():
    config = Config()
    result = grid_search(load_history(config), config)
    print(f"Tried {result.combos_tried} combinations.")
    print("Best params on training data:", result.best_params)
    print(f"Train score:            {result.train_score:.3f}")
    print(f"Test score (unseen):    {result.test_score:.3f}")
    print(f"Test return:            {result.test_return_pct:.2f}%  (buy & hold: {result.test_buy_hold_pct:.2f}%)")
    print("\nTop 10 on training data:")
    print(result.leaderboard.head(10).to_string(index=False))


def cmd_run():
    run_forever(Config())


COMMANDS = {"backtest": cmd_backtest, "optimize": cmd_optimize, "run": cmd_run}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(1)
    COMMANDS[sys.argv[1]]()
