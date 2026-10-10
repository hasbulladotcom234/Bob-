"""Command-line entry point. Run `python main.py --help` for the full list.

    python main.py setup              connect your Alpaca paper account
    python main.py status             balance, position and current signal (never trades)
    python main.py run                run the bot (MODE=paper / sandbox / live, see README)

    python main.py data update        download / update history for every coin
    python main.py data check         data quality report
    python main.py data benchmarks    how every coin and the basket performed
    python main.py data inspect PEPE/USD   holes, bad prices and biggest moves for one coin

    python main.py explore            statistical facts about one coin
    python main.py backtest           test my_strategy.py on history
    python main.py optimize           tune my_strategy.py's PARAM_GRID with a train/test split
"""
import argparse
import logging
import sys

import pandas as pd

from tradingbot.config import Config

log = logging.getLogger("tradingbot")


def load_history(config):
    """History for config.symbol from the local store, updating it first if
    the network is available."""
    from tradingbot.data import research_exchange
    from tradingbot.store import BarStore

    store = BarStore(exchange_id=config.exchange_id)
    try:
        store.update(research_exchange(config), config.symbol, config.timeframe)
    except Exception as err:
        print(f"Couldn't update {config.symbol} ({type(err).__name__}: {err}); using stored data.")
    df = store.load(config.symbol, config.timeframe)
    if df.empty:
        sys.exit(f"No data for {config.symbol} {config.timeframe}. Check your internet connection "
                 f"and run: python main.py data update --symbols {config.symbol}")
    df = df.tail(config.history_bars).reset_index(drop=True)
    print(f"{len(df)} {config.timeframe} bars of {config.symbol}: "
          f"{df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}\n")
    return df


# ---- trading ---------------------------------------------------------------

def cmd_setup(args):
    from tradingbot.setup_wizard import run_setup
    run_setup()


def cmd_status(args):
    from tradingbot.bot import Bot
    config = Config()
    print(f"Mode: {config.mode} | {config.exchange_id} {config.symbol} {config.timeframe}")
    Bot(config).step(dry_run=True)  # read-only: never places orders


def cmd_run(args):
    from tradingbot.bot import run_forever
    run_forever(Config())


# ---- data ------------------------------------------------------------------

def _symbols(args, config):
    from tradingbot.universe import default_universe
    return [s.strip() for s in args.symbols.split(",")] if args.symbols else default_universe()


def cmd_data_update(args):
    from tradingbot.data import research_exchange
    from tradingbot.store import BarStore
    config = Config()
    timeframe = args.timeframe or config.timeframe
    store, exchange = BarStore(exchange_id=config.exchange_id), research_exchange(config)
    symbols = _symbols(args, config)
    print(f"Updating {len(symbols)} symbols, {timeframe} bars, from {args.since} if new...\n")
    for sym in symbols:
        try:
            r = store.update(exchange, sym, timeframe, since=args.since)
        except Exception as err:
            print(f"  {sym:10} FAILED: {type(err).__name__}: {err}")
            continue
        if r.total_rows == 0:
            print(f"  {sym:10} no data (not listed on {config.exchange_id}?)")
        else:
            print(f"  {sym:10} +{r.new_rows:>6} bars  ({r.total_rows} total, {r.first} -> {r.last})")
    print(f"\nStored under {store.root}/. Next: python main.py data check")


def cmd_data_check(args):
    from tradingbot.data import make_exchange
    from tradingbot.quality import quality_report
    from tradingbot.store import BarStore
    config = Config()
    timeframe = args.timeframe or config.timeframe
    store = BarStore(exchange_id=config.exchange_id)
    tf_seconds = make_exchange(config.exchange_id).parse_timeframe(timeframe)
    report = quality_report(store, timeframe, tf_seconds)
    if report.empty:
        sys.exit("No stored data yet. Run: python main.py data update")
    cols = ["symbol", "status", "rows", "first", "last", "coverage_pct", "bad_prints", "bad_wicks", "issues"]
    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        print(report[cols].to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    print("\nstatus: ok = clean | warn = usable, read the issue | bad = corrupt bars, don't trust it")


def cmd_data_benchmarks(args):
    from tradingbot.metrics import performance_table
    from tradingbot.panel import benchmarks, load_panel, to_daily
    from tradingbot.store import BarStore
    config = Config()
    timeframe = args.timeframe or config.timeframe
    store = BarStore(exchange_id=config.exchange_id)
    symbols = store.symbols(timeframe)
    if not symbols:
        sys.exit("No stored data yet. Run: python main.py data update")
    daily = to_daily(load_panel(store, symbols, timeframe, start=args.start))
    curves = pd.concat([benchmarks(daily), daily / daily.bfill().iloc[0]], axis=1)
    table = performance_table(curves).sort_values("sharpe", ascending=False)
    print(f"{daily.index[0]:%Y-%m-%d} -> {daily.index[-1]:%Y-%m-%d}, daily closes, buy & hold, no costs\n")
    print(table.to_string(float_format=lambda x: f"{x:.2f}"))
    print("\nequal_weight = every coin in equal amounts, rebalanced daily. Coins that started later "
          "(or stopped, like delisted ones) count only while they traded.\n"
          "Volatility and Sharpe skip data holes, so a price jump across a hole isn't counted as one move.")


def cmd_data_inspect(args):
    from tradingbot.cleaning import clean_bars, gaps
    from tradingbot.data import make_exchange
    from tradingbot.store import BarStore
    config = Config()
    timeframe = args.timeframe or config.timeframe
    raw = BarStore(exchange_id=config.exchange_id).load(args.symbol, timeframe, clean=False)
    if raw.empty:
        sys.exit(f"No stored data for {args.symbol} {timeframe}.")
    tf_seconds = make_exchange(config.exchange_id).parse_timeframe(timeframe)
    clean, rep = clean_bars(raw)
    fmt = lambda x: f"{x:.6g}"
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(f"{args.symbol} {timeframe}: {len(raw)} bars, {raw['timestamp'].iloc[0]} -> {raw['timestamp'].iloc[-1]}\n")
        g = gaps(raw, tf_seconds)
        print(f"== Data holes of a day or more: {len(g)}")
        if len(g):
            print(g.to_string(index=False, float_format=fmt))
        print(f"\n== Fixed by cleaning: {rep['bad_closes']} bad closes, {rep['bad_opens']} bad opens, "
              f"{rep['bad_wicks']} bad wicks")
        if len(rep["changes"]):
            print(rep["changes"].head(args.limit).to_string(index=False, float_format=fmt))
            if len(rep["changes"]) > args.limit:
                print(f"... and {len(rep['changes']) - args.limit} more")
        moves = clean.assign(move_pct=100 * clean["close"].pct_change()).dropna()
        top = moves.reindex(moves["move_pct"].abs().sort_values(ascending=False).index).head(10)
        print("\n== Biggest one-bar moves left after cleaning (should look like real market moves):")
        print(top[["timestamp", "open", "high", "low", "close", "move_pct"]].to_string(index=False, float_format=fmt))


# ---- research --------------------------------------------------------------

def cmd_explore(args):
    from tradingbot.explore import report
    print(report(load_history(Config())))


def cmd_backtest(args):
    from tradingbot.backtester import run_backtest
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


def cmd_optimize(args):
    from tradingbot.optimizer import grid_search
    config = Config()
    result = grid_search(load_history(config), config)
    print(f"Tried {result.combos_tried} combinations.")
    print("Best params on training data:", result.best_params)
    print(f"Train score:            {result.train_score:.3f}")
    print(f"Test score (unseen):    {result.test_score:.3f}")
    print(f"Test return:            {result.test_return_pct:.2f}%  (buy & hold: {result.test_buy_hold_pct:.2f}%)")
    print("\nTop 10 on training data:")
    print(result.leaderboard.head(10).to_string(index=False))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn, help_ in [("setup", cmd_setup, "connect your Alpaca paper account"),
                            ("status", cmd_status, "read-only snapshot of the bot"),
                            ("run", cmd_run, "run the trading bot"),
                            ("explore", cmd_explore, "statistical facts about one coin"),
                            ("backtest", cmd_backtest, "backtest my_strategy.py"),
                            ("optimize", cmd_optimize, "tune my_strategy.py")]:
        sub.add_parser(name, help=help_).set_defaults(fn=fn)

    data = sub.add_parser("data", help="download and inspect market data").add_subparsers(dest="data_cmd", required=True)
    up = data.add_parser("update", help="download / update history")
    up.add_argument("--symbols", help="comma-separated, e.g. BTC/USD,ETH/USD (default: whole universe)")
    up.add_argument("--timeframe", help="e.g. 1h, 4h, 1d (default: TIMEFRAME setting)")
    up.add_argument("--since", default="2021-01-01", help="start date for symbols not stored yet")
    up.set_defaults(fn=cmd_data_update)
    chk = data.add_parser("check", help="data quality report")
    chk.add_argument("--timeframe")
    chk.set_defaults(fn=cmd_data_check)
    bm = data.add_parser("benchmarks", help="performance of every coin and the basket")
    bm.add_argument("--timeframe")
    bm.add_argument("--start", help="only from this date, e.g. 2023-01-01")
    bm.set_defaults(fn=cmd_data_benchmarks)
    ins = data.add_parser("inspect", help="holes, bad prices and biggest moves for one coin")
    ins.add_argument("symbol", help="e.g. PEPE/USD")
    ins.add_argument("--timeframe")
    ins.add_argument("--limit", type=int, default=30, help="max fixes to list")
    ins.set_defaults(fn=cmd_data_inspect)
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = build_parser().parse_args()
    try:
        args.fn(args)
    except ModuleNotFoundError as err:
        sys.exit(f"\nMissing add-on '{err.name}'. Install the project's add-ons with:\n"
                 f"    python install.py")
