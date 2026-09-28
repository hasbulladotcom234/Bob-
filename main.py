"""CLI entry point.

Usage:
    python main.py backtest        # backtest current strategy params on historical data
    python main.py optimize        # grid-search + walk-forward validate strategy params
    python main.py calibrate       # fit stochastic market models (GBM/Merton/Heston/Bates) to history
    python main.py montecarlo      # backtest across simulated price histories from a calibrated model
    python main.py run             # run the bot loop (paper by default; see .env / README)
"""
import argparse

from tradingbot.backtester import run_backtest
from tradingbot.bot import run_forever
from tradingbot.config import Config
from tradingbot.data import bars_per_year, fetch_ohlcv
from tradingbot.montecarlo import run_monte_carlo
from tradingbot.optimizer import grid_search
from tradingbot.simulation import MODELS, calibrate
from tradingbot.strategy import generate_signals


def _fetch(config: Config, limit: int):
    df = fetch_ohlcv(config.exchange_id, config.symbol, config.timeframe, limit=limit)
    print(f"Loaded {len(df)} {config.timeframe} candles of {config.symbol} from {config.exchange_id} "
          f"({df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]})\n")
    return df


def cmd_backtest(args):
    config = Config()
    df = _fetch(config, args.bars)
    result = run_backtest(df, config)
    print(f"Total return:     {result.total_return_pct:.2f}%")
    print(f"Max drawdown:     {result.max_drawdown_pct:.2f}%")
    print(f"Win rate:         {result.win_rate_pct:.2f}%")
    print(f"Number of trades: {result.num_trades}")
    print(f"Sharpe (approx):  {result.sharpe:.2f}")


def cmd_optimize(args):
    config = Config()
    df = _fetch(config, args.bars)
    result = grid_search(df, config)
    print("Best params found on training data:")
    print(result.best_params)
    print(f"Train score: {result.train_score:.3f}")
    print(f"Test score (out-of-sample): {result.test_score:.3f}")
    print("\nTop 10 parameter sets:")
    print(result.leaderboard.head(10).to_string(index=False))


def cmd_calibrate(args):
    config = Config()
    df = _fetch(config, args.bars)
    per_year = bars_per_year(config.timeframe)
    for model in MODELS:
        d = calibrate(df, model).describe(per_year)
        print(f"{model:>7}: drift {d['annual_drift_pct']:+.1f}%/yr, vol {d['annual_vol_pct']:.1f}%/yr, "
              f"jumps {d['jumps_per_year']:.1f}/yr (mean {d['jump_mean'] * 100:+.2f}%, sd {d['jump_std'] * 100:.2f}%), "
              f"vol half-life {d['vol_half_life_bars']:.0f} bars, rho {d['rho']:+.2f}, xi {d['xi']:.2e}")

    s = config.strategy
    last = generate_signals(df, s, config.round_trip_cost()).iloc[-1]
    print(f"\nRight now, for a {s.stop_loss_pct:.0%} stop / {s.take_profit_pct:.0%} target bracket:")
    print(f"  P(take-profit before stop-loss): {last['p_take'] * 100:.1f}%")
    print(f"  Expected bars until exit:        {last['expected_bars']:.0f}")
    print(f"  Expected return after costs:     {last['edge'] * 100:+.2f}% "
          f"(round-trip cost {config.round_trip_cost() * 100:.2f}%)")


def cmd_montecarlo(args):
    config = Config()
    df = _fetch(config, args.bars)
    mc = run_monte_carlo(df, config, model=args.model, n_paths=args.paths,
                         drift=args.drift, seed=args.seed)
    real = run_backtest(df, config)
    s = mc.summary
    print(f"Model: {args.model}, drift: {args.drift}, {s['paths']} simulated paths of {len(df)} bars\n")
    print(f"Mean return:            {s['mean_return_pct']:+.2f}%")
    print(f"Median return:          {s['median_return_pct']:+.2f}%")
    print(f"Chance of losing money: {s['prob_loss_pct']:.1f}%")
    print(f"5% VaR (return):        {s['var_5_pct']:+.2f}%")
    print(f"5% CVaR (return):       {s['cvar_5_pct']:+.2f}%   (average of the worst 5% of paths)")
    print(f"Median max drawdown:    {s['median_max_drawdown_pct']:.2f}%")
    print(f"Worst max drawdown:     {s['worst_max_drawdown_pct']:.2f}%")
    print(f"Avg trades per path:    {s['mean_trades']:.1f}")
    print(f"Median Sharpe:          {s['median_sharpe']:.2f}")
    print(f"Median buy & hold:      {mc.paths['buy_and_hold_pct'].median():+.2f}%")
    pct_rank = (mc.paths["total_return_pct"] < real.total_return_pct).mean() * 100
    print(f"\nReal-history backtest returned {real.total_return_pct:+.2f}%, "
          f"better than {pct_rank:.0f}% of simulated paths.")


def cmd_run(args):
    run_forever(Config())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    for name, func, bars in (("backtest", cmd_backtest, 1000), ("optimize", cmd_optimize, 1500),
                             ("calibrate", cmd_calibrate, 1000), ("montecarlo", cmd_montecarlo, 1000)):
        p = sub.add_parser(name)
        p.add_argument("--bars", type=int, default=bars, help="candles of history to fetch")
        p.set_defaults(func=func)
        if name == "montecarlo":
            p.add_argument("--model", choices=MODELS, default="bates")
            p.add_argument("--paths", type=int, default=200)
            p.add_argument("--drift", choices=("historical", "zero"), default="historical",
                           help="'zero' strips the calibrated trend to test for real edge")
            p.add_argument("--seed", type=int, default=None)

    sub.add_parser("run").set_defaults(func=cmd_run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
