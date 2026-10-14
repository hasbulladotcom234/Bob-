"""Signal research: does a score predict which coins do better next?

A signal gives every coin a score on every day. This module measures, across
all coins and all days, whether higher scores were followed by higher
returns, the way a quant research team would:

1. Information coefficient (IC): each day, the rank correlation between the
   coins' scores and their next returns (+1 = the ranking was perfect,
   0 = no link, -1 = exactly backwards). Averaged over all days, with a
   t-statistic for "could this be luck?".
2. Group portfolios: every `rebalance` days, split the coins into groups by
   score (top third, middle, bottom third), hold each group in equal amounts,
   and track what each group made, after trading fees.
3. Stability: the IC year by year, and coin by coin.

Prices are daily bars built from the cleaned hourly data. Everything at or
after the holdout date is cut off BEFORE anything is computed, so no number
here has ever seen the locked period.
"""
import inspect
import types
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .metrics import performance

MIN_COINS = 5   # days with fewer coins scored than this are skipped (too few to rank)
FIELDS = ("open", "high", "low", "close", "volume")


# ---- data ------------------------------------------------------------------

def daily_bars(df: pd.DataFrame, freq: str = "1D") -> pd.DataFrame:
    """Resample one coin's candles into bigger bars (e.g. hourly -> daily).
    Periods with no candles at all stay empty (NaN): data holes and the time
    after a coin was delisted are never filled in."""
    bars = df.set_index("timestamp").resample(freq).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    bars.loc[bars["close"].isna(), "volume"] = np.nan
    return bars


def load_research_data(store, symbols, timeframe: str = "1h", freq: str = "1D",
                       start=None, end_before=None) -> types.SimpleNamespace:
    """One table per price field (rows = days, columns = coins), from cleaned
    data, cut off before `end_before` (the holdout date)."""
    per_coin = {}
    for sym in symbols:
        df = store.load(sym, timeframe)
        if df.empty:
            continue
        if end_before is not None:
            df = df[df["timestamp"] < pd.Timestamp(end_before)]
        if start is not None:
            df = df[df["timestamp"] >= pd.Timestamp(start)]
        if not df.empty:
            per_coin[sym] = daily_bars(df, freq)
    if not per_coin:
        raise ValueError("No data to research. Run: python main.py data update")
    index = pd.date_range(min(b.index.min() for b in per_coin.values()),
                          max(b.index.max() for b in per_coin.values()), freq=freq)
    tables = {f: pd.DataFrame({s: b[f] for s, b in per_coin.items()}).reindex(index) for f in FIELDS}
    return types.SimpleNamespace(**tables)


# ---- the measurements -------------------------------------------------------

def forward_returns(close: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Return from each day's close to the close `horizon` days later."""
    return close.shift(-horizon) / close - 1


def cross_sectional_ic(signal: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    """Each day: Spearman rank correlation between the coins' scores and their
    forward returns (only coins that have both)."""
    both = signal.notna() & fwd.notna()
    s = signal.where(both).rank(axis=1)
    f = fwd.where(both).rank(axis=1)
    s = s.sub(s.mean(axis=1), axis=0)
    f = f.sub(f.mean(axis=1), axis=0)
    ic = (s * f).sum(axis=1) / np.sqrt((s ** 2).sum(axis=1) * (f ** 2).sum(axis=1))
    return ic.where(both.sum(axis=1) >= MIN_COINS).dropna()


def ic_stats(ic: pd.Series, horizon: int) -> dict:
    """Mean IC and how sure we can be it isn't luck.

    With a horizon of h days, consecutive days' forward returns overlap (they
    share h-1 days), so there are only about n/h independent observations.
    Using all n would overstate the evidence, so the t-stat uses n/h.
    """
    n = len(ic)
    if n < 2 or ic.std() == 0:
        return {"mean_ic": np.nan, "t_stat": np.nan, "pct_positive": np.nan, "days": n}
    effective_n = n / horizon
    return {"mean_ic": ic.mean(), "t_stat": ic.mean() / ic.std() * np.sqrt(effective_n),
            "pct_positive": 100 * (ic > 0).mean(), "days": n}


def group_assignments(signal: pd.DataFrame, groups: int) -> pd.DataFrame:
    """Each day, put each scored coin into a group 1 (lowest scores) .. `groups`
    (highest). Days with too few coins get no groups."""
    pct = signal.rank(axis=1, pct=True)
    g = np.ceil(pct * groups).clip(1, groups)
    enough = signal.notna().sum(axis=1) >= MIN_COINS
    return g.where(enough, axis=0)


@dataclass
class GroupBacktest:
    daily: pd.DataFrame            # daily return of each group, the market, top - bottom, top - market
    turnover: float                # average share of the top group's holdings replaced per rebalance
    costs_per_year_pct: float      # fees per year from replacing holdings (top group)
    stats: pd.DataFrame = field(default=None)


def group_backtest(signal: pd.DataFrame, close: pd.DataFrame, groups: int = 3, rebalance: int = 7,
                   fee: float = 0.0025) -> GroupBacktest:
    """Hold each score group in equal amounts, re-sorting every `rebalance`
    days. A group decided at day t's close earns returns from day t+1, so no
    group ever uses a price it couldn't have known. Fees are charged on the
    top group (the one you'd actually buy) and on both ends of top - bottom.
    """
    returns = close.pct_change(fill_method=None)
    assigned = group_assignments(signal, groups)
    valid_days = assigned.notna().any(axis=1)
    rebal_days = assigned.index[valid_days][::rebalance]
    at_rebal = assigned.loc[rebal_days].fillna(0)          # 0 = not held
    held = at_rebal.reindex(assigned.index).ffill().shift(1)

    out = pd.DataFrame(index=close.index)
    for g in range(1, groups + 1):
        out[f"group_{g}"] = returns.where(held == g).mean(axis=1)
    out["market"] = returns.mean(axis=1)

    # fees: on each re-sort, how much of a group's holdings is swapped out.
    # weights: each member holds 1/(number of members); traded = total change in weights.
    def traded_per_rebalance(g):
        members = (at_rebal == g).astype(float)
        w = members.div(members.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
        return (w - w.shift(1).fillna(0)).abs().sum(axis=1)   # first row = the initial buy

    traded = {g: traded_per_rebalance(g) for g in (groups, 1)}
    positions = close.index.get_indexer(rebal_days) + 1           # fees are paid the day after a re-sort
    ok = positions < len(close.index)
    pay_days = close.index[positions[ok]]
    top_cost = pd.Series(fee * traded[groups].values[ok], index=pay_days)
    bottom_cost = pd.Series(fee * traded[1].values[ok], index=pay_days)

    top, bottom = out[f"group_{groups}"], out["group_1"]
    out["top_after_fees"] = top.sub(top_cost.reindex(out.index).fillna(0))
    out["top_minus_market"] = out["top_after_fees"] - out["market"]
    out["top_minus_bottom"] = (top - bottom).sub(
        (top_cost.reindex(out.index).fillna(0) + bottom_cost.reindex(out.index).fillna(0)))
    out = out.dropna(how="all")

    one_way = traded[groups].iloc[1:] / 2                    # skip the initial buy
    rebalances_per_year = 365 / rebalance
    turnover = float(one_way.mean()) if len(one_way) else 0.0
    return GroupBacktest(daily=out, turnover=turnover,
                         costs_per_year_pct=100 * fee * 2 * turnover * rebalances_per_year)


def placebo_test(signal: pd.DataFrame, close: pd.DataFrame, groups: int = 3, rebalance: int = 7,
                 runs: int = 200, seed: int = 0) -> dict:
    """How often does a scrambled version of the signal do as well?

    Each run gives every coin another coin's scores (the columns are shuffled),
    which keeps everything about the signal (how often it changes, how spread
    out it is) except its link to the coin it describes. If scrambled signals
    match the real one's top-minus-bottom return often, the real result is
    probably luck. Fees are left out on both sides so they compare like for like.
    """
    def spread(sig):
        daily = group_backtest(sig, close, groups=groups, rebalance=rebalance, fee=0).daily
        return daily["top_minus_bottom"].mean() * 365

    actual = spread(signal)
    rng = np.random.default_rng(seed)
    cols = list(signal.columns)
    fake = []
    for _ in range(runs):
        shuffled = signal.copy()
        shuffled.columns = list(rng.permutation(cols))
        fake.append(spread(shuffled[cols].where(close.notna())))
    fake = np.array(fake)
    return {"actual_pct": 100 * actual, "placebo_p5_pct": 100 * np.percentile(fake, 5),
            "placebo_p95_pct": 100 * np.percentile(fake, 95),
            "share_as_good": float((fake >= actual).mean()) if actual >= 0 else float((fake <= actual).mean())}


def lookahead_check(signal_fn, data: types.SimpleNamespace, cut: float = 0.6) -> bool:
    """True if the signal uses future data: computing it on the first 60% of
    the days must give the same numbers as computing it on everything."""
    n = int(len(data.close) * cut)
    if n < 2:
        return False
    head = types.SimpleNamespace(**{f: getattr(data, f).iloc[:n] for f in FIELDS})
    full = signal_fn(data).iloc[:n]
    part = signal_fn(head)
    a, b = full.astype(float), part.astype(float)
    same = ((a - b).abs() <= 1e-9 * (1 + b.abs())) | (a.isna() & b.isna())
    return not bool(same.all().all())


# ---- the full report ----------------------------------------------------------

@dataclass
class ResearchResult:
    name: str
    description: str
    start: pd.Timestamp
    end: pd.Timestamp
    coins_per_day: float
    uses_future: bool
    ic_by_horizon: pd.DataFrame
    ic_by_year: pd.DataFrame
    ic_by_coin: pd.DataFrame
    groups: GroupBacktest
    rebalance: int
    placebo: dict = None


def per_coin_ic(signal: pd.DataFrame, fwd: pd.DataFrame) -> pd.DataFrame:
    """For each coin on its own: over time, did a higher score than usual come
    before a higher return than usual? (Spearman correlation through time.)"""
    rows = []
    for coin in signal.columns:
        both = pd.concat([signal[coin], fwd[coin]], axis=1).dropna()
        if len(both) >= 30:
            rows.append({"coin": coin, "ic": both.iloc[:, 0].rank().corr(both.iloc[:, 1].rank()), "days": len(both)})
    return pd.DataFrame(rows).set_index("coin").sort_values("ic") if rows else pd.DataFrame()


def run_research(signal_fn, data: types.SimpleNamespace, horizons=(1, 3, 7, 14, 30), rebalance: int = 7,
                 groups: int = 3, fee: float = 0.0025, placebo_runs: int = 200) -> ResearchResult:
    signal = signal_fn(data)
    if not isinstance(signal, pd.DataFrame) or signal.shape != data.close.shape:
        raise ValueError(f"{signal_fn.__name__} must return a table shaped like d.close "
                         f"(one row per day, one column per coin)")
    signal = signal.replace([np.inf, -np.inf], np.nan).where(data.close.notna())

    rows = []
    for h in horizons:
        rows.append({"horizon_days": h, **ic_stats(cross_sectional_ic(signal, forward_returns(data.close, h)), h)})
    ic_h = pd.DataFrame(rows).set_index("horizon_days")

    main_ic = cross_sectional_ic(signal, forward_returns(data.close, rebalance))
    by_year = pd.DataFrame({year: ic_stats(ic, rebalance) for year, ic in main_ic.groupby(main_ic.index.year)}).T
    by_year.index.name = "year"

    gb = group_backtest(signal, data.close, groups=groups, rebalance=rebalance, fee=fee)
    curves = (1 + gb.daily.fillna(0)).cumprod()
    gb.stats = pd.DataFrame({c: performance(curves[c]) for c in curves.columns}).T

    scored = signal.notna().sum(axis=1)
    return ResearchResult(
        name=signal_fn.__name__,
        description=(inspect.getdoc(signal_fn) or "(no description)").strip(),
        start=signal.index[scored >= MIN_COINS].min(), end=signal.index.max(),
        coins_per_day=float(scored[scored >= MIN_COINS].mean()),
        uses_future=lookahead_check(signal_fn, data),
        ic_by_horizon=ic_h, ic_by_year=by_year,
        ic_by_coin=per_coin_ic(signal, forward_returns(data.close, rebalance)),
        groups=gb, rebalance=rebalance,
        placebo=placebo_test(signal, data.close, groups, rebalance, placebo_runs) if placebo_runs else None,
    )


def format_report(r: ResearchResult, holdout_start=None) -> str:
    f2 = lambda x: f"{x:.2f}"
    f3 = lambda x: f"{x:.3f}"
    lines = []
    add = lines.append
    add(f"Signal: {r.name}")
    add(f"Idea:   {r.description}")
    add(f"Data:   {r.start:%Y-%m-%d} -> {r.end:%Y-%m-%d}, about {r.coins_per_day:.0f} coins per day"
        + (f" (locked holdout from {pd.Timestamp(holdout_start):%Y-%m-%d} not used)" if holdout_start else ""))
    if r.uses_future:
        add("\n!!! WARNING: this signal uses FUTURE data (e.g. .shift(-1)). Every number below is fake.")

    add("\n== 1. Does it predict? Information coefficient (IC) ==")
    add("Each day: rank correlation between the coins' scores and their returns over the next N days.")
    add("mean_ic: +0.05 is already useful in practice. t_stat: about 2 or more = unlikely to be luck (for ONE idea).")
    add(r.ic_by_horizon.to_string(formatters={"mean_ic": f3, "t_stat": f2, "pct_positive": lambda x: f"{x:.0f}%"}))

    g = r.groups
    add(f"\n== 2. Portfolios: hold each third of the coins by score, re-sorted every {r.rebalance} days ==")
    add("group_1 = lowest scores ... highest group = highest scores. If the idea works, returns climb from")
    add("group_1 up to the top group. top_after_fees = what buying the top group would have made after fees.")
    add("top_minus_market = top group vs simply holding every coin. top_minus_bottom = long-short spread")
    add("(a research yardstick: shorting crypto isn't possible on Alpaca).")
    cols = ["cagr_pct", "ann_vol_pct", "sharpe", "max_drawdown_pct"]
    add(g.stats[cols].to_string(float_format=f2))
    add(f"Top group: {100 * g.turnover:.0f}% of holdings replaced per re-sort, about {g.costs_per_year_pct:.1f}% "
        f"a year in fees (already subtracted above).")
    if r.placebo:
        p = r.placebo
        add(f"\nLuck check (placebo): each coin given another coin's scores, many times.")
        add(f"  Your top-minus-bottom (before fees): {p['actual_pct']:+.1f}% a year.")
        add(f"  Scrambled scores usually land between {p['placebo_p5_pct']:+.1f}% and {p['placebo_p95_pct']:+.1f}% a year "
            f"by pure luck.")
        add(f"  Scrambled scores did at least as well {100 * p['share_as_good']:.0f}% of the time"
            + (" -> could easily be luck." if p["share_as_good"] > 0.05 else " -> unlikely to be luck (for ONE idea)."))

    add(f"\n== 3. Year by year ({r.rebalance}-day IC) ==")
    add("A real effect shows up in most years, not just one.")
    add(r.ic_by_year[["mean_ic", "t_stat", "pct_positive"]].to_string(
        formatters={"mean_ic": f3, "t_stat": f2, "pct_positive": lambda x: f"{x:.0f}%"}))

    if len(r.ic_by_coin):
        add(f"\n== 4. Coin by coin ({r.rebalance}-day, through time) ==")
        add("For each coin alone: did a higher score than usual come before a better return than usual?")
        positive = (r.ic_by_coin["ic"] > 0).sum()
        add(f"{positive} of {len(r.ic_by_coin)} coins positive. Lowest and highest:")
        shown = pd.concat([r.ic_by_coin.head(3), r.ic_by_coin.tail(3)]).drop_duplicates()
        add(shown.to_string(float_format=f3))
    return "\n".join(lines)
