import ccxt
import numpy as np
import pandas as pd
import pytest

from tests.fakes import FakeAlpacaData, make_bars
from tradingbot.data import fetch_range, with_retries
from tradingbot.panel import benchmarks, load_panel, returns
from tradingbot.quality import check_bars
from tradingbot.store import BarStore

H = 3600_000
T0 = int(pd.Timestamp("2024-01-01").timestamp() * 1000)


def test_fetch_range_follows_page_tokens():
    fake = FakeAlpacaData({"BTC/USD": make_bars(T0, 1200)}, page_size=500)
    df = fetch_range(fake.exchange, "BTC/USD", "1h", T0)
    assert len(df) == 1200
    assert df[["open", "high", "low", "close", "volume"]].notna().all().all()
    assert [r.get("page_token") for r in fake.requests] == [None, "500", "1000"]
    assert fake.requests[0]["timeframe"] == "1H" and fake.requests[0]["loc"] == "us"
    assert df["timestamp"].iloc[0] == pd.Timestamp("2024-01-01")


def test_unknown_symbol_gives_empty_frame():
    fake = FakeAlpacaData({})
    assert fetch_range(fake.exchange, "NOPE/USD", "1h", T0).empty


def test_retries_rate_limits_but_not_bad_requests(monkeypatch):
    fake = FakeAlpacaData({"BTC/USD": make_bars(T0, 10)}, fail_first=2)
    sleeps = []
    monkeypatch.setattr("tradingbot.data.time.sleep", sleeps.append)
    df = fetch_range(fake.exchange, "BTC/USD", "1h", T0)
    assert len(df) == 10 and sleeps == [1.0, 2.0]

    calls = []

    def bad():
        calls.append(1)
        raise ccxt.BadSymbol("no such market")
    with pytest.raises(ccxt.BadSymbol):
        with_retries(bad, sleep=lambda s: None)
    assert len(calls) == 1


def test_store_backfills_then_updates_incrementally(tmp_path, storage_format):
    bars = make_bars(T0, 300)
    now = (T0 + 200 * H + H // 2) / 1000      # halfway through bar 200
    fake = FakeAlpacaData({"BTC/USD": bars})
    store = BarStore(root=str(tmp_path))

    r1 = store.update(fake.exchange, "BTC/USD", "1h", since="2024-01-01", now=now)
    assert r1.total_rows == 200                # bar 200 is unfinished, so not stored
    assert store.load("BTC/USD", "1h")["timestamp"].iloc[-1] == pd.Timestamp(T0 + 199 * H, unit="ms")

    fake.requests.clear()
    r2 = store.update(fake.exchange, "BTC/USD", "1h", now=now + 50 * 3600)
    assert r2.new_rows == 50 and r2.total_rows == 250
    first_request_start = fake.exchange.parse8601(fake.requests[0]["start"])
    assert first_request_start == T0 + (199 - 3) * H   # only re-fetches the last few bars
    stored = store.load("BTC/USD", "1h")
    assert stored["timestamp"].is_unique and stored["timestamp"].is_monotonic_increasing
    assert store.symbols("1h") == ["BTC/USD"]


def test_store_skips_symbols_without_data(tmp_path, storage_format):
    store = BarStore(root=str(tmp_path))
    r = store.update(FakeAlpacaData({}).exchange, "NOPE/USD", "1h", since="2024-01-01", now=T0 / 1000 + 10 * 3600)
    assert r.total_rows == 0 and store.symbols("1h") == []


def _frame(rows):
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def test_quality_clean_data_is_ok():
    df = _frame(make_bars(T0, 500))
    q = check_bars(df, 3600, now=(T0 + 500 * H) / 1000)
    assert q["status"] == "ok" and q["coverage_pct"] == 100 and q["bad_prints"] == 0


def test_quality_detects_problems():
    rows = make_bars(T0, 500, skip=set(range(100, 160)))
    rows[10][2] = rows[10][4] * 0.5            # high below close
    q = check_bars(_frame(rows), 3600, now=(T0 + 600 * H) / 1000)
    assert q["status"] == "bad"
    assert q["ohlc_violations"] >= 1
    assert q["longest_gap_bars"] == 60
    assert "data hole of 60 bars" in q["issues"] and "old" in q["issues"]


def test_quality_counts_bad_prints_and_spots_delisting():
    rows = make_bars(T0, 500)
    rows[300][4] *= 30                          # one-bar 30x spike that snaps back (bad print)
    rows[300][2] = rows[300][4]
    q = check_bars(_frame(rows), 3600, now=(T0 + 500 * H) / 1000 + 30 * 86400)
    assert q["bad_prints"] == 1
    assert "fixed automatically" in q["issues"] and "delisted" in q["issues"]


def test_quality_flags_blank_prices():
    rows = make_bars(T0, 50)
    rows[5][4] = float("nan")
    q = check_bars(_frame(rows), 3600, now=(T0 + 50 * H) / 1000)
    assert q["status"] == "bad" and q["blank_prices"] == 1


def test_panel_aligns_symbols_and_fills_gaps(tmp_path, storage_format):
    store = BarStore(root=str(tmp_path))
    store.save("BTC/USD", "1h", _frame(make_bars(T0, 100, seed=1)))
    store.save("SOL/USD", "1h", _frame(make_bars(T0 + 50 * H, 50, seed=2, skip={10})))
    close = load_panel(store, ["BTC/USD", "SOL/USD"], "1h")
    assert len(close) == 100
    assert close["SOL/USD"].iloc[:50].isna().all()          # not trading yet: stays NaN
    assert close["SOL/USD"].iloc[50:].notna().all()         # missing bar filled with last price
    assert close["SOL/USD"].iloc[60] == close["SOL/USD"].iloc[59]
    vol = load_panel(store, ["BTC/USD", "SOL/USD"], "1h", field="volume")
    assert vol["SOL/USD"].iloc[60] == 0 and np.isnan(vol["SOL/USD"].iloc[0])

    bench = benchmarks(close)
    r = returns(close)
    assert bench["btc_buy_hold"].iloc[-1] == pytest.approx(close["BTC/USD"].iloc[-1] / close["BTC/USD"].iloc[0])
    assert bench["equal_weight"].iloc[10] == pytest.approx((1 + r["BTC/USD"].iloc[1:11]).prod())


def test_csv_round_trip_is_exact(tmp_path):
    df = _frame(make_bars(T0, 200, start_price=0.000012345678))   # tiny prices like SHIB/PEPE
    store = BarStore(root=str(tmp_path), use_parquet=False)
    store.save("PEPE/USD", "1h", df)
    assert store.path("PEPE/USD", "1h").endswith(".csv")
    back = store.load("PEPE/USD", "1h")
    assert (back["timestamp"] == df["timestamp"]).all()
    assert (back[["open", "high", "low", "close", "volume"]].values == df[["open", "high", "low", "close", "volume"]].values).all()


def test_switching_format_converts_and_keeps_one_copy(tmp_path):
    pytest.importorskip("pyarrow")
    import os
    df = _frame(make_bars(T0, 50))
    csv_store = BarStore(root=str(tmp_path), use_parquet=False)
    csv_store.save("BTC/USD", "1h", df)
    pq_store = BarStore(root=str(tmp_path), use_parquet=True)
    assert len(pq_store.load("BTC/USD", "1h")) == 50          # reads the CSV
    pq_store.save("BTC/USD", "1h", pq_store.load("BTC/USD", "1h"))
    assert os.path.exists(pq_store.path("BTC/USD", "1h"))
    assert not os.path.exists(csv_store.path("BTC/USD", "1h"))
    assert pq_store.symbols("1h") == ["BTC/USD"]

    with pytest.raises(RuntimeError, match="pyarrow"):
        csv_store.load("BTC/USD", "1h")                         # parquet file, no pyarrow: clear message


def test_panel_does_not_bridge_long_holes_or_revive_delisted_coins(tmp_path):
    store = BarStore(root=str(tmp_path), use_parquet=False)
    sol = make_bars(T0, 300, seed=3, skip=set(range(100, 200)))      # 100-bar outage
    sol[200:] = [[r[0], r[1] * 5, r[2] * 5, r[3] * 5, r[4] * 5, r[5]] for r in sol[200:]]  # price 5x after the hole
    store.save("SOL/USD", "1h", _frame(sol))
    store.save("BTC/USD", "1h", _frame(make_bars(T0, 400, seed=1)))
    store.save("MKR/USD", "1h", _frame(make_bars(T0, 150, seed=2)))   # delisted after 150 bars
    close = load_panel(store, ["SOL/USD", "BTC/USD", "MKR/USD"], "1h")
    r = returns(close)
    assert close["SOL/USD"].iloc[130:200].isna().all()                # long hole stays empty
    assert r["SOL/USD"].abs().max() < 1                               # no fake +400% across the hole
    assert close["MKR/USD"].iloc[150:].isna().all()                   # dead coin isn't "flat"
    assert close["SOL/USD"].iloc[300:].isna().all()                   # SOL's data ends at bar 300


def test_performance_skips_jumps_across_holes():
    from tradingbot.metrics import performance
    idx = pd.date_range("2024-01-01", periods=400, freq="D")
    rng = np.random.default_rng(0)
    eq = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.02, 400))), index=idx)
    holey = eq.copy()
    holey.iloc[200:300] = np.nan
    holey.iloc[300:] *= 10                                            # level jumps while data is missing
    smooth, jumpy = performance(eq), performance(holey)
    assert jumpy["ann_vol_pct"] < smooth["ann_vol_pct"] * 1.3          # vol isn't blown up by the jump
    assert jumpy["total_return_pct"] == pytest.approx(100 * (holey.iloc[-1] / holey.iloc[0] - 1))
