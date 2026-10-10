"""End-to-end: the real CLI commands against a fake Alpaca, in a temp folder."""
import pandas as pd

import main
from tests.fakes import FakeAlpacaData, make_bars

T0 = int(pd.Timestamp("2024-01-01").timestamp() * 1000)


def run(argv):
    args = main.build_parser().parse_args(argv)
    args.fn(args)


def test_data_pipeline_end_to_end(tmp_path, monkeypatch, capsys, storage_format):
    monkeypatch.chdir(tmp_path)
    fake = FakeAlpacaData({
        "BTC/USD": make_bars(T0, 2000, seed=1, start_price=40000),
        "ETH/USD": make_bars(T0 + 500 * 3600_000, 1500, seed=2, start_price=2000, skip={700, 701}),
    })
    monkeypatch.setattr("tradingbot.data.research_exchange", lambda config: fake.exchange)

    run(["data", "update", "--symbols", "BTC/USD,ETH/USD,NOPE/USD", "--timeframe", "1h", "--since", "2024-01-01"])
    out = capsys.readouterr().out
    assert "BTC/USD    +  2000 bars" in out
    assert "NOPE/USD   no data" in out

    run(["data", "update", "--symbols", "BTC/USD", "--timeframe", "1h"])
    assert "+     0 bars" in capsys.readouterr().out          # nothing new: no duplicates

    run(["data", "check", "--timeframe", "1h"])
    out = capsys.readouterr().out
    assert "BTC/USD" in out and "ETH/USD" in out

    run(["data", "check", "--timeframe", "1h"])
    rows = [line.split() for line in capsys.readouterr().out.splitlines() if "/USD" in line]
    assert rows and all(r[1] in ("ok", "warn") for r in rows)

    from tradingbot.metrics import performance_table
    from tradingbot.panel import benchmarks, load_panel
    from tradingbot.store import BarStore
    close = load_panel(BarStore(), ["BTC/USD", "ETH/USD"], "1h")
    assert close["BTC/USD"].notna().all()
    table = performance_table(benchmarks(close))
    assert table.loc["btc_buy_hold", "ann_vol_pct"] > 1     # real numbers, not zeros/NaN

    run(["data", "benchmarks", "--timeframe", "1h"])
    out = capsys.readouterr().out
    assert "equal_weight" in out and "NaN" not in out.split("equal_weight =")[0].replace("calmar", "")

    run(["data", "inspect", "ETH/USD", "--timeframe", "1h"])
    out = capsys.readouterr().out
    assert "Data holes" in out and "Fixed by cleaning" in out and "Biggest one-bar moves" in out

    monkeypatch.setattr(main, "Config", lambda: _config("BTC/USD"))
    run(["backtest"])
    out = capsys.readouterr().out
    assert "2000 1h bars of BTC/USD" in out and "Buy & hold return" in out


def _config(symbol):
    from tradingbot.config import Config
    c = Config()
    c.exchange_id, c.symbol, c.timeframe = "alpaca", symbol, "1h"
    return c
