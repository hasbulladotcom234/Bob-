"""Interactive charts, written as web pages you open in your browser.

Each page is a plain HTML file in reports/ that draws its charts with
Plotly (the library ships inside the plotly Python package, so it works
offline). Hover any chart for exact values, drag to zoom, double-click to
reset, click legend entries to hide/show lines.

Colours follow a validated palette (readable for colour-blind viewers, with
its own steps for dark mode): one hue per job, never decoration.
"""
import html
import json
import os
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd

REPORTS = "reports"

# Colour roles. Charts refer to roles ("@series-1"), the page swaps in the light or dark value.
THEMES = {
    "light": {"surface": "#fcfcfb", "page": "#f9f9f7", "text": "#0b0b0b", "text2": "#52514e",
              "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
              "series-1": "#2a78d6", "series-2": "#eb6834",
              "pos": "#2a78d6", "neg": "#e34948", "context": "rgba(137,135,129,0.35)",
              "ramp-1": "#86b6ef", "ramp-2": "#2a78d6", "ramp-3": "#104281",
              "critical": "#d03b3b"},
    "dark": {"surface": "#1a1a19", "page": "#0d0d0d", "text": "#f0efec", "text2": "#c3c2b7",
             "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835",
             "series-1": "#3987e5", "series-2": "#d95926",
             "pos": "#3987e5", "neg": "#e66767", "context": "rgba(137,135,129,0.35)",
             "ramp-1": "#184f95", "ramp-2": "#3987e5", "ramp-3": "#9ec5f4",
             "critical": "#d03b3b"},
}

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<script src="plotly.min.js"></script>
<style>
:root { color-scheme: light; --page:#f9f9f7; --surface:#fcfcfb; --text:#0b0b0b; --text2:#52514e;
        --muted:#898781; --border:rgba(11,11,11,0.10); --grid:#e1e0d9; }
@media (prefers-color-scheme: dark) { :root:where(:not([data-theme="light"])) {
  color-scheme: dark; --page:#0d0d0d; --surface:#1a1a19; --text:#f0efec; --text2:#c3c2b7;
  --border:rgba(255,255,255,0.10); --grid:#2c2c2a; } }
:root[data-theme="dark"] { color-scheme: dark; --page:#0d0d0d; --surface:#1a1a19; --text:#f0efec;
  --text2:#c3c2b7; --border:rgba(255,255,255,0.10); --grid:#2c2c2a; }
body { margin:0; background:var(--page); color:var(--text);
       font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
main { max-width:1100px; margin:0 auto; padding:24px 16px 64px; }
h1 { font-size:24px; margin:0 0 4px; } h2 { font-size:18px; margin:0 0 4px; }
.sub { color:var(--text2); margin:0 0 24px; }
section { background:var(--surface); border:1px solid var(--border); border-radius:12px;
          padding:20px; margin:0 0 20px; }
section p { color:var(--text2); margin:0 0 12px; max-width:75ch; }
.chart { width:100%; }
.warn { border-color:#d03b3b; } .warn h2::before { content:"\\26A0  "; }
table { border-collapse:collapse; font-variant-numeric:tabular-nums; font-size:13px; width:100%; }
th, td { text-align:right; padding:4px 10px; border-bottom:1px solid var(--grid); white-space:nowrap; }
th:first-child, td:first-child { text-align:left; } th { color:var(--text2); font-weight:600; }
.scroll { overflow-x:auto; }
.toggle { float:right; font:inherit; color:var(--text2); background:none; border:1px solid var(--border);
          border-radius:8px; padding:4px 10px; cursor:pointer; }
</style></head>
<body><main>
<button class="toggle" onclick="flip()">Light / dark</button>
<h1>__TITLE__</h1><p class="sub">__SUB__</p>
__SECTIONS__
</main>
<script>
const THEMES = __THEMES__;
const FIGS = __FIGS__;
function mode() {
  const t = document.documentElement.dataset.theme;
  if (t) return t;
  return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}
function paint(x, th) {           // swap "@role" strings for this theme's colours
  if (typeof x === "string" && x[0] === "@") return th[x.slice(1)] || x;
  if (Array.isArray(x)) return x.map(v => paint(v, th));
  if (x && typeof x === "object") { const o = {}; for (const k in x) o[k] = paint(x[k], th); return o; }
  return x;
}
function axis(a, th) {
  return Object.assign({gridcolor: th.grid, linecolor: th.axis, zerolinecolor: th.axis, tickcolor: th.axis,
    showline: true, ticks: "outside", ticklen: 4, automargin: true,
    showspikes: true, spikemode: "across", spikethickness: 1, spikecolor: th.muted, spikedash: "solid"}, a || {});
}
function render() {
  const th = THEMES[mode()];
  for (const f of FIGS) {
    const lay = paint(f.layout || {}, th);
    const layout = Object.assign({
      paper_bgcolor: th.surface, plot_bgcolor: th.surface, height: f.height || 380,
      font: {family: 'system-ui,-apple-system,"Segoe UI",sans-serif', size: 13, color: th.text2},
      margin: {l: 64, r: 24, t: 16, b: 48}, hovermode: "x unified", barcornerradius: 4, bargap: 0.35,
      legend: {orientation: "h", y: 1.12, x: 0, font: {color: th.text2}},
      hoverlabel: {bgcolor: th.surface, bordercolor: th.axis, font: {color: th.text}},
    }, lay);
    layout.xaxis = axis(lay.xaxis, th); layout.yaxis = axis(lay.yaxis, th);
    Plotly.react(f.id, paint(f.data, th), layout, {displaylogo: false, responsive: true});
  }
}
function flip() {
  document.documentElement.dataset.theme = mode() === "dark" ? "light" : "dark"; render();
}
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
render();
</script></body></html>
"""


def _clean(values):
    """Lists for JSON: NaN/inf become gaps (None)."""
    out = []
    for v in values:
        if isinstance(v, (pd.Timestamp, np.datetime64)):
            out.append(pd.Timestamp(v).isoformat())
        elif v is None or (isinstance(v, float) and not np.isfinite(v)):
            out.append(None)
        elif isinstance(v, (np.floating, np.integer)):
            out.append(None if not np.isfinite(v) else v.item())
        else:
            out.append(v)
    return out


def log_ticks(prefix="", suffix=""):
    """Log-axis labels written in full ($0.50, $1, $2, $5, $10, $20 ...) instead of Plotly's '2' for 20."""
    vals = [m * 10.0 ** e for e in range(-4, 7) for m in (1, 2, 5)]
    def label(v):
        return f"{prefix}{v:,.0f}{suffix}" if v >= 1 else f"{prefix}{v:g}{suffix}"
    return {"type": "log", "tickmode": "array", "tickvals": vals, "ticktext": [label(v) for v in vals]}


def line(x, y, name, color, width=2, **extra):
    return {"type": "scatter", "mode": "lines", "x": _clean(x), "y": _clean(y), "name": name,
            "line": {"color": color, "width": width}, **extra}


def bars(x, y, colors, name="", **extra):
    return {"type": "bar", "x": _clean(x), "y": _clean(y), "name": name,
            "marker": {"color": colors, "line": {"width": 0}}, **extra}


def sign_colors(values):
    """Diverging colour per bar: positive one hue, negative the opposite."""
    return ["@pos" if (v is not None and v >= 0) else "@neg" for v in _clean(values)]


def table_html(df: pd.DataFrame, fmt="{:.2f}") -> str:
    """fmt: one format for every number, or {column: format} (others use {:.2f})."""
    def cell(v, col):
        f = fmt.get(col, "{:.2f}") if isinstance(fmt, dict) else fmt
        if isinstance(v, (float, np.floating)):
            return "" if not np.isfinite(v) else f.format(v)
        if isinstance(v, pd.Timestamp):
            return f"{v:%Y-%m-%d}"
        return html.escape(str(v))
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in [df.index.name or ""] + list(df.columns))
    rows = "".join("<tr>" + f"<td>{html.escape(str(i))}</td>" + "".join(f"<td>{cell(v, c)}</td>" for v, c in zip(row, df.columns)) + "</tr>"
                   for i, row in zip(df.index, df.itertuples(index=False)))
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>'


def write_page(filename: str, title: str, subtitle: str, sections: list, folder: str = REPORTS) -> str:
    """sections: dicts with heading, text, and optionally figure (data/layout/height), table (html), warn."""
    os.makedirs(folder, exist_ok=True)
    js = os.path.join(folder, "plotly.min.js")
    if not os.path.exists(js):
        import plotly.offline
        with open(js, "w", encoding="utf-8") as f:
            f.write(plotly.offline.get_plotlyjs())
    parts, figs = [], []
    for i, s in enumerate(sections):
        body = f"<h2>{html.escape(s['heading'])}</h2>"
        body += "".join(f"<p>{html.escape(p)}</p>" for p in s.get("text", []))
        if s.get("figure"):
            fid = f"fig{i}"
            figs.append({"id": fid, **s["figure"]})
            body += f'<div class="chart" id="{fid}"></div>'
        if s.get("table"):
            body += s["table"]
        parts.append(f'<section class="{"warn" if s.get("warn") else ""}">{body}</section>')
    page = (PAGE.replace("__TITLE__", html.escape(title)).replace("__SUB__", html.escape(subtitle))
            .replace("__SECTIONS__", "\n".join(parts)).replace("__THEMES__", json.dumps(THEMES))
            .replace("__FIGS__", json.dumps(figs)))
    path = os.path.join(folder, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    return path


def open_in_browser(path: str) -> None:
    webbrowser.open(Path(path).resolve().as_uri())


# ---- pages -----------------------------------------------------------------

def research_page(r, signal: pd.DataFrame, holdout_start=None, folder: str = REPORTS) -> str:
    """Charts for one research run (a ResearchResult) plus the scores as a CSV."""
    os.makedirs(folder, exist_ok=True)
    scores_csv = os.path.join(folder, f"{r.name}_scores.csv")
    signal.round(6).to_csv(scores_csv, index_label="date")
    h = r.rebalance
    sections = []
    if r.uses_future:
        sections.append({"heading": "This signal uses future data", "warn": True,
                         "text": ["It looks at prices after the day it scores (e.g. .shift(-1)). "
                                  "Every chart below is fake until that's fixed."]})

    ic = r.ic_by_horizon
    se = (ic["mean_ic"] / ic["t_stat"]).abs()
    sections.append({
        "heading": "1. Does a higher score come before a higher return?",
        "text": ["Each bar is the average information coefficient (IC): every day, how well the coins' ranking "
                 "by your score matched their ranking by return over the next N days. Above zero = higher scores "
                 "did better; below zero = they did worse. The thin whiskers show +/-2 standard errors: "
                 "if a whisker crosses zero, that horizon could be luck."],
        "figure": {"height": 320, "data": [bars([f"{d} day{'s' if d > 1 else ''}" for d in ic.index], ic["mean_ic"],
                                                 sign_colors(ic["mean_ic"]), "mean IC",
                                                 error_y={"type": "data", "array": _clean(2 * se), "color": "@muted",
                                                          "thickness": 1, "width": 6})],
                   "layout": {"hovermode": "closest", "showlegend": False,
                              "yaxis": {"title": {"text": "mean IC"}, "zeroline": True}, "xaxis": {"showspikes": False}}},
        "table": table_html(ic.rename(columns={"mean_ic": "mean IC", "t_stat": "t-stat",
                                               "pct_positive": "% days positive"}), {"mean IC": "{:.3f}", "t-stat": "{:.2f}", "% days positive": "{:.0f}%", "days": "{:.0f}"}),
    })

    b = r.buckets
    sections.append({
        "heading": "2. The next day, coin by score bucket",
        "text": ["Every day the coins are split into 5 buckets by score (1 = lowest scores, 5 = highest). Each bar "
                 "is how much better or worse than that day's average coin each bucket did the NEXT day, averaged "
                 "over every day. A working idea gives a staircase; a flat or jagged pattern means no reliable link."],
        "figure": {"height": 300, "data": [bars([f"bucket {int(i)}" for i in b.index], b["avg_excess_pct"],
                                                 sign_colors(b["avg_excess_pct"]), "next day vs average coin (%)",
                                                 error_y={"type": "data", "array": _clean(2 * b["se_pct"]),
                                                          "color": "@muted", "thickness": 1, "width": 6})],
                   "layout": {"hovermode": "closest", "showlegend": False, "xaxis": {"showspikes": False},
                              "yaxis": {"title": {"text": "% vs average coin, next day"}, "zeroline": True}}},
    })

    roll = r.ic_series.rolling(90, min_periods=30).mean()
    sections.append({
        "heading": f"3. Is it steady, or did one period carry it? ({h}-day IC, 90-day average)",
        "text": ["The line is the IC averaged over the previous 90 days. A real effect stays mostly on one side of "
                 "zero; an effect that only exists in one stretch (say early 2021) is fragile."],
        "figure": {"height": 300, "data": [line(roll.index, roll.values, "90-day average IC", "@series-1")],
                   "layout": {"showlegend": False, "yaxis": {"title": {"text": "IC"}, "zeroline": True}}},
        "table": table_html(r.ic_by_year[["mean_ic", "t_stat", "pct_positive"]].rename(
            columns={"mean_ic": "mean IC", "t_stat": "t-stat", "pct_positive": "% days positive"}), {"mean IC": "{:.3f}", "t-stat": "{:.2f}", "% days positive": "{:.0f}%", "days": "{:.0f}"}),
    })

    daily = r.groups.daily
    group_cols = [c for c in daily.columns if c.startswith("group_")]
    curves = (1 + daily[group_cols + ["market"]].fillna(0)).cumprod()
    n = len(group_cols)
    ramp = ["@ramp-1", "@ramp-2", "@ramp-3"] if n == 3 else ["@series-1"] * n
    names = {c: ("lowest scores" if i == 0 else "highest scores" if i == n - 1 else "middle")
             for i, c in enumerate(group_cols)}
    traces = [line(curves.index, curves[c], f"{c.replace('_', ' ')} ({names[c]})", ramp[i])
              for i, c in enumerate(group_cols)]
    traces.append(line(curves.index, curves["market"], "every coin (market)", "@series-2"))
    sections.append({
        "heading": f"4. What $1 became in each score group (re-sorted every {h} days, before fees)",
        "text": ["Each line holds one third of the coins, chosen by score, in equal amounts. If the idea works, the "
                 "highest-score line ends on top and the lowest at the bottom. The orange line holds every coin. "
                 "Log scale: equal vertical distances are equal percentage moves."],
        "figure": {"height": 420, "data": traces, "layout": {"yaxis": {**log_ticks("$"), "title": {"text": "growth of $1"}}}},
        "table": table_html(r.groups.stats[["cagr_pct", "ann_vol_pct", "sharpe", "max_drawdown_pct"]].rename(
            columns={"cagr_pct": "yearly return %", "ann_vol_pct": "volatility %", "max_drawdown_pct": "worst fall %"})),
    })

    rel = (1 + daily[["top_vs_market", "top_vs_bottom"]].fillna(0)).cumprod()
    sections.append({
        "heading": "5. Top group against the market and the bottom group (after fees)",
        "text": [f"Above 1 = the top group's money grew faster. Fees from swapping holdings: about "
                 f"{r.groups.costs_per_year_pct:.1f}% a year ({100 * r.groups.turnover:.0f}% of the top group "
                 f"replaced every {h} days)."],
        "figure": {"height": 340, "data": [line(rel.index, rel["top_vs_market"], "top vs market", "@series-1"),
                                           line(rel.index, rel["top_vs_bottom"], "top vs bottom", "@series-2")],
                   "layout": {"yaxis": {**log_ticks(suffix="x"), "title": {"text": "relative growth"}}}},
    })

    if r.placebo and r.placebo.get("runs") is not None:
        p = r.placebo
        sections.append({
            "heading": "6. Luck check: your idea against scrambled versions of itself",
            "text": ["Each bar counts scrambled runs: every coin was given another coin's scores, which keeps "
                     "everything about the signal except its link to the right coin. The red line is your idea. "
                     f"Scrambled versions did at least as well {100 * p['share_as_good']:.0f}% of the time: "
                     "the further the red line sits outside the pile, the less likely it's luck."],
            "figure": {"height": 300, "data": [
                {"type": "histogram", "x": _clean(p["runs"]), "nbinsx": 30, "name": "scrambled runs",
                 "marker": {"color": "@series-1", "line": {"width": 0}}}],
                "layout": {"hovermode": "closest", "showlegend": False, "bargap": 0.08,
                           "xaxis": {"title": {"text": "top minus bottom, % a year (before fees)"}, "showspikes": False},
                           "yaxis": {"title": {"text": "scrambled runs"}},
                           "shapes": [{"type": "line", "xref": "x", "yref": "paper", "x0": p["actual_pct"],
                                       "x1": p["actual_pct"], "y0": 0, "y1": 1,
                                       "line": {"color": "@critical", "width": 2}}],
                           "annotations": [{"x": p["actual_pct"], "y": 1, "xref": "x", "yref": "paper",
                                            "text": f"your idea: {p['actual_pct']:+.1f}%", "showarrow": False,
                                            "yanchor": "bottom", "font": {"color": "@text"}}]}},
        })

    if len(r.ic_by_coin):
        c = r.ic_by_coin.sort_values("ic")
        sections.append({
            "heading": f"7. Coin by coin ({h}-day IC through time)",
            "text": ["For each coin on its own: did a higher score than usual come before a better return than "
                     "usual? Mostly one colour = the effect is broad; a few extreme coins = it may be driven by them."],
            "figure": {"height": max(280, 22 * len(c) + 60),
                       "data": [bars(c["ic"], list(c.index), sign_colors(c["ic"]), "IC", orientation="h")],
                       "layout": {"hovermode": "closest", "showlegend": False, "margin": {"l": 90},
                                  "xaxis": {"title": {"text": "IC"}, "zeroline": True, "showspikes": False},
                                  "yaxis": {"showspikes": False, "dtick": 1}}},
        })

    sections.append({"heading": "The raw numbers",
                     "text": [f"Every coin's score on every day is in {scores_csv} (open it in Excel). Rows are days, "
                              f"columns are coins."]})
    sub = (f"{r.description}  |  {r.start:%Y-%m-%d} to {r.end:%Y-%m-%d}, about {r.coins_per_day:.0f} coins a day"
           + (f"  |  locked holdout from {pd.Timestamp(holdout_start):%Y-%m-%d} not used" if holdout_start else ""))
    return write_page(f"research_{r.name}.html", f"Research: {r.name}", sub, sections, folder)


def data_page(close: pd.DataFrame, quality: pd.DataFrame = None, holdout_start=None, folder: str = REPORTS) -> str:
    """Every coin's growth, the BTC and basket benchmarks, drawdowns and data quality."""
    from .metrics import performance_table
    from .panel import benchmarks
    growth = close / close.bfill().iloc[0]
    bench = benchmarks(close)
    traces = [line(growth.index, growth[c], c, "@context", width=1, hovertemplate=f"{c}: %{{y:.2f}}<extra></extra>",
                   showlegend=False) for c in growth.columns if c != "BTC/USD"]
    if "btc_buy_hold" in bench:
        traces.append(line(bench.index, bench["btc_buy_hold"], "Bitcoin", "@series-1"))
    traces.append(line(bench.index, bench["equal_weight"], "every coin, equal amounts", "@series-2"))
    dd = bench / bench.cummax() - 1
    table = performance_table(pd.concat([bench, growth], axis=1)).sort_values("sharpe", ascending=False)
    sections = [
        {"heading": "What $1 became in each coin",
         "text": ["Grey lines are individual coins (hover to see which), each starting at $1 on its first day of data. "
                  "Blue holds only Bitcoin; orange holds every coin in equal amounts, rebalanced daily. "
                  "Log scale: equal vertical distances are equal percentage moves. Drag to zoom, double-click to reset."],
         "figure": {"height": 480, "data": traces, "layout": {"hovermode": "closest",
                                                             "yaxis": {**log_ticks("$"), "title": {"text": "growth of $1"}}}}},
        {"heading": "How far below its peak (drawdown)",
         "text": ["0% = at a new high. -80% = 80% below the best level so far. This is what holding through crypto "
                  "actually felt like."],
         "figure": {"height": 320, "data": [line(dd.index, 100 * dd[c], label, color) for c, label, color in
                                             [("btc_buy_hold", "Bitcoin", "@series-1"),
                                              ("equal_weight", "every coin", "@series-2")] if c in dd],
                    "layout": {"yaxis": {"title": {"text": "% below peak"}, "ticksuffix": "%"}}}},
        {"heading": "Scoreboard (buy and hold, no fees)",
         "text": ["Sorted by Sharpe ratio: return per unit of risk. Coins that started later are measured from their "
                  "first day."],
         "table": table_html(table[["total_return_pct", "cagr_pct", "ann_vol_pct", "sharpe", "max_drawdown_pct",
                                    "years"]].rename(columns={"total_return_pct": "total %", "cagr_pct": "yearly %",
                                                              "ann_vol_pct": "volatility %",
                                                              "max_drawdown_pct": "worst fall %"}))},
    ]
    if quality is not None and len(quality):
        q = quality.set_index("symbol")[["status", "rows", "coverage_pct", "bad_prints", "bad_wicks",
                                         "kept_by_market", "issues"]]
        sections.append({"heading": "Data quality", "text": ["From the raw downloaded data, before cleaning."],
                         "table": table_html(q, {"coverage_pct": "{:.1f}%"})})
    sub = f"Daily closes, cleaned, {close.index[0]:%Y-%m-%d} to {close.index[-1]:%Y-%m-%d}" + (
        f" (stops before the locked holdout, {pd.Timestamp(holdout_start):%Y-%m-%d})" if holdout_start else "")
    return write_page("data_explorer.html", "Your market data", sub, sections, folder)


def coin_page(symbol: str, raw: pd.DataFrame, clean: pd.DataFrame, report: dict, holdout_start=None,
              folder: str = REPORTS) -> str:
    """One coin's hourly prices, raw against cleaned, with every cleaning decision marked."""
    traces = [
        line(raw["timestamp"], raw["close"], "raw price (as downloaded)", "@context", width=1, type="scattergl"),
        line(clean["timestamp"], clean["close"], "cleaned price (used in research)", "@series-1", width=1.5,
             type="scattergl"),
    ]
    ch = report.get("changes", pd.DataFrame())
    if len(ch):
        traces.append({"type": "scatter", "mode": "markers", "x": _clean(ch["timestamp"]), "y": _clean(ch["raw"]),
                       "name": "removed / fixed bad price", "text": list(ch["kind"]),
                       "hovertemplate": "%{text}: %{y:.6g}<extra></extra>",
                       "marker": {"color": "@critical", "symbol": "x", "size": 9}})
    kept = report.get("kept", pd.DataFrame())
    if len(kept):
        traces.append({"type": "scatter", "mode": "markers", "x": _clean(kept["timestamp"]),
                       "y": _clean(kept["price"]), "name": "kept: the market moved too", "text": list(kept["kind"]),
                       "hovertemplate": "kept %{text}: %{y:.6g}<extra></extra>",
                       "marker": {"color": "@series-2", "symbol": "circle-open", "size": 10, "line": {"width": 2}}})
    sections = [{
        "heading": "Hourly price, raw vs cleaned",
        "text": ["Grey = the price exactly as downloaded. Blue = after cleaning. Red crosses = prices the cleaner "
                 "removed or fixed (hover for the kind). Orange circles = sudden moves it KEPT because Bitcoin moved "
                 "the same way that hour. Zoom into any cross (drag a box) and judge for yourself: does it look like "
                 "a glitch or a real move?"],
        "figure": {"height": 520, "data": traces, "layout": {"hovermode": "closest",
                                                            "yaxis": {"type": "log", "dtick": "D2", "exponentformat": "power",
                                                                      "title": {"text": "price (log scale)"}}}},
    }]
    if len(ch):
        sections.append({"heading": f"Every fix ({len(ch)}), largest first", "table": table_html(
            ch.set_index("timestamp")[["kind", "raw", "fixed_to", "level_before", "level_after", "size_pct"]],
            "{:.6g}")})
    sub = (f"{len(raw)} hourly bars, {raw['timestamp'].iloc[0]:%Y-%m-%d} to {raw['timestamp'].iloc[-1]:%Y-%m-%d}"
           + (f" (stops before the locked holdout, {pd.Timestamp(holdout_start):%Y-%m-%d})" if holdout_start else ""))
    return write_page(f"coin_{symbol.replace('/', '-')}.html", f"{symbol}: prices and cleaning decisions", sub,
                      sections, folder)
