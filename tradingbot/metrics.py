"""Performance statistics shared by research, backtests and reports."""
import numpy as np
import pandas as pd


def periods_per_year(index) -> float:
    """Crypto trades 24/7, so a year has 365 days of bars."""
    if len(index) < 2:
        return 0.0
    step = pd.Series(index).diff().median().total_seconds()
    return 365 * 24 * 3600 / step if step else 0.0


def performance(equity: pd.Series) -> dict:
    """Standard stats for an equity curve (or any growth-of-$1 series) indexed by timestamp."""
    equity = equity.dropna()
    if len(equity) < 2:
        return {}
    ppy = periods_per_year(equity.index)
    r = equity.pct_change().dropna()
    years = len(r) / ppy if ppy else np.nan
    total = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (1 + total) ** (1 / years) - 1 if years and total > -1 else np.nan
    vol = r.std() * np.sqrt(ppy)
    dd = (equity / equity.cummax() - 1).min()
    return {
        "total_return_pct": 100 * total,
        "cagr_pct": 100 * cagr,
        "ann_vol_pct": 100 * vol,
        "sharpe": r.mean() / r.std() * np.sqrt(ppy) if r.std() > 0 else 0.0,
        "max_drawdown_pct": 100 * dd,
        "calmar": cagr / abs(dd) if dd < 0 else np.nan,  # yearly return per unit of worst drawdown
        "years": years,
    }


def performance_table(curves: pd.DataFrame) -> pd.DataFrame:
    """performance() for each column, one row per column."""
    return pd.DataFrame({col: performance(curves[col]) for col in curves.columns}).T
