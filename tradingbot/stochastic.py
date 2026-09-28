"""Stochastic-calculus tools for the strategy: estimating the local drift
and volatility of log price, and pricing a stop-loss / take-profit bracket
as a two-barrier first-passage problem.

Model: over a short window, log price is treated as Brownian motion with
drift,

    d(log S_t) = nu dt + sigma dW_t

(the Ito form of geometric Brownian motion dS = mu S dt + sigma S dW, with
nu = mu - sigma^2 / 2). Time is measured in bars, so nu and sigma are per
bar.

A long trade with take-profit at +tp and stop-loss at -sl is exited the
first time log price leaves the interval (-b, a), where a = log(1 + tp) and
b = -log(1 - sl). Solving the generator equation
nu f' + (sigma^2 / 2) f'' = 0 with f(a) = 1, f(-b) = 0 gives the
probability of hitting the take-profit first:

    P(take) = (1 - exp(k b)) / (exp(-k a) - exp(k b)),   k = 2 nu / sigma^2

which reduces to b / (a + b) when nu = 0. The expected exit time is
(a P - b (1 - P)) / nu, or a b / sigma^2 when nu = 0.

None of this predicts price. It turns "the recent trend looks up" into a
number (the chance the bracket pays off, and the expected return after
costs) so trades whose odds don't cover their costs can be skipped. The
drift estimate is noisy -- drift is the hardest quantity in finance to
estimate -- which is why it is shrunk toward zero before use.
"""
import numpy as np
import pandas as pd


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close).diff()


def ewma_drift_vol(close: pd.Series, lookback: int, shrinkage: float = 1.0):
    """Exponentially weighted per-bar estimates of log drift (nu) and
    volatility (sigma). `shrinkage` in [0, 1] scales the drift toward zero
    (1.0 = raw estimate, 0.0 = assume no drift)."""
    r = log_returns(close)
    nu = r.ewm(span=lookback, min_periods=max(2, lookback // 4)).mean() * shrinkage
    sigma = r.ewm(span=lookback, min_periods=max(2, lookback // 4)).std()
    return nu, sigma


def _barriers(stop_loss_pct: float, take_profit_pct: float):
    a = np.log1p(take_profit_pct)
    b = -np.log1p(-stop_loss_pct)
    return a, b


def take_profit_probability(nu, sigma, stop_loss_pct: float, take_profit_pct: float):
    """Probability that drifted Brownian motion in log price reaches the
    take-profit barrier before the stop-loss barrier. Works on scalars or
    arrays; returns NaN where sigma is missing or non-positive."""
    nu = np.asarray(nu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    a, b = _barriers(stop_loss_pct, take_profit_pct)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        k = 2.0 * nu / sigma ** 2
        # Two algebraically equal forms, each numerically stable on one side of k = 0.
        pos = np.expm1(-k * b) / np.expm1(-k * (a + b))
        neg = np.expm1(k * b) * np.exp(k * a) / np.expm1(k * (a + b))
        p = np.where(k > 0, pos, neg)
        p = np.where(np.abs(k) * (a + b) < 1e-9, b / (a + b), p)
        p = np.where(np.isfinite(sigma) & (sigma > 0), p, np.nan)
    p = np.clip(p, 0.0, 1.0)
    return p if p.ndim else float(p)


def expected_exit_bars(nu, sigma, stop_loss_pct: float, take_profit_pct: float):
    """Expected number of bars until either barrier is hit."""
    nu = np.asarray(nu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    a, b = _barriers(stop_loss_pct, take_profit_pct)
    p = np.asarray(take_profit_probability(nu, sigma, stop_loss_pct, take_profit_pct))
    with np.errstate(divide="ignore", invalid="ignore"):
        drifted = (a * p - b * (1 - p)) / nu
        driftless = a * b / sigma ** 2
        t = np.where(np.abs(nu) < 1e-12, driftless, drifted)
    return t if t.ndim else float(t)


def bracket_edge(p_take, stop_loss_pct: float, take_profit_pct: float, round_trip_cost: float = 0.0):
    """Expected fractional return of a trade held until the stop or target
    is hit, net of round-trip fees and slippage."""
    p_take = np.asarray(p_take, dtype=float)
    edge = p_take * take_profit_pct - (1 - p_take) * stop_loss_pct - round_trip_cost
    return edge if edge.ndim else float(edge)
