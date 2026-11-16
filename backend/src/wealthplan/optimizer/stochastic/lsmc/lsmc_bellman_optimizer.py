"""LSMC Bellman optimizer for consumption and equity share.

Notation (monthly steps t = 0..T-1):

* W_t   liquid wealth at the start of month t, on the wealth grid w_0 = 0 < ... < w_{N-1}.
* y_t   deterministic net cashflow of month t (salary / pension minus fixed expenses).
* c_t   consumption (control 1); X_t = W_t + y_t - c_t >= 0 is the invested amount.
* a_t   equity share of X_t in [0, a_max] (control 2); the rest earns the savings rate.
* R_t   equity gross return of month t from the simulated GBM paths, R_f = savings gross return.
* p_t   probability to survive month t (Gompertz), beta the monthly discount factor.

Bellman equation (u = CRRA utility):

    V_t(w) = max_{c, a}  u(c) + beta * p_t * E_t[ V_{t+1}( X * (a R_t + (1 - a) R_f) ) ]
    V_{T-1}(w) = u(w + y_{T-1})        (consume everything in the last month)

LSMC step: for every regression path s, post-decision wealth grid point x_j and equity share
a_k the realised value Y = p_t V_{t+1}(x_j G_{k,s}, s) is regressed on a polynomial in the
path's equity level S_t (``WealthRegressor``); the fitted values are the continuation values.
The equity share is chosen per (x_j, s) as the best a_k, the consumption by a search over x.

The forward simulation re-evaluates the regression on fresh out-of-sample paths, so the
reported figures are not biased by the regression paths.
"""

import datetime as dt
import logging
import time
from typing import Any, Callable, Dict, List, Optional

import numba as nb
import numpy as np
import pandas as pd
from tqdm import tqdm

from result_cache.result_cache import ResultCache
from wealthplan.cashflows.cashflow_base import CashflowBase
from wealthplan.optimizer.math_tools.penality_functions import square_penalty
from wealthplan.optimizer.math_tools.utility_functions import crra_utility, crra_utility_scalar
from wealthplan.optimizer.stochastic.market_model.gbm_returns import GBM
from wealthplan.optimizer.stochastic.survival_process.survival_model import SurvivalModel
from wealthplan.optimizer.stochastic.survival_process.survival_process import SurvivalProcess
from wealthplan.optimizer.stochastic.wealth_regression.wealth_regression import WealthRegressor

logger = logging.getLogger(__name__)

ProgressFn = Callable[[str, float], None]


# ---------------------------------------------------------------------------- numba kernels


@nb.njit(parallel=True, cache=True)
def project_realised_values(grid: np.ndarray, v_next: np.ndarray, gross: np.ndarray, q: np.ndarray,
                            scale: float):
    """
    LSMC regression step without materialising the realised values.

    The realised values Y[s, j, k] = scale * V_{t+1}(x_j * G_{k,s}, s) are projected on the
    orthonormal regression basis q (n_sims, D) on the fly.

    Parameters
    ----------
    grid : np.ndarray, shape (n_w,)
        Wealth grid (ascending, grid[0] = 0).
    v_next : np.ndarray, shape (n_sims, n_w)
        Value function of the next month per regression path.
    gross : np.ndarray, shape (n_a, n_sims)
        Gross portfolio returns a_k R_s + (1 - a_k) R_f (> 0).
    q : np.ndarray, shape (n_sims, D)
        Orthonormal basis of the regression design.
    scale : float
        Factor applied to all values (survival probability p_t).

    Returns
    -------
    proj : np.ndarray, shape (D, n_w, n_a)
        q.T @ Y.
    sum_y, sum_y2 : np.ndarray, shape (n_w, n_a)
        Column sums of Y and Y**2 (for R²).
    """
    n_a, n_sims = gross.shape
    n_w = grid.shape[0]
    d = q.shape[1]
    proj = np.zeros((d, n_w, n_a))
    sum_y = np.zeros((n_w, n_a))
    sum_y2 = np.zeros((n_w, n_a))
    for k in nb.prange(n_a):
        # thread-local accumulators (writing into the shared arrays inside the loop causes
        # false sharing between threads)
        acc = np.zeros((n_w, d))
        acc_y = np.zeros(n_w)
        acc_y2 = np.zeros(n_w)
        for s in range(n_sims):
            v = v_next[s]
            g = gross[k, s]
            qs = q[s]
            i = 0
            for j in range(n_w):  # x_j * g ascending in j: walk the bracket forward
                y = grid[j] * g
                while i < n_w - 2 and grid[i + 1] < y:
                    i += 1
                w = (y - grid[i]) / (grid[i + 1] - grid[i])
                val = scale * (v[i] + w * (v[i + 1] - v[i]))
                acc_y[j] += val
                acc_y2[j] += val * val
                for b in range(d):
                    acc[j, b] += qs[b] * val
        for j in range(n_w):
            sum_y[j, k] = acc_y[j]
            sum_y2[j, k] = acc_y2[j]
            for b in range(d):
                proj[b, j, k] = acc[j, b]
    return proj, sum_y, sum_y2


@nb.njit(cache=True)
def _max_share(x: float, reserve: float) -> float:
    """Largest equity share that keeps at least ``reserve`` in the savings account."""
    if reserve <= 0.0:
        return 1.0
    if x <= reserve:
        return 0.0
    return 1.0 - reserve / x


@nb.njit(parallel=True, cache=True)
def best_continuation(q: np.ndarray, proj: np.ndarray, grid: np.ndarray, shares: np.ndarray,
                      reserve: float) -> np.ndarray:
    """
    Fitted continuation value per path and grid point, maximised over the admissible equity
    shares (those keeping the emergency reserve in the savings account; 0 % is always allowed).

    Parameters
    ----------
    q : np.ndarray, shape (n_sims, D)
    proj : np.ndarray, shape (D, n_w, n_a)
    grid, shares : wealth grid (n_w,) and equity share grid (n_a,)
    reserve : float
        Minimum amount in the savings account (0: no constraint).

    Returns
    -------
    np.ndarray, shape (n_sims, n_w)
        max_k sum_b q[s, b] * proj[b, j, k] over admissible k.
    """
    n_sims, d = q.shape
    _, n_w, n_a = proj.shape
    out = np.empty((n_sims, n_w))
    for s in nb.prange(n_sims):
        for j in range(n_w):
            best = -np.inf
            cap = _max_share(grid[j], reserve) + 1e-12
            for k in range(n_a):
                if k > 0 and shares[k] > cap:
                    # the cap itself lies between shares[k-1] and shares[k]: interpolate linearly
                    if shares[k - 1] < cap - 1e-12:
                        v0 = 0.0
                        v1 = 0.0
                        for b in range(d):
                            v0 += q[s, b] * proj[b, j, k - 1]
                            v1 += q[s, b] * proj[b, j, k]
                        w = (cap - shares[k - 1]) / (shares[k] - shares[k - 1])
                        vc = v0 + w * (v1 - v0)
                        if vc > best:
                            best = vc
                    break
                v = 0.0
                for b in range(d):
                    v += q[s, b] * proj[b, j, k]
                if v > best:
                    best = v
            out[s, j] = best
    return out


@nb.njit(cache=True)
def _segment_optimum(m: float, xl: float, xr: float, hl: float, hr: float, beta: float, gamma: float,
                     epsilon: float):
    """
    max over x in [xl, xr] of u(m - x) + beta * h(x) with h linear between (xl, hl) and (xr, hr).

    The objective is concave on the segment; the first-order condition u'(c) = beta * slope
    gives c + epsilon = (beta * slope)^(-1/gamma) in closed form (CRRA, also for gamma = 1).
    """
    slope = (hr - hl) / (xr - xl)
    if slope <= 0.0:
        x = xl
    else:
        c = (beta * slope) ** (-1.0 / gamma) - epsilon
        x = min(max(m - c, xl), xr)
    return x, crra_utility_scalar(m - x, gamma, epsilon) + beta * (hl + slope * (x - xl))


@nb.njit(cache=True)
def _optimal_investment(m: float, grid: np.ndarray, h: np.ndarray, l: int, beta: float, gamma: float,
                        epsilon: float):
    """
    Optimal post-decision wealth x in [0, m] for resources m = w + y > 0.

    h is the continuation value on the grid (linear in between, extrapolated above the grid).
    A forward-moving pointer l finds the best grid point (the optimal x is non-decreasing in
    m, so callers pass the previous l), then the two adjacent segments are solved in closed form.

    Returns
    -------
    (x, value, l)
    """
    n = grid.shape[0]
    f = crra_utility_scalar(m - grid[l], gamma, epsilon) + beta * h[l]
    while l + 1 < n and grid[l + 1] <= m:
        f_next = crra_utility_scalar(m - grid[l + 1], gamma, epsilon) + beta * h[l + 1]
        if f_next < f:
            break
        l += 1
        f = f_next
    x_best, f_best = grid[l], f
    if l > 0:
        x, fx = _segment_optimum(m, grid[l - 1], grid[l], h[l - 1], h[l], beta, gamma, epsilon)
        if fx > f_best:
            x_best, f_best = x, fx
    if l + 1 < n:
        xr = min(grid[l + 1], m)
        if xr > grid[l]:
            hr = h[l] + (h[l + 1] - h[l]) * (xr - grid[l]) / (grid[l + 1] - grid[l])
            x, fx = _segment_optimum(m, grid[l], xr, h[l], hr, beta, gamma, epsilon)
            if fx > f_best:
                x_best, f_best = x, fx
    elif m > grid[l]:  # above the grid: extrapolate with the slope of the last segment
        slope = (h[n - 1] - h[n - 2]) / (grid[n - 1] - grid[n - 2])
        x, fx = _segment_optimum(m, grid[l], m, h[l], h[l] + slope * (m - grid[l]), beta, gamma, epsilon)
        if fx > f_best:
            x_best, f_best = x, fx
    return x_best, f_best, l


@nb.njit(parallel=True, cache=True)
def bellman_values(grid: np.ndarray, cont: np.ndarray, y: float, beta: float, gamma: float, epsilon: float,
                   shortfall_penalty: float) -> np.ndarray:
    """
    V_t(w_j, s) = max_{0 <= x <= w_j + y} u(w_j + y - x) + beta * cont(x, s).

    If w + y < 0 the fixed expenses cannot be paid: c = 0, x = 0 and each missing euro
    costs ``shortfall_penalty``.

    Parameters
    ----------
    grid : np.ndarray, shape (n_w,)
    cont : np.ndarray, shape (n_sims, n_w)
        Continuation value (already including the survival probability) per path.
    y : float
        Net cashflow of the month.

    Returns
    -------
    np.ndarray, shape (n_sims, n_w)
    """
    n_sims, n_w = cont.shape
    out = np.empty((n_sims, n_w))
    for s in nb.prange(n_sims):
        h = cont[s]
        l = 0
        for j in range(n_w):
            m = grid[j] + y
            if m <= 0.0:
                out[s, j] = crra_utility_scalar(0.0, gamma, epsilon) + beta * h[0] + shortfall_penalty * m
                continue
            _, out[s, j], l = _optimal_investment(m, grid, h, l, beta, gamma, epsilon)
    return out


@nb.njit(cache=True)
def _decide(m: float, grid: np.ndarray, coef: np.ndarray, phi: np.ndarray, shares: np.ndarray, beta: float,
            gamma: float, epsilon: float, h: np.ndarray, best_share: np.ndarray, reserve: float):
    """
    Optimal invested amount x and equity share for resources m > 0 at one market state.

    The regressed continuation value (coef: (D, n_w, n_a), basis values phi) is maximised over
    the equity shares per grid point; the consumption follows from ``_optimal_investment``,
    the equity share is interpolated between the neighbouring grid points of x.
    h and best_share are work arrays of length n_w. Equity shares that would leave less than
    ``reserve`` in the savings account are not admissible.
    """
    n_w, n_a, d_max = grid.shape[0], shares.shape[0], phi.shape[0]
    for j in range(n_w):
        vmax, abest = -np.inf, 0.0
        cap = _max_share(grid[j], reserve) + 1e-12
        for k in range(n_a):
            if k > 0 and shares[k] > cap:
                # the cap itself lies between shares[k-1] and shares[k]: interpolate linearly
                if shares[k - 1] < cap - 1e-12:
                    v0 = 0.0
                    v1 = 0.0
                    for i in range(d_max):
                        v0 += phi[i] * coef[i, j, k - 1]
                        v1 += phi[i] * coef[i, j, k]
                    w = (cap - shares[k - 1]) / (shares[k] - shares[k - 1])
                    vc = v0 + w * (v1 - v0)
                    if vc > vmax:
                        vmax, abest = vc, min(cap, 1.0)
                break
            v = 0.0
            for i in range(d_max):
                v += phi[i] * coef[i, j, k]
            if v > vmax:
                vmax, abest = v, shares[k]
        h[j] = vmax
        best_share[j] = abest
    x, _, l = _optimal_investment(m, grid, h, 0, beta, gamma, epsilon)
    if l + 1 < n_w and x > grid[l]:
        wl = (x - grid[l]) / (grid[l + 1] - grid[l])
        alpha = best_share[l] + wl * (best_share[l + 1] - best_share[l])
    elif l > 0 and x < grid[l]:
        wl = (x - grid[l - 1]) / (grid[l] - grid[l - 1])
        alpha = best_share[l - 1] + wl * (best_share[l] - best_share[l - 1])
    else:
        alpha = best_share[l]
    return x, min(alpha, _max_share(x, reserve))


@nb.njit(parallel=True, cache=True)
def forward_paths(grid: np.ndarray, coeffs: np.ndarray, means: np.ndarray, stds: np.ndarray, degrees: np.ndarray,
                  shares: np.ndarray, cashflows: np.ndarray, equity: np.ndarray, alive: np.ndarray, w0: float,
                  equity0: float, rf: float, equity_tax: float, savings_tax: float, beta: float, gamma: float,
                  epsilon: float, terminal_consume: bool, deflator: float, reserve: np.ndarray):
    """
    Simulate the optimal policy on out-of-sample paths.

    At every month the regressed continuation value is evaluated at the path's market state
    (``_decide``). Holdings are tracked per asset in real euros: the equity holding grows with
    the nominal equity return after tax 1 + (R - 1)(1 - equity_tax), the savings holding with
    rf (nominal, after tax); both are deflated by ``deflator`` = 1 / monthly inflation factor.

    Returns
    -------
    wealth, consumption, equity_share, invested, equity_before, savings_before, capital_tax :
        np.ndarray, shape (T, n_paths), NaN after death. ``*_before`` are the holdings at the
        start of the month (before consumption and rebalancing); ``capital_tax`` is the capital
        gains tax on the returns of month t (negative: loss credited).
    """
    n_paths, n_t = equity.shape
    n_w = grid.shape[0]
    wealth = np.full((n_t, n_paths), np.nan)
    cons = np.full((n_t, n_paths), np.nan)
    share = np.full((n_t, n_paths), np.nan)
    invested = np.full((n_t, n_paths), np.nan)
    eq_before = np.full((n_t, n_paths), np.nan)
    sav_before = np.full((n_t, n_paths), np.nan)
    cap_tax = np.full((n_t, n_paths), np.nan)
    d_max = coeffs.shape[1]
    for p in nb.prange(n_paths):
        eq_hold = min(max(equity0, 0.0), w0)
        sav_hold = w0 - eq_hold
        phi = np.zeros(d_max)
        h = np.empty(n_w)
        best_share = np.empty(n_w)
        for t in range(n_t):
            if alive[p, t] == 0.0:
                break
            w = eq_hold + sav_hold
            wealth[t, p] = w
            eq_before[t, p] = eq_hold
            sav_before[t, p] = sav_hold
            m = w + cashflows[t]
            if t == n_t - 1:  # last month
                cons[t, p] = max(m, 0.0) if terminal_consume else 0.0
                invested[t, p] = 0.0 if terminal_consume else max(m, 0.0)
                share[t, p] = 0.0
                break
            if m <= 0.0:  # fixed expenses cannot be paid
                cons[t, p] = 0.0
                invested[t, p] = 0.0
                share[t, p] = 0.0
                eq_hold = 0.0
                sav_hold = 0.0
                continue
            z = (np.log(equity[p, t]) - means[t]) / stds[t] if stds[t] > 0.0 else 0.0
            for i in range(d_max):
                phi[i] = z ** i if i <= degrees[t] else 0.0
            x, alpha = _decide(m, grid, coeffs[t], phi, shares, beta, gamma, epsilon, h, best_share, reserve[t])
            cons[t, p] = m - x
            invested[t, p] = x
            share[t, p] = alpha
            r = equity[p, t + 1] / equity[p, t]
            eq_hold = alpha * x * (1.0 + (r - 1.0) * (1.0 - equity_tax)) * deflator
            sav_hold = (1.0 - alpha) * x * rf * deflator
            # rf is after tax: pre-tax interest = after-tax interest / (1 - savings_tax)
            cap_tax[t, p] = deflator * (alpha * x * (r - 1.0) * equity_tax + (1.0 - alpha) * x * (rf - 1.0)
                                        * savings_tax / (1.0 - savings_tax))
    return wealth, cons, share, invested, eq_before, sav_before, cap_tax


def decide(policy: Dict[str, np.ndarray], t: int, wealth: float, equity_level: float = 1.0) -> Dict[str, float]:
    """
    Optimal decision of month t for a given wealth, from a saved policy (``policy_dict``).

    Parameters
    ----------
    policy : dict
        Arrays of ``LSMCBellmanOptimizer.policy_dict()`` (also loadable from policy.npz).
    t : int
        Month index.
    wealth : float
        Liquid wealth at the start of the month (equity + savings), in real euros of the start
        date (divide nominal amounts by ``policy["price_index"][t]``).
    equity_level : float
        Equity level S_t / S_0 (only matters for a regression degree > 0).

    Returns
    -------
    dict
        resources (wealth + net cashflow), consumption, invested, equity_share, equity, savings,
        shortfall.
    """
    cf = float(policy["cashflows"][t])
    m = wealth + cf
    n_t = policy["cashflows"].shape[0]
    out = {"net_cashflow": cf, "resources": m, "shortfall": max(-m, 0.0)}
    if t >= n_t - 1 or m <= 0.0:
        terminal_consume = bool(policy["terminal_consume"])
        c = max(m, 0.0) if (t >= n_t - 1 and terminal_consume) else 0.0
        out.update(consumption=c, invested=max(m, 0.0) - c, equity_share=0.0)
    else:
        d_max = policy["coeffs"].shape[1]
        std = float(policy["state_std"][t])
        z = (np.log(equity_level) - float(policy["state_mean"][t])) / std if std > 0 else 0.0
        phi = np.array([z ** i if i <= int(policy["state_degree"][t]) else 0.0 for i in range(d_max)])
        grid = policy["wealth_grid"]
        x, alpha = _decide(m, grid, policy["coeffs"][t], phi, policy["equity_shares"], float(policy["beta"]),
                           float(policy["gamma"]), float(policy["epsilon"]), np.empty(grid.size),
                           np.empty(grid.size), float(policy["reserve"][t]))
        out.update(consumption=m - x, invested=x, equity_share=alpha)
    out["equity"] = out["equity_share"] * out["invested"]
    out["savings"] = out["invested"] - out["equity"]
    return out


# ---------------------------------------------------------------------------- optimizer


class LSMCBellmanOptimizer:
    """Least-squares Monte Carlo Bellman optimizer for consumption and equity share."""

    def __init__(
        self,
        run_config_id: str,
        start_date: dt.date,
        end_date: dt.date,
        retirement_date: dt.date,
        initial_wealth: float,
        yearly_return_savings: float,
        cashflows: List[CashflowBase],
        gbm_returns: GBM,
        survival_model: SurvivalModel,
        survival_process: SurvivalProcess,
        current_age: float,
        beta: float,
        gamma: float,
        epsilon: float,
        n_sims: int,
        n_sims_oos: int,
        seed: Optional[int],
        w_max: float,
        n_wealth: int,
        n_equity_shares: int,
        max_equity_share: float = 1.0,
        regression_degree: int = 0,
        shortfall_penalty: float = 1.0,
        apply_terminal_penalty: bool = False,
        savings_tax_rate: float = 0.0,
        equity_tax_rate: float = 0.0,
        initial_equity: float = 0.0,
        inflation_rate: float = 0.0,
        reserve_months: float = 0.0,
        use_cache: bool = True,
        params_hash: str = "",
        cache_root: Optional[str] = None,
    ) -> None:
        """
        Initialize the LSMC Bellman optimizer.

        Parameters
        ----------
        run_config_id : str
            Name of the run.
        start_date, end_date, retirement_date : dt.date
            Plan horizon (month starts) and retirement month.
        initial_wealth : float
            Liquid wealth at the start.
        yearly_return_savings : float
            Yearly rate of the risk-free savings account.
        cashflows : List[CashflowBase]
            Deterministic cashflows (salary, expenses, pensions, ...).
        gbm_returns : GBM
            Equity model; its seed drives the regression paths (out-of-sample: seed + 1).
        survival_model : SurvivalModel
            Gompertz model for the monthly survival probabilities p_t (backward induction).
        survival_process : SurvivalProcess
            Gompertz death times for the out-of-sample paths.
        current_age : float
            Age at the start date in years.
        beta : float
            Monthly discount factor.
        gamma, epsilon : float
            CRRA risk aversion and consumption offset.
        n_sims, n_sims_oos : int
            Regression paths (backward induction) and out-of-sample paths (forward simulation).
        seed : int, optional
            Seed of the regression paths (overrides the GBM seed if given).
        w_max, n_wealth : float, int
            Wealth grid [0, w_max] with n_wealth points, denser at low wealth.
        n_equity_shares, max_equity_share : int, float
            Equity share grid: n_equity_shares points in [0, max_equity_share].
        regression_degree : int
            Polynomial degree of the LSMC regression in the log equity level. 0 (constant
            basis = plain Monte Carlo mean) is the right choice for i.i.d. GBM returns; higher
            degrees only add regression noise unless the market model has a state.
        shortfall_penalty : float
            Utility cost per euro of fixed expenses that cannot be paid.
        apply_terminal_penalty : bool
            True: terminal value square_penalty(w) instead of consuming everything at the end.
        savings_tax_rate, equity_tax_rate : float
            Capital gains tax on interest and (after partial exemption) on equity returns,
            charged when the return accrues (``TaxModel``).
        initial_equity : float
            Part of the initial wealth currently held in equity (forward simulation only).
        inflation_rate : float
            Yearly inflation. The optimizer works in real euros of the start date: cashflows
            are deflated by the price index, returns by the monthly inflation factor; taxes
            apply to the nominal returns.
        reserve_months : float
            Emergency reserve: at least this many months of (real) fixed costs stay in the
            savings account (0: no reserve).
        use_cache : bool
            Cache the regression coefficients per month (ResultCache).
        params_hash : str
            Hash of all parameters (cache key, set by the config mapper).
        cache_root : str, optional
            Cache folder (default $CACHE_FOLDER or ./.cache).
        """
        self.run_config_id = run_config_id
        self.start_date = start_date
        self.end_date = end_date
        self.retirement_date = retirement_date
        self.initial_wealth = float(initial_wealth)
        self.yearly_return_savings = yearly_return_savings
        self.cashflows = cashflows
        self.gbm_returns = gbm_returns
        self.survival_model = survival_model
        self.survival_process = survival_process
        self.current_age = current_age
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_sims = n_sims
        self.n_sims_oos = n_sims_oos
        self.seed = seed if seed is not None else gbm_returns.seed
        self.regression_degree = regression_degree
        self.shortfall_penalty = shortfall_penalty
        self.apply_terminal_penalty = apply_terminal_penalty

        self.months: List[dt.date] = [d.date() for d in pd.date_range(start=start_date, end=end_date, freq="MS")]
        self.n_months = len(self.months)
        if self.n_months < 2:
            raise ValueError("The plan needs at least two months.")

        self.dt = 1.0 / 12.0
        self.cf = np.array([self.monthly_cashflow(m) for m in self.months])
        self.time_grid = np.array([(m - self.months[0]).days / 365.0 for m in self.months])
        self.age_grid = self.current_age + self.time_grid
        self.p_survive = self.survival_model.conditional_survival_probabilities(self.age_grid, self.dt)
        self.survival_probabilities = np.concatenate(([1.0], np.cumprod(self.p_survive[:-1])))
        self.savings_tax_rate = savings_tax_rate
        self.equity_tax_rate = equity_tax_rate
        self.initial_equity = float(initial_equity)
        # savings gross return after tax on the (nominal) interest; real: divided by inflation
        self.inflation_rate = inflation_rate
        self.inflation_factor = (1.0 + inflation_rate) ** self.dt
        self.rf_nominal = 1.0 + ((1.0 + yearly_return_savings) ** self.dt - 1.0) * (1.0 - savings_tax_rate)
        self.rf = self.rf_nominal / self.inflation_factor
        self.price_index = self.inflation_factor ** np.arange(self.n_months)
        self.cf_nominal = self.cf
        self.cf = self.cf_nominal / self.price_index
        items = self.cashflow_items().values / self.price_index[:, None]
        self.fixed_costs = -np.where(items < 0.0, items, 0.0).sum(axis=1)  # real
        self.reserve_months = reserve_months
        # reserve on the average fixed costs of the next 12 months (yearly insurances would
        # otherwise move money between savings account and equity every January)
        avg = np.array([self.fixed_costs[t:t + 12].mean() for t in range(self.n_months)])
        self.reserve = reserve_months * avg

        self.wealth_grid = w_max * np.linspace(0.0, 1.0, n_wealth) ** 2
        self.equity_shares = np.linspace(0.0, max_equity_share, n_equity_shares)

        # Cache folder keyed by the parameter hash only (not the run name): identical parameters
        # under another name reuse the result.
        self.cache = ResultCache(run_id="lsmc", enabled=use_cache, params_hash=params_hash, cache_root=cache_root)

        # outputs (populated by solve())
        d_max = regression_degree + 1
        self.coeffs = np.zeros((self.n_months, d_max, n_wealth, n_equity_shares))
        self.state_mean = np.zeros(self.n_months)
        self.state_std = np.zeros(self.n_months)
        self.state_degree = np.zeros(self.n_months, dtype=np.int64)
        self.r_squared: Dict[dt.date, float] = {}
        self.value_in_sample: float = float("nan")
        self.from_cache: bool = False
        self.runtimes: Dict[str, float] = {}

        self.opt_wealth = pd.DataFrame()
        self.opt_consumption = pd.DataFrame()
        self.opt_equity_share = pd.DataFrame()
        self.opt_invested = pd.DataFrame()
        self.opt_equity_before = pd.DataFrame()
        self.opt_savings_before = pd.DataFrame()
        self.opt_capital_tax = pd.DataFrame()
        self.survival_paths = np.empty((0, 0))
        self.monthly_cashflows = pd.Series(self.cf, index=self.months)

    # ------------------------------------------------------------------ helpers

    def monthly_cashflow(self, date: dt.date) -> float:
        """Sum deterministic cashflows for the given month."""
        return sum(cf.cashflow(date) for cf in self.cashflows)

    def _terminal_value(self) -> np.ndarray:
        """V_{T-1} on the wealth grid: consume everything in the last month, or the penalty."""
        if self.apply_terminal_penalty:
            return square_penalty(self.wealth_grid)
        return crra_utility(self.wealth_grid + self.cf[-1], self.gamma, self.epsilon)

    def _equity_paths(self, n_sims: int, seed: Optional[int], moment_matching: bool = False) -> np.ndarray:
        return self.gbm_returns.simulate(n_sims=n_sims, dates=self.months, seed=seed,
                                         moment_matching=moment_matching)

    def _store(self, t: int, reg: WealthRegressor) -> None:
        d = reg.degree + 1
        n_w, n_a = self.wealth_grid.size, self.equity_shares.size
        self.coeffs[t] = 0.0
        self.coeffs[t, :d] = reg.coeffs.reshape(d, n_w, n_a)
        self.state_mean[t] = reg.mean
        self.state_std[t] = reg.std
        self.state_degree[t] = reg.degree
        self.r_squared[self.months[t]] = reg.r_squared

    def _load_cache(self) -> bool:
        """Load all months from the cache; False if any month is missing."""
        decision_months = self.months[:-1]
        if not all(self.cache.has(m) for m in decision_months):
            return False
        for t, m in enumerate(decision_months):
            data = self.cache.load_date(m)
            d = int(data["degree"]) + 1
            self.coeffs[t] = 0.0
            self.coeffs[t, :d] = data["coeffs"]
            self.state_mean[t] = data["mean"]
            self.state_std[t] = data["std"]
            self.state_degree[t] = data["degree"]
            self.r_squared[m] = data["r_squared"]
        self.value_in_sample = float(self.cache.load_date(self.months[0]).get("value_in_sample", np.nan))
        return True

    # ------------------------------------------------------------------ solver

    def backward_induction(self, progress: Optional[ProgressFn] = None) -> None:
        """Regression coefficients of the continuation value for every month (cached)."""
        if self._load_cache():
            logger.info("Cache hit for all %d months (%s)", self.n_months - 1, self.cache.cache_dir)
            self.from_cache = True
            return

        # moment matching: the comparison of the equity shares is not distorted by the random
        # mean / variance of the monthly draws
        equity = self._equity_paths(self.n_sims, self.seed, moment_matching=True)
        n_w = self.wealth_grid.size
        v_next = np.tile(self._terminal_value(), (self.n_sims, 1))  # (n_sims, n_w)

        n_steps = self.n_months - 1
        for t in tqdm(range(n_steps - 1, -1, -1), total=n_steps, desc="Backward induction", mininterval=1.0):
            # after tax, real
            r = (1.0 + (equity[:, t + 1] / equity[:, t] - 1.0) * (1.0 - self.equity_tax_rate)) / self.inflation_factor
            gross = self.equity_shares[:, None] * r[None, :] + (1.0 - self.equity_shares[:, None]) * self.rf
            reg = WealthRegressor(equity[:, t], degree=self.regression_degree)
            proj, sum_y, sum_y2 = project_realised_values(self.wealth_grid, v_next, gross, reg.q,
                                                          float(self.p_survive[t]))
            reg.fit_projection(proj.reshape(proj.shape[0], -1), sum_y.ravel(), sum_y2.ravel())
            self._store(t, reg)

            cont = best_continuation(reg.q, proj, self.wealth_grid, self.equity_shares,
                                     float(self.reserve[t]))  # best admissible equity share per (path, x)
            v_next = bellman_values(self.wealth_grid, cont, float(self.cf[t]), self.beta, self.gamma, self.epsilon,
                                    self.shortfall_penalty)

            self.cache.store_date(self.months[t], {
                "coeffs": reg.coeffs.reshape(reg.degree + 1, n_w, -1), "mean": reg.mean, "std": reg.std,
                "degree": reg.degree, "r_squared": reg.r_squared,
            })
            if progress is not None:
                progress("backward induction", (n_steps - t) / n_steps)

        # V_0 at the initial wealth from the Bellman equation itself (interpolating the concave
        # V_0 between grid points would understate it); all paths share the state at t = 0.
        m0 = self.initial_wealth + float(self.cf[0])
        if m0 > 0.0:
            self.value_in_sample = float(_optimal_investment(m0, self.wealth_grid, cont[0], 0, self.beta, self.gamma,
                                                             self.epsilon)[1])
        else:
            self.value_in_sample = float(np.interp(self.initial_wealth, self.wealth_grid, v_next.mean(axis=0)))
        data = self.cache.load_date(self.months[0]) if self.cache.has(self.months[0]) else None
        if data is not None:
            data["value_in_sample"] = self.value_in_sample
            self.cache.store_date(self.months[0], data)

    def roll_forward(self) -> None:
        """Forward simulation of the optimal policy on out-of-sample paths."""
        oos_seed = None if self.seed is None else self.seed + 1
        equity = self._equity_paths(self.n_sims_oos, oos_seed)
        self.survival_paths = self.survival_process.simulate(n_sims=self.n_sims_oos, dates=self.months)
        wealth, cons, share, invested, eq_before, sav_before, cap_tax = forward_paths(
            self.wealth_grid, self.coeffs, self.state_mean, self.state_std, self.state_degree, self.equity_shares,
            self.cf, equity, self.survival_paths, self.initial_wealth, self.initial_equity, self.rf_nominal,
            self.equity_tax_rate, self.savings_tax_rate, self.beta, self.gamma, self.epsilon,
            not self.apply_terminal_penalty, 1.0 / self.inflation_factor, self.reserve)
        self.opt_capital_tax = pd.DataFrame(cap_tax, index=self.months)
        self.opt_equity_before = pd.DataFrame(eq_before, index=self.months)
        self.opt_savings_before = pd.DataFrame(sav_before, index=self.months)
        self.opt_wealth = pd.DataFrame(wealth, index=self.months)
        self.opt_consumption = pd.DataFrame(cons, index=self.months)
        self.opt_equity_share = pd.DataFrame(share, index=self.months)
        self.opt_invested = pd.DataFrame(invested, index=self.months)

    def solve(self, progress: Optional[ProgressFn] = None) -> None:
        """Backward induction (or cache), then forward simulation."""
        logger.info("LSMCBellmanOptimizer.solve() started.")
        t0 = time.perf_counter()
        self.backward_induction(progress=lambda s, f: progress(s, 0.85 * f) if progress else None)
        self.runtimes["backward_induction"] = time.perf_counter() - t0
        if progress is not None:
            progress("forward simulation", 0.85)
        t0 = time.perf_counter()
        self.roll_forward()
        self.runtimes["forward_simulation"] = time.perf_counter() - t0
        logger.info("LSMCBellmanOptimizer.solve() finished (%s).",
                    ", ".join(f"{k} {v:.1f} s" for k, v in self.runtimes.items()))

    # ------------------------------------------------------------------ results

    @property
    def retirement_index(self) -> int:
        """First month index on or after the retirement date (n_months if none)."""
        return next((i for i, m in enumerate(self.months) if m >= self.retirement_date), self.n_months)

    def realised_utility(self) -> np.ndarray:
        """Discounted lifetime utility of every out-of-sample path (same objective as the solver)."""
        cons = self.opt_consumption.values
        alive = ~np.isnan(cons)
        disc = self.beta ** np.arange(self.n_months)[:, None]
        u = crra_utility(np.where(alive, cons, 0.0), self.gamma, self.epsilon)
        shortfall = np.maximum(-(self.opt_wealth.values + self.cf[:, None]), 0.0)
        u = u - self.shortfall_penalty * np.where(alive, shortfall, 0.0)
        return np.sum(np.where(alive, disc * u, 0.0), axis=0)

    def summary(self) -> Dict[str, Any]:
        """Key figures of the solved plan (amounts in euro)."""
        if self.opt_wealth.empty:
            raise RuntimeError("No solution available — call solve() first.")
        r = self.retirement_index
        wealth, cons, share = self.opt_wealth.values, self.opt_consumption.values, self.opt_equity_share.values
        u = self.realised_utility()
        shortfall = np.nansum(np.maximum(-(wealth + self.cf[:, None]), 0.0) > 0.0, axis=0) > 0

        def at(data: np.ndarray, t: int, q: float = 50.0) -> float:
            if t >= data.shape[0] or not np.isfinite(data[t]).any():
                return float("nan")
            return float(np.nanpercentile(data[t], q))

        with np.errstate(all="ignore"):
            out: Dict[str, Any] = {
                "name": self.run_config_id,
                "start": self.months[0].isoformat(),
                "end": self.months[-1].isoformat(),
                "n_months": self.n_months,
                "n_sims_regression": self.n_sims,
                "n_sims_oos": self.n_sims_oos,
                "age_start": round(float(self.age_grid[0]), 2),
                "age_retirement": round(float(self.age_grid[min(r, self.n_months - 1)]), 2),
                "value_in_sample": self.value_in_sample,
                "value_oos": float(u.mean()),
                "value_oos_stderr": float(u.std(ddof=1) / np.sqrt(u.size)) if u.size > 1 else 0.0,
                "consumption_start": at(cons, 0),
                "equity_share_start": at(share, 0),
                "equity_share_retirement": at(share, r),
                "wealth_retirement_median": at(wealth, r),
                "wealth_retirement_p05": at(wealth, r, 5.0),
                "consumption_working_median": float(np.nanmedian(cons[:r])) if r > 0 else float("nan"),
                "consumption_retired_median": float(np.nanmedian(cons[r:])) if r < self.n_months else float("nan"),
                "prob_shortfall": float(shortfall.mean()),
                "life_expectancy": float(self.age_grid[0] + self.survival_paths.sum(axis=1).mean() / 12.0),
                "regression_r2_median": float(np.nanmedian(list(self.r_squared.values()))),
                "from_cache": self.from_cache,
            }
        out.update({f"runtime_{k}_s": v for k, v in self.runtimes.items()})
        return out

    def quantiles(self, q: tuple = (5, 25, 50, 75, 95)) -> pd.DataFrame:
        """Monthly quantiles and means of wealth, consumption, equity share and equity amount."""
        cols: Dict[str, Any] = {"age": self.age_grid, "net_cashflow": self.cf,
                                "alive": np.nanmean(self.survival_paths, axis=0)}
        series = {"wealth": self.opt_wealth.values, "consumption": self.opt_consumption.values,
                  "equity_share": self.opt_equity_share.values,
                  "equity_amount": self.opt_equity_share.values * self.opt_invested.values,
                  "equity_before": self.opt_equity_before.values, "savings_before": self.opt_savings_before.values}
        with np.errstate(all="ignore"):
            for name, data in series.items():
                valid = np.isfinite(data).any(axis=1)
                for p in q:
                    col = np.full(self.n_months, np.nan)
                    col[valid] = np.nanpercentile(data[valid], p, axis=1)
                    cols[f"{name}_p{p:02d}"] = col
                cols[f"{name}_mean"] = np.nanmean(data, axis=1)
        return pd.DataFrame(cols, index=pd.Index(self.months, name="month"))

    def policy_dict(self) -> Dict[str, np.ndarray]:
        """Everything ``decide`` needs, as arrays (saved as policy.npz with the results)."""
        return {
            "months": np.array([m.isoformat() for m in self.months]),
            "cashflows": self.cf, "wealth_grid": self.wealth_grid, "equity_shares": self.equity_shares,
            "coeffs": self.coeffs, "state_mean": self.state_mean, "state_std": self.state_std,
            "state_degree": self.state_degree, "beta": np.float64(self.beta), "gamma": np.float64(self.gamma),
            "epsilon": np.float64(self.epsilon), "terminal_consume": np.bool_(not self.apply_terminal_penalty),
            "price_index": self.price_index, "reserve": self.reserve,
        }

    def decide(self, t: int, wealth: float, equity_level: float = 1.0) -> Dict[str, float]:
        """Optimal decision of month t for a given wealth (see module function ``decide``)."""
        return decide(self.policy_dict(), t, wealth, equity_level)

    def cashflow_items(self) -> pd.DataFrame:
        """
        Deterministic cashflows per month and item (signed: income +, expenses -).

        Pensions are split into contributions, Rürup tax refund, gross payouts and income tax.
        """
        rows = []
        for m in self.months:
            row: Dict[str, float] = {}
            for cf in self.cashflows:
                parts = cf.components(m) if hasattr(cf, "components") else {
                    getattr(cf, "name", type(cf).__name__): cf.cashflow(m)}
                for key, value in parts.items():
                    row[key] = row.get(key, 0.0) + value
            rows.append(row)
        return pd.DataFrame(rows, index=pd.Index(self.months, name="month")).fillna(0.0)

    def cashflow_categories(self) -> Dict[str, str]:
        """Category of every column of ``cashflow_items`` (for the composition plot)."""
        cats: Dict[str, str] = {}
        for cf in self.cashflows:
            kind = getattr(cf, "kind", type(cf).__name__)
            if hasattr(cf, "components"):
                for p in cf.pensions:
                    cats[f"{p.name} contribution"] = "Pension contributions"
                    cats[f"{p.name} pension (gross)"] = "Pensions (gross)"
                cats["Rürup tax refund"] = "Tax refund"
                cats["Income tax on pensions"] = "Income tax"
            elif kind == "Salary":
                cats[kind] = "Salary"
            elif kind == "Insurance":
                cats[cf.name] = "Insurances"
            else:
                cats[getattr(cf, "name", kind)] = "Living costs"
        return cats

    def policy_grid(self, t: int, state: float = 1.0) -> tuple:
        """
        Equity share and continuation value on the wealth grid at month t for a given equity
        level (state = S_t / S_0; 1.0 = no market move). Used for the policy plots.
        """
        z = (np.log(state) - self.state_mean[t]) / self.state_std[t] if self.state_std[t] > 0 else 0.0
        phi = z ** np.arange(self.coeffs.shape[1])
        phi[self.state_degree[t] + 1:] = 0.0
        values = np.einsum("i,ijk->jk", phi, self.coeffs[t])
        return self.equity_shares[values.argmax(axis=1)], values.max(axis=1)

    def plot(self) -> None:
        """Show the result figures interactively (matplotlib)."""
        from wealthplan.plots import show_figures

        show_figures(self)
