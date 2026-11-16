"""Validation of the optimizer against closed-form solutions.

    python -m wealthplan.validation

Cases (CRRA utility u(c) = c^(1-γ)/(1-γ), monthly steps t = 0..T-1, everything consumed in the
last month, no borrowing):

1. Deterministic consumption smoothing (safe return R per month, no income): the Euler equation
   gives c_{t+1} = g c_t with g = (β R)^(1/γ); the budget Σ c_t / R^t = W_0 gives
   c_0 = W_0 / Σ_t (g / R)^t.
2. As 1 with a constant income y (human capital): c_0 = (W_0 + Σ_t y / R^t) / Σ_t (g / R)^t.
3. As 1 with tax on interest: R = 1 + r_m (1 - τ).
4. As 1 with inflation π, in real terms: R = R_nominal / (1 + π)^(1/12).
5. Mortality without annuities (Yaari): c_{t+1} / c_t = (β p_t R)^(1/γ) with the one-month
   survival probability p_t; c_0 from the budget.
6. Merton: GBM equity, safe rate, no income, no mortality: the optimal equity share is
   constant (myopic) and maximises E[(a R + (1 - a) R_f)^(1-γ)] / (1-γ) for one month
   (exact reference by Gauss-Hermite quadrature; the continuous-time value (μ - r) / (γ σ²)
   is reported for information).
"""

import copy
import datetime as dt
import logging
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List

import numpy as np

from config.stochastic.lsmc.lsmc_config_mapper import LSMCConfigMapper
from wealthplan.optimizer.stochastic.lsmc.lsmc_bellman_optimizer import LSMCBellmanOptimizer

logger = logging.getLogger(__name__)

START = dt.date(2026, 1, 1)


@dataclass
class CheckResult:
    name: str
    formula: str
    expected: float
    model: float
    rel_error: float
    tolerance: float
    passed: bool
    note: str = ""


def base_config(years: int = 10, wealth: float = 100_000.0, rate: float = 0.03, gamma: float = 2.0,
                beta_yearly: float = 0.98) -> Dict[str, Any]:
    """Minimal plan: no cashflows, (practically) no mortality, no taxes, no inflation, no reserve."""
    end = START.replace(year=START.year + years)
    return {
        "simulation_parameters": {
            "run_config_id": "validation", "initial_wealth": wealth, "start_date": START, "end_date": end,
            "retirement_date": end.replace(year=end.year + 1), "current_age": 40.0,
            "yearly_return_savings": rate, "beta": beta_yearly ** (1.0 / 12.0),
        },
        "lifecycle": {},
        "stochastic": {
            "yearly_return": rate, "sigma": 1e-6, "n_sims": 50, "n_sims_oos": 50, "seed": 1,
            "survival_model": {"table": "custom", "b": 1e-12, "c": 0.01, "seed": 2},
        },
        "technical": {"w_max": 4.0 * wealth, "n_wealth": 800, "n_equity_shares": 11, "max_equity_share": 1.0,
                      "regression_degree": 0, "shortfall_penalty": 1.0, "apply_terminal_penalty": False,
                      "use_cache": False},
        "utility_function": {"gamma": gamma, "epsilon": 1e-6},
        "tax": {"tax_savings_interest": False, "tax_equity_returns": False},
        "inflation": {"rate": 0.0},
        "reserve": {"enabled": False},
    }


def solve(data: Dict[str, Any]) -> LSMCBellmanOptimizer:
    opt = LSMCBellmanOptimizer(**LSMCConfigMapper.map_yaml_to_params(copy.deepcopy(data)))
    opt.solve()
    return opt


def _closed_form_c0(wealth: float, income: np.ndarray, gross: np.ndarray, growth: np.ndarray) -> float:
    """c_0 from the budget Σ c_t / Π R = W_0 + Σ y_t / Π R with c_t = c_0 Π growth."""
    n = income.size
    disc = np.concatenate(([1.0], np.cumprod(1.0 / gross[: n - 1])))  # 1 / R^t
    path = np.concatenate(([1.0], np.cumprod(growth[: n - 1])))  # c_t / c_0
    return (wealth + np.sum(income * disc)) / np.sum(path * disc)


def _result(name: str, formula: str, expected: float, model: float, tolerance: float, note: str = "") -> CheckResult:
    rel = abs(model - expected) / max(abs(expected), 1e-12)
    return CheckResult(name, formula, float(expected), float(model), float(rel), tolerance, bool(rel <= tolerance),
                       note)


def _consumption_check(name: str, formula: str, data: Dict[str, Any], gross_monthly: float,
                       income: float = 0.0) -> List[CheckResult]:
    opt = solve(data)
    n = opt.n_months
    beta, gamma = opt.beta, opt.gamma
    gross = np.full(n, gross_monthly)
    growth = (beta * gross) ** (1.0 / gamma)
    c0 = _closed_form_c0(opt.initial_wealth, np.full(n, income), gross, growth)
    cons = np.nanmedian(opt.opt_consumption.values, axis=1)
    # average growth over the horizon: log-linear least-squares fit over all months but the last
    # (single months jitter by the grid: the continuation value is linear between grid points,
    # its slope piecewise constant; a ratio of two single months would pick up that jitter)
    mean_growth = float(np.exp(np.polyfit(np.arange(n - 1), np.log(cons[:n - 1]), 1)[0]))
    return [
        _result(f"{name}: c_0", formula, c0, cons[0], 0.003),
        _result(f"{name}: mean growth", "g = (β R)^(1/γ)", growth[0], mean_growth, 1e-4),
    ]


def check_smoothing() -> List[CheckResult]:
    data = base_config()
    rm = 1.03 ** (1 / 12)
    return _consumption_check("Consumption smoothing", "c_0 = W_0 / Σ (g/R)^t", data, rm)


def check_income() -> List[CheckResult]:
    data = base_config()
    data["lifecycle"] = {"salary": {"monthly_amount": 2_000.0,
                                    "retirement_date": data["simulation_parameters"]["retirement_date"]}}
    data["technical"]["w_max"] = 1_000_000.0
    rm = 1.03 ** (1 / 12)
    return _consumption_check("Constant income", "c_0 = (W_0 + Σ y/R^t) / Σ (g/R)^t", data, rm, income=2_000.0)


def check_interest_tax() -> List[CheckResult]:
    data = base_config()
    data["tax"] = {"tax_savings_interest": True, "tax_equity_returns": True, "capital_gains_tax_rate": 0.26375,
                   "equity_partial_exemption": 0.0}
    rm = 1.0 + (1.03 ** (1 / 12) - 1.0) * (1.0 - 0.26375)
    return _consumption_check("Tax on interest", "R = 1 + r_m (1 - τ)", data, rm)


def check_inflation() -> List[CheckResult]:
    data = base_config()
    data["inflation"] = {"rate": 0.02}
    rm = 1.03 ** (1 / 12) / 1.02 ** (1 / 12)
    return _consumption_check("Inflation (real terms)", "R = R_nom / (1+π)^(1/12)", data, rm)


def check_mortality() -> List[CheckResult]:
    data = base_config(years=15)
    data["simulation_parameters"]["current_age"] = 65.0
    data["stochastic"]["survival_model"] = {"table": "destatis_unisex", "seed": 2}
    data["stochastic"]["n_sims_oos"] = 400
    opt = solve(data)
    n = opt.n_months
    rm = 1.03 ** (1 / 12)
    gross = np.full(n, rm)
    growth = (opt.beta * opt.p_survive * gross) ** (1.0 / opt.gamma)
    c0 = _closed_form_c0(opt.initial_wealth, np.zeros(n), gross, growth)
    # a path alive until the end
    alive_all = np.where(opt.survival_paths.all(axis=1))[0]
    cons = opt.opt_consumption.values[:, alive_all[0]]
    t = n // 2
    return [
        _result("Mortality (Yaari): c_0", "c_0 = W_0 / Σ_t Π_s (β p_s R)^(1/γ) / R^t", c0, cons[0], 0.003),
        _result("Mortality (Yaari): c_t / c_0 at mid horizon", "Π_s<t (β p_s R)^(1/γ)",
                float(np.prod(growth[:t])), cons[t] / cons[0], 0.005),
    ]


def merton_reference(mu: float, sigma: float, rate: float, gamma: float, n_nodes: int = 80) -> float:
    """Optimal one-month equity share by Gauss-Hermite quadrature (log-normal equity return)."""
    x, w = np.polynomial.hermite_e.hermegauss(n_nodes)
    w = w / w.sum()
    dt_ = 1.0 / 12.0
    r_eq = np.exp((np.log1p(mu) - 0.5 * sigma ** 2) * dt_ + sigma * np.sqrt(dt_) * x)
    r_f = (1.0 + rate) ** dt_
    shares = np.linspace(0.0, 1.0, 10_001)
    util = (((shares[:, None] * r_eq[None, :] + (1.0 - shares[:, None]) * r_f) ** (1.0 - gamma)) @ w) / (1.0 - gamma)
    return float(shares[np.argmax(util)])


def check_merton() -> List[CheckResult]:
    mu, sigma, rate, gamma = 0.07, 0.18, 0.02, 3.0
    data = base_config(years=10, rate=rate, gamma=gamma)
    data["stochastic"].update(yearly_return=mu, sigma=sigma, n_sims=2_000, n_sims_oos=500)
    data["technical"].update(n_equity_shares=21, w_max=1_000_000.0)
    opt = solve(data)
    reference = merton_reference(mu, sigma, rate, gamma)
    model = float(np.nanmedian(opt.opt_equity_share.values[: 5 * 12]))
    continuous = (np.log1p(mu) - np.log1p(rate)) / (gamma * sigma ** 2)
    step = 1.0 / (opt.equity_shares.size - 1)
    res = _result("Merton equity share", "argmax E[(a R + (1-a) R_f)^(1-γ)]", reference, model, 0.0)
    res.tolerance = step / max(reference, 1e-9)  # one step of the equity share grid
    res.passed = abs(model - reference) <= step + 1e-9
    res.note = f"continuous-time (μ−r)/(γσ²) = {continuous:.3f}; share grid step {step:.2f}"
    return [res]


CHECKS: Dict[str, Callable[[], List[CheckResult]]] = {
    "smoothing": check_smoothing,
    "income": check_income,
    "interest_tax": check_interest_tax,
    "inflation": check_inflation,
    "mortality": check_mortality,
    "merton": check_merton,
}


def run_all() -> List[CheckResult]:
    results: List[CheckResult] = []
    for name, check in CHECKS.items():
        logger.info("validation check %s", name)
        results.extend(check())
    return results


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    results = run_all()
    width = max(len(r.name) for r in results)
    for r in results:
        print(f"{'PASS' if r.passed else 'FAIL'}  {r.name:{width}s}  expected {r.expected:12.6g}  model "
              f"{r.model:12.6g}  rel.err {r.rel_error:.2e}  (tol {r.tolerance:.1e})  {r.note}")
    print(f"{sum(r.passed for r in results)} / {len(results)} checks passed")


def results_as_dicts(results: List[CheckResult]) -> List[Dict[str, Any]]:
    return [asdict(r) for r in results]


if __name__ == "__main__":
    main()
