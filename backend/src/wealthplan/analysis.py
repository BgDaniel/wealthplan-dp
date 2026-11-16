"""Derived analyses of a solved plan: depletion risk, taxes, Rürup vs. ETF savings plan.

All amounts in real euros of the start date (deflated with the plan's price index).
"""

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from wealthplan.cashflows.pension.ruerup_pension import RuerupPension


def depletion_risk(opt, months_of_costs: float = 6.0) -> pd.DataFrame:
    """
    Per month, among the paths alive: probability that the wealth is below ``months_of_costs``
    times the month's fixed costs, and probability that the fixed costs cannot be paid.
    """
    fixed = opt.fixed_costs
    wealth = opt.opt_wealth.values
    alive = np.isfinite(wealth)
    n_alive = np.maximum(alive.sum(axis=1), 1)
    low = np.where(alive, wealth < months_of_costs * fixed[:, None], False).sum(axis=1) / n_alive
    short = np.where(alive, wealth + opt.cf[:, None] < 0.0, False).sum(axis=1) / n_alive
    return pd.DataFrame({"age": opt.age_grid, "alive_paths": alive.sum(axis=1), "low_wealth": low,
                         "shortfall": short}, index=opt.months)


def yearly_taxes(opt) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Taxes per calendar year.

    Returns
    -------
    deterministic : pd.DataFrame
        Index year: age at the start of the year, income tax on pensions (>= 0), Rürup refund (>= 0).
    capital_tax : np.ndarray, shape (n_years, n_paths)
        Capital gains tax per year and path (only months alive; NaN if dead the whole year).
    """
    items = opt.cashflow_items().div(opt.price_index, axis=0)  # real
    years = np.array([m.year for m in opt.months])
    income_tax = -items.get("Income tax on pensions", pd.Series(0.0, index=items.index)).values
    refund = items.get("Rürup tax refund", pd.Series(0.0, index=items.index)).values
    cap = opt.opt_capital_tax.values
    uniq = np.unique(years)
    if (years == uniq[-1]).sum() < 12 and uniq.size > 1:  # incomplete last year of the plan
        uniq = uniq[:-1]
    det = pd.DataFrame(index=pd.Index(uniq, name="year"))
    det["age"] = [opt.age_grid[np.argmax(years == y)] for y in uniq]
    det["income_tax"] = [income_tax[years == y].sum() for y in uniq]
    det["rurup_refund"] = [refund[years == y].sum() for y in uniq]
    cap_year = np.full((uniq.size, cap.shape[1]), np.nan)
    for i, y in enumerate(uniq):
        block = cap[years == y]
        any_alive = np.isfinite(block).any(axis=0)
        cap_year[i, any_alive] = np.nansum(block[:, any_alive], axis=0)
    return det, cap_year


def ruerup_vs_etf(opt, etf_ter: float = 0.002) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Rürup against an ETF savings plan with the same net cost (contribution minus tax refund).

    ETF: return = equity return net of the TER, capital gains tax as in the model (accrual,
    partial exemption); from retirement a level monthly withdrawal that empties the capital at
    the plan end. Rürup: net pension = gross pension minus the additional income tax it causes.
    Flows are weighted with the probability to be alive; the ETF capital left at death goes to
    the heirs (shown separately).

    Returns
    -------
    frame : pd.DataFrame
        Per month: age, survival, net flows of both, cumulative expected flows, ETF capital.
    info : dict
        Capitals at retirement, net monthly payouts, break-even age (or None).
    """
    tp = next((cf for cf in opt.cashflows if hasattr(cf, "pensions")), None)
    plans = [p for p in tp.pensions if isinstance(p, RuerupPension)] if tp is not None else []
    if not plans:
        return pd.DataFrame(), {}
    tax = tp.tax
    others = [p for p in tp.pensions if not isinstance(p, RuerupPension)]
    def net_pension(date) -> float:
        """Rürup pension minus the additional income tax it causes (nominal)."""
        scale = tp.tax_scale(date)
        taxable_others = sum(p.payout(date) * tp.taxable_share(p) for p in others)
        taxable_ruerup = sum(p.payout(date) * tp.taxable_share(p) for p in plans)
        extra = (tax.pension_income_tax_monthly(taxable_others + taxable_ruerup, scale)
                 - tax.pension_income_tax_monthly(taxable_others, scale))
        return sum(p.payout(date) for p in plans) - extra

    mu = opt.gbm_returns.yearly_return
    r_m = (((1.0 + mu) * (1.0 - etf_ter)) ** (1.0 / 12.0) - 1.0) * (1.0 - tax.equity_tax_rate)  # nominal
    r_idx = opt.retirement_index
    n = opt.n_months
    rue = np.zeros(n)
    etf = np.zeros(n)
    capital = np.zeros(n)
    k = 0.0
    k_ret = 0.0
    withdrawal = 0.0
    for t, m in enumerate(opt.months):
        if t < r_idx:
            c = sum(p.monthly_contribution(m) for p in plans)
            outlay = c - tax.basis_pension_refund_monthly(c, tp.tax_scale(m))
            rue[t] = etf[t] = -outlay
            k = (k + outlay) * (1.0 + r_m)
        else:
            if t == r_idx:
                k_ret = k
                months_left = n - r_idx
                withdrawal = k * r_m / (1.0 - (1.0 + r_m) ** -months_left) if r_m > 0 else k / months_left
            rue[t] = net_pension(m)
            etf[t] = withdrawal
            k = max((k - withdrawal) * (1.0 + r_m), 0.0)
        capital[t] = k

    # nominal -> real euros of the start date
    rue, etf, capital = rue / opt.price_index, etf / opt.price_index, capital / opt.price_index
    alive = opt.survival_probabilities
    death_prob = alive * (1.0 - opt.p_survive)  # dies during month t
    frame = pd.DataFrame({
        "age": opt.age_grid, "alive": alive, "ruerup_flow": rue, "etf_flow": etf, "etf_capital": capital,
        "ruerup_cum": np.cumsum(alive * rue), "etf_cum": np.cumsum(alive * etf),
    }, index=opt.months)
    frame["etf_cum_with_heirs"] = frame["etf_cum"] + np.cumsum(death_prob * capital)
    p_ret = opt.price_index[min(r_idx, n - 1)]
    after = frame.iloc[r_idx:] if r_idx < n else frame.iloc[0:0]
    cross = after[after["ruerup_cum"] >= after["etf_cum"]]
    info = {
        "ruerup_capital_at_retirement": sum(p.capital_at_pension_start() for p in plans) / p_ret,
        "etf_capital_at_retirement": k_ret / p_ret,
        "ruerup_net_pension": rue[r_idx] if r_idx < n else 0.0,
        "ruerup_gross_pension": sum(p.payout(opt.months[min(r_idx, n - 1)]) for p in plans) / p_ret,
        "etf_monthly_withdrawal": etf[r_idx] if r_idx < n else 0.0,
        "net_monthly_cost": -rue[0] if r_idx > 0 else 0.0,
        "break_even_age": float(cross["age"].iloc[0]) if len(cross) else None,
        "etf_ter": etf_ter,
    }
    return frame, info
