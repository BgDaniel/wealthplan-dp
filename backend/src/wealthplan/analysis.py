"""Derived analyses of a solved plan: depletion risk and taxes.

All amounts in real euros of the start date (deflated with the plan's price index).
"""

from typing import Tuple

import numpy as np
import pandas as pd



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
    oom = opt.out_of_money()  # incl. the path-dependent Rürup pension
    return pd.DataFrame({"age": opt.age_grid, "alive_paths": alive.sum(axis=1), "low_wealth": low,
                         "shortfall": oom["share_now"].values, "shortfall_ever": oom["share_ever"].values},
                        index=opt.months)


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
    if not opt.opt_ruerup_tax.empty:  # income tax caused by the fund-linked Rürup pension (median path)
        rtax = opt.opt_ruerup_tax.values
        with np.errstate(all="ignore"):
            det["income_tax"] += [float(np.nanmedian(np.nansum(rtax[years == y], axis=0)))
                                  if np.isfinite(rtax[years == y]).any() else 0.0 for y in uniq]
    det["rurup_refund"] = [refund[years == y].sum() for y in uniq]
    cap_year = np.full((uniq.size, cap.shape[1]), np.nan)
    for i, y in enumerate(uniq):
        block = cap[years == y]
        any_alive = np.isfinite(block).any(axis=0)
        cap_year[i, any_alive] = np.nansum(block[:, any_alive], axis=0)
    return det, cap_year
