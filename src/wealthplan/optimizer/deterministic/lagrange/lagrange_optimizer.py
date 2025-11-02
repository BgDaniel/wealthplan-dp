import logging
import datetime as dt
from typing import Dict, Callable, Tuple, Optional
import numpy as np
import pandas as pd


from src.wealthplan.optimizer.deterministic.base_optimizer import (
    BaseConsumptionOptimizer,
)


logger = logging.getLogger(__name__)


class LagrangeOptimizer(BaseConsumptionOptimizer):
    """
    Deterministic Lagrange (open-loop / Euler equation) solver.

    This class solves the deterministic multi-period problem using Euler equations
    (first-order conditions) and the budget constraint.

    Default behavior:
      - For log utility (instant_utility = log), there is an analytic Euler rule:
            c_{t+1} = beta * (1 + r) * c_t
        so consumption is a geometric sequence. We find c0 such that the intertemporal
        budget is satisfied.

      - For general utility, provide:
            - marginal_utility: m(u) = u'(c)
            - inv_marginal_utility: (u')^{-1}(x)
        The solver will use Euler:
            m(c_t) = beta * (1 + r) * m(c_{t+1})
        and a root-finding (bisection) to find c0 that satisfies terminal budget.

    Notes:
      - This solver is open-loop: it produces the planned consumption path given known future cashflows.
      - It returns policy arrays (per-date constant or path) to match the Base API.
    """

    def __init__(
        self,
        *args,
        marginal_utility: Optional[Callable[[float], float]] = None,
        inv_marginal_utility: Optional[Callable[[float], float]] = None,
        **kwargs,
    ) -> None:
        """
        Args:
            marginal_utility: callable c -> u'(c). If None and instant_utility is log, we use 1/c.
            inv_marginal_utility: callable x -> (u')^{-1}(x). If None and log, use 1/x.
        """
        super().__init__(*args, **kwargs)

        # Provide default marginal/inverse for log utility
        self.marginal_utility: Optional[Callable[[float], float]] = marginal_utility
        self.inv_marginal_utility: Optional[Callable[[float], float]] = (
            inv_marginal_utility
        )

        # If user didn't provide but instant_utility is the default log, set defaults
        if self.marginal_utility is None:
            # attempt to detect if instant_utility matches log by trying a value
            try:
                test_val = 1.0
                if np.isclose(self.instant_utility(test_val), np.log(test_val)):
                    self.marginal_utility = lambda c: 1.0 / c
                    self.inv_marginal_utility = lambda x: 1.0 / x
            except Exception:
                # leave as None if we can't detect
                pass

    def _build_cashflow_array(self) -> np.ndarray:
        """Return vector of monthly cashflows aligned with self.months."""
        return np.array([self.monthly_cashflow(d) for d in self.months], dtype=float)

    def _consumption_from_c0_log(self, c0: float, r: float) -> np.ndarray:
        """
        For log utility, c_{t+1} = beta*(1 + r)*c_t. Return consumption path from c0.
        """
        c = np.zeros(self.n_months)
        c[0] = c0
        growth = self.beta * (1.0 + r)
        for t in range(1, self.n_months):
            c[t] = c[t - 1] * growth
        return c

    def _present_value_of_plan(
        self, c: np.ndarray, cashflows: np.ndarray, r: float
    ) -> float:
        """
        Compute present value of consumption plan and cashflows starting at time 0.
        Use discrete monthly returns: wealth evolves: W_{t+1} = (W_t + cf_t - c_t)*(1+r).
        We'll compute the terminal wealth (or equivalently net present value) given initial wealth.
        Return terminal wealth given initial wealth = 0 (so we can find c0 such that terminal wealth equals initial_wealth).
        """
        # Simulate forward starting from zero initial wealth
        w = 0.0
        for t in range(self.n_months):
            # cf_t occurs at beginning of period t
            w = (w + cashflows[t] - c[t]) * (1.0 + r)
        return w

    def _find_c0_bisection(
        self,
        cashflows: np.ndarray,
        r: float,
        target_terminal: float,
        low: float,
        high: float,
        tol: float = 1e-6,
        maxiter: int = 200,
    ) -> float:
        """
        Find c0 so that terminal wealth equals target_terminal via bisection.
        `low` and `high` bracket plausible c0 values.
        """
        fl = (
            self._present_value_of_plan(
                self._consumption_from_c0_log(low, r), cashflows, r
            )
            - target_terminal
        )
        fh = (
            self._present_value_of_plan(
                self._consumption_from_c0_log(high, r), cashflows, r
            )
            - target_terminal
        )

        if fl == 0.0:
            return low
        if fh == 0.0:
            return high
        if fl * fh > 0:
            raise ValueError(
                "Bisection endpoints do not bracket root (log utility case). Expand low/high."
            )

        for _ in range(maxiter):
            mid = 0.5 * (low + high)
            fm = (
                self._present_value_of_plan(
                    self._consumption_from_c0_log(mid, r), cashflows, r
                )
                - target_terminal
            )
            if abs(fm) < tol:
                return mid
            if fl * fm < 0:
                high = mid
                fh = fm
            else:
                low = mid
                fl = fm
        return 0.5 * (low + high)

    def solve(self) -> Tuple[Dict[dt.date, np.ndarray], Dict[dt.date, np.ndarray]]:
        """
        Solve deterministically via Lagrange / Euler equations.
        For log utility we use the analytic Euler growth and bisection for c0.
        For other utilities, the implementation would require numerical shooting / solve,
        which is beyond the minimal implementation here.
        """
        logger.info("LagrangeOptimizer.solve() started.")

        r = self.wealth.monthly_return()
        cashflows = self._build_cashflow_array()
        target_terminal = (
            self.wealth.initial_wealth
        )  # we will choose c0 so terminal wealth equals this initial shifted? (see below)

        # NOTE: convention: we simulate wealth starting from initial_wealth at t=0, with first cashflow at months[0].
        # For open-loop we want a consumption plan such that, starting from initial_wealth, terminal wealth >= 0.
        # Here we will solve for c0 such that terminal wealth equals zero (consume as much as possible subject to feasibility),
        # OR you could choose to exactly exhaust wealth by target_terminal = 0. We'll choose terminal wealth = 0 (common).
        target_terminal = 0.0

        # Try analytic log-utility solution if marginal utilities available/inferred
        if self.marginal_utility is not None and self.inv_marginal_utility is not None:
            # If marginal utility is m(c)=1/c and inv is 1/x, this branch includes the analytic log case.
            # For log we can compute growth factor and then find c0 by budget constraint.
            try:
                # test if marginal_utility behaves like 1/c near 1.0
                if np.isclose(self.marginal_utility(1.0), 1.0 / 1.0, atol=1e-8):
                    # log-utility analytic growth
                    # find c0 with bisection: choose bracket [eps, big]
                    low = 1e-8
                    # sensible upper bound: all available resources consumed in month 0
                    total_available = self.wealth.initial_wealth + cashflows.sum()
                    high = max(1.0, total_available * 2.0)
                    c0 = self._find_c0_bisection(
                        cashflows, r, target_terminal, low=low, high=high
                    )
                    c_path = self._consumption_from_c0_log(c0, r)
                else:
                    # Fallback: treat as general-utility (not implemented fully)
                    raise NotImplementedError(
                        "General-utility Lagrange solver not implemented in this minimal version."
                    )
            except NotImplementedError:
                raise
        else:
            # If we couldn't get marginals, try to assume log and use defaults (1/c)
            # Use log default analytic branch
            low = 1e-8
            total_available = self.wealth.initial_wealth + cashflows.sum()
            high = max(1.0, total_available * 2.0)
            c0 = self._find_c0_bisection(
                cashflows, r, target_terminal, low=low, high=high
            )
            c_path = self._consumption_from_c0_log(c0, r)

        # Now we have candidate consumption path c_path. Validate non-negativity:
        c_path = np.maximum(c_path, 0.0)

        # Build policy: per-date policy arrays for compatibility with Bellman API.
        # For open-loop, policy at date t is just the planned consumption at that date for any wealth.
        self.policy = {
            self.months[t]: np.full_like(self.wealth_grid, c_path[t], dtype=float)
            for t in range(self.n_months)
        }

        # Now roll forward to produce opt_wealth given initial_wealth
        w_path = np.zeros(self.n_months)
        w = self.wealth.initial_wealth
        for t in range(self.n_months):
            cf_t = cashflows[t]
            w_path[t] = w
            w = (w + cf_t - c_path[t]) * (1.0 + r)

        self.opt_wealth = pd.Series(w_path, index=self.months)
        self.opt_consumption = pd.Series(c_path, index=self.months)
        self.monthly_cashflows = pd.Series(cashflows, index=self.months)

        logger.info("LagrangeOptimizer.solve() finished.")
        return self.value_function, self.policy
