import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import math
from wealthplan.optimizer.bellman_optimizer import create_grid


class DynamicGridBuilder:
    def __init__(
        self,
        T: float,
        delta_w: float,
        w_max: float,
        n_steps: int,
        L_t: np.ndarray,
        c_step: float,
        cf: np.nd.array,
        alpha: float = 0.2,
        beta:float = 0.005,
        max_grid_shift: float = 0.3,
    ):
        """
        W0      : initial wealth
        T       : time horizon
        delta_w : step size for wealth grid
        n_steps : number of time steps
        L_t     : deterministic wealth path (numpy array of length n_steps)
                  must satisfy L_t[0] = W0 and L_t[-1] = 0
        """
        assert len(L_t) == n_steps, "L_t length must match n_steps"
        assert (
            abs(L_t[-1] - 0.0) <= c_step
        ), f"L_t[-1] must be within one grid step (±{self.c_step}) of 0"

        self.T = T
        self.alpha = alpha
        self.beta = beta
        self.delta_w = delta_w
        self.w_max = w_max
        self.n_steps = n_steps
        self.L_t = L_t.astype(np.float32)

        self.cf = cf
        self.max_grid_shift = max_grid_shift

        # Time grid
        self.t = np.linspace(0, T, n_steps)

    def build_initial_grid(self):
        self.grid = []

        beta_upper = 4.0 / self.T * math.log(1 + self.alpha)
        beta_lower = 4.0 / self.T * math.log(1 - self.alpha)

        gamma_upper = beta_upper * self.t * (1.0 - self.t / self.T)
        gamma_lower = beta_lower * self.t * (1.0 - self.t / self.T)

        upper_boundary = self.L_t * np.exp(gamma_upper)
        lower_boundary = self.L_t * np.exp(gamma_lower)

        for i, (u, l) in enumerate(zip(upper_boundary, lower_boundary)):
            if 0 < i < len(upper_boundary) - 1:
                u += self.cf[i]

            if abs(u - l) > 10e-5:
                # Clip to avoid negative wealth
                w_values = create_grid(max(0.0, l), u, self.delta_w)
            else:
                w_values = np.array([l], dtype=np.float32)

            self.grid.append(w_values)

        return self.grid

    def extend_grid(self, simulations):
        """
        Extend or shrink grid boundaries based on Monte Carlo simulations.

        simulations: array of shape (n_steps, n_sim)
        boundary_threshold: fraction of sims at boundary to trigger expansion
        lower_percentile, upper_percentile: percentiles for normal-fit interior distribution
        """
        for t in range(1, self.n_steps - 1):
            sim_t = simulations[t]

            sim_t = sim_t[sim_t != 0.0]

            # If everything was zero, do nothing
            if sim_t.size == 0:
                continue

            grid_t = self.grid[t]

            lower_bound = grid_t[0]
            upper_bound = grid_t[-1]

            # Percentage of simulations at boundaries
            pct_lower = np.mean(sim_t <= lower_bound)
            pct_upper = np.mean(sim_t >= upper_bound)

            if pct_lower <= self.beta and pct_upper <= self.beta:
                continue

            if pct_lower > self.beta >= pct_upper:
                new_lower = np.min(sim_t) - pct_lower * self.max_grid_shift * self.w_max
                new_upper = upper_bound
            elif pct_lower <= self.beta < pct_upper:
                new_lower = lower_bound
                new_upper = np.max(sim_t) + pct_upper * self.max_grid_shift * self.w_max
            elif pct_lower > self.beta and pct_upper > self.beta:
                new_lower = np.min(sim_t) - pct_lower * self.max_grid_shift * self.w_max
                new_upper = np.max(sim_t) + pct_upper * self.max_grid_shift * self.w_max

            new_lower = max(0.0, new_lower)

            self.grid[t] = create_grid(new_lower, new_upper, self.delta_w)

        return self.grid

    @property
    def upper_bounds(self):
        return np.array([row[-1] for row in self.grid])

    @property
    def lower_bounds(self):
        return np.array([row[0] for row in self.grid])

    def plot_grid_bounds(self):
        plt.figure(figsize=(10, 5))
        # Shaded area between bounds
        plt.fill_between(
            self.t, self.lower_bounds, self.upper_bounds, color="lightblue", alpha=0.3
        )
        # Upper and lower bound lines
        plt.plot(self.t, self.upper_bounds, label="Upper Bound", color="blue")
        plt.plot(self.t, self.lower_bounds, label="Lower Bound", color="red")
        # Deterministic path
        plt.plot(
            self.t,
            self.L_t,
            label="Deterministic Path L(t)",
            color="black",
            linewidth=2,
        )

        plt.xlabel("Time")
        plt.ylabel("Wealth")
        plt.title("Dynamic Wealth Grid Bounds with Deterministic Path")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    n_steps = 60
    W0 = 40_000
    T = 5.0
    delta_w = 250.0

    # Example arbitrary deterministic path (satisfies W0 at t=0 and 0 at t=T)
    t_grid = np.linspace(0, T, n_steps)
    L_t = W0 * (1 - (t_grid / T) ** 2)  # quadratic decay instead of linear

    model = DynamicGridBuilder(T=T, delta_w=delta_w, n_steps=n_steps, L_t=L_t)

    # Access grid and plot
    print("Grid at t=0:", model.grid[0])
    print("Grid at t=T/2:", model.grid[n_steps // 2])
    print("Grid at t=T:", model.grid[-1])

    model.plot_grid_bounds()
