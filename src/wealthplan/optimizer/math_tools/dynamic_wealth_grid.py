import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import math


class DynamicGridBuilder:
    def __init__(
        self,
        T: float,
        delta_w: float,
        n_steps: int,
        L_t: np.ndarray,
        c_step: float,
        alpha: float = 0.2
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
        assert abs(L_t[-1] - 0.0) <= c_step, \
            f"L_t[-1] must be within one grid step (±{self.c_step}) of 0"

        self.T = T
        self.alpha = alpha
        self.delta_w = delta_w
        self.n_steps = n_steps
        self.L_t = L_t.astype(np.float32)

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

        for u, l in zip(upper_boundary, lower_boundary):
            if u - l > 0:
                w_values = np.arange(l - self.delta_w, u + self.delta_w, self.delta_w)
                # Clip to avoid negative wealth
                w_values = np.clip(w_values, 0.0, None)
            else:
                w_values = np.array([l], dtype=np.float32)

            self.grid.append(w_values)

        return self.grid

    def extend_grid(self, simulations, lower_percentile=2, upper_percentile=2):
        """
        Extend or shrink grid boundaries based on Monte Carlo simulations.

        simulations: array of shape (n_steps, n_sim)
        boundary_threshold: fraction of sims at boundary to trigger expansion
        lower_percentile, upper_percentile: percentiles for normal-fit interior distribution
        """
        for t in range(self.n_steps):
            sim_t = simulations[t]
            grid_t = self.grid[t]

            lower_bound = grid_t[0]
            upper_bound = grid_t[-1]

            # Percentage of simulations at boundaries
            pct_lower = np.mean(sim_t <= lower_bound)
            pct_upper = np.mean(sim_t >= upper_bound)

            # --- STEP 1: Shrink boundaries only if current boundary is outside simulations ---
            if lower_bound < np.min(sim_t):
                new_lower = np.min(sim_t)  # shrink lower only if below all sims
            else:
                new_lower = lower_bound  # keep

            if upper_bound > np.max(sim_t):
                new_upper = np.max(sim_t)  # shrink upper only if above all sims
            else:
                new_upper = upper_bound

            # --- STEP 2: Fit normal distribution to interior points ---
            interior_sims = sim_t[(sim_t > new_lower) & (sim_t < new_upper)]

            if len(interior_sims) > 0:
                mu = np.mean(interior_sims)
                sigma = np.std(interior_sims)

                # Calculate percentiles from normal fit
                norm_lower = norm.ppf(lower_percentile / 100.0, loc=mu, scale=sigma)
                norm_upper = norm.ppf(upper_percentile / 100.0, loc=mu, scale=sigma)

                # Update boundaries based on normal fit
                new_lower = min(new_lower, norm_lower)
                new_upper = max(new_upper, norm_upper)
            else:
                # If all simulations are at boundary, expand a bit
                new_lower = new_lower - self.delta_w
                new_upper = new_upper + self.delta_w

            # Ensure at least one grid point
            if new_upper <= new_lower:
                new_upper = new_lower + self.delta_w

            # --- STEP 3: Update the grid ---
            self.grid[t] = np.arange(new_lower, new_upper + self.delta_w, self.delta_w)

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

    model = DynamicGridBuilder(
        T=T, delta_w=delta_w, n_steps=n_steps, L_t=L_t
    )

    # Access grid and plot
    print("Grid at t=0:", model.grid[0])
    print("Grid at t=T/2:", model.grid[n_steps // 2])
    print("Grid at t=T:", model.grid[-1])

    model.plot_grid_bounds()
