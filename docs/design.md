# Wealth Planning Dynamic Programming - Design Document

## Overview

The `wealthplan-dp` package provides a framework for solving multi-period wealth planning problems using dynamic programming. It optimizes consumption and portfolio allocation decisions while accounting for income, expenses, pensions, and other financial considerations to maximize long-term financial well-being.

## Problem Formulation

### State Variables
- **Wealth (W_t)**: Total wealth at time t
- **Time (t)**: Current period in the planning horizon

### Control Variables
- **Consumption (c_t)**: Amount consumed in period t
- **Portfolio Weight (w_t)**: Fraction of investable wealth allocated to risky asset

### Dynamics

The wealth evolution follows:

```
W_{t+1} = (W_t - c_t + y_t - e_t) * [w_t * R_risky + (1-w_t) * R_f] + pension_t
```

Where:
- `y_t`: Income in period t
- `e_t`: Expenses (e.g., rent) in period t
- `R_risky`: Gross return on risky asset (1 + r_risky)
- `R_f`: Gross return on risk-free asset (1 + r_f)
- `pension_t`: Pension/retirement income in period t

### Objective

Maximize discounted lifetime utility:

```
max E[Σ_{t=0}^{T-1} β^t * u(c_t) + β^T * u(W_T)]
```

Where:
- `β`: Discount factor
- `u(c)`: Utility function (typically CRRA: u(c) = c^(1-γ)/(1-γ))
- `u(W_T)`: Terminal utility (bequest motive)

## Architecture

### Module Structure

```
wealthplan/
├── models.py          # Model definitions and dynamics
├── dp_solver.py       # Dynamic programming solver
├── direct_solver.py   # Direct transcription/NLP solver
└── utils.py           # Utilities and visualization
```

### Key Classes

#### 1. UtilityFunction
Implements various utility function forms:
- **CRRA** (Constant Relative Risk Aversion): Standard in financial economics
- **Log utility**: Special case of CRRA with γ=1
- **Quadratic**: For simple testing

Methods:
- `evaluate(consumption)`: Compute utility value
- `marginal_utility(consumption)`: Compute marginal utility

#### 2. WealthPlanModel
Encapsulates the wealth planning problem:
- Time horizon and discount factor
- Asset returns (risk-free and risky)
- Income, expense, and pension schedules
- Utility function specification

Methods:
- `wealth_dynamics(...)`: Compute next-period wealth
- `utility(consumption)`: Evaluate utility
- `set_income_schedule()`, `set_expense_schedule()`, etc.

#### 3. DPSolver
Solves via backward induction (Bellman equation):

**Algorithm:**
1. Discretize wealth state space into grid
2. Initialize terminal value function: V_T(W) = u(W)
3. For t = T-1 down to 0:
   - For each wealth level W in grid:
     - Optimize: V_t(W) = max_{c,w} [u(c) + β * V_{t+1}(W')]
     - Store optimal policies c*(W) and w*(W)
4. Use interpolation for continuous wealth values

Methods:
- `solve()`: Run backward induction
- `get_policy(wealth, t)`: Get optimal policy at given state
- `simulate_path(initial_wealth)`: Simulate optimal trajectory

#### 4. DirectSolver (Optional)
Alternative approach using direct transcription:
- Treats all decisions as optimization variables
- Formulates as single large NLP (nonlinear program)
- Uses scipy.optimize.minimize with SLSQP

**Advantages:**
- Can handle stochastic elements
- More flexible constraints
- May be faster for some problem structures

**Disadvantages:**
- No policy function (solution specific to initial wealth)
- May struggle with long horizons
- Requires good initial guess

## Implementation Details

### Interpolation
- Uses scipy's `interp1d` with linear interpolation
- Ensures value function continuity
- Handles boundary conditions via extrapolation

### Grid Design
- Wealth grid: Uniform spacing from W_min to W_max
- Portfolio grid: Uniform spacing from 0 to 1
- Consumption grid: Adaptive based on available wealth

### Optimization in DP
- Grid search over consumption and portfolio controls
- Evaluates Bellman equation at each grid point
- Selects maximum value

### Constraints
- Non-negativity: Wealth, consumption must be ≥ 0
- Portfolio bounds: Weight in [0, 1]
- Budget constraint: Enforced through dynamics

## Usage Patterns

### Basic Workflow

```python
from wealthplan import WealthPlanModel, DPSolver, UtilityFunction

# 1. Create model
model = WealthPlanModel(T=30, discount_factor=0.96)
model.set_income_schedule(income_array)

# 2. Create solver
solver = DPSolver(model, wealth_grid_size=100)

# 3. Solve
solver.solve()

# 4. Simulate
path = solver.simulate_path(initial_wealth=100.0)
```

### Customization Options

**Utility Function:**
```python
# Higher risk aversion
util = UtilityFunction(utility_type="CRRA", gamma=5.0)
model = WealthPlanModel(..., utility_function=util)
```

**Income Profiles:**
```python
from wealthplan.utils import create_income_profile

income = create_income_profile(
    T=30, 
    working_years=25,
    starting_income=50.0,
    income_growth=0.03
)
```

**Grid Resolution:**
```python
# Finer grid for more accuracy
solver = DPSolver(
    model,
    wealth_grid_size=200,
    consumption_grid_size=100,
    portfolio_grid_size=21
)
```

## Computational Complexity

### DP Solver
- **Time complexity:** O(T * N_W * N_c * N_w)
  - T: Time horizon
  - N_W: Wealth grid size
  - N_c: Consumption grid size  
  - N_w: Portfolio grid size

- **Space complexity:** O(T * N_W)
  - Stores value function and policies for each time period

### Direct Solver
- **Variables:** 2*T (consumption and portfolio for each period)
- **Constraints:** T+1 (wealth dynamics)
- Complexity depends on NLP solver (typically O(n³) per iteration)

## Extensions and Future Work

### Potential Enhancements

1. **Stochastic Returns:** Add uncertainty to asset returns
2. **Multiple Assets:** Extend to N-asset portfolio problem
3. **Labor Supply:** Endogenous work/leisure choice
4. **Housing:** Include home ownership decisions
5. **Taxes:** Progressive taxation and tax-advantaged accounts
6. **Insurance:** Life insurance, health insurance optimization
7. **Borrowing Constraints:** Credit limits and interest rates
8. **Transaction Costs:** Trading costs and rebalancing

### Performance Optimization

1. **Parallelization:** Parallelize grid search within each time step
2. **Adaptive Grids:** Refine grid around optimal solution
3. **Machine Learning:** Use neural networks to approximate value function
4. **Compiled Code:** Numba JIT compilation for critical loops

## References

1. Stokey, N. L., & Lucas, R. E. (1989). *Recursive Methods in Economic Dynamics*
2. Ljungqvist, L., & Sargent, T. J. (2018). *Recursive Macroeconomic Theory*
3. Merton, R. C. (1969). "Lifetime Portfolio Selection under Uncertainty"
4. Cocco, J. F., Gomes, F. J., & Maenhout, P. J. (2005). "Consumption and Portfolio Choice over the Life Cycle"

## License

MIT License - See LICENSE file for details
