# Design Document

## Overview

This document outlines the design and architecture of the wealthplan-dp package, a deterministic, multi-period model for personal wealth planning using dynamic programming.

## Architecture

### Core Components

1. **Models** (`models.py`)
   - Model definitions for wealth planning problems
   - Utility functions (CRRA, exponential, etc.)
   - State dynamics and transition equations
   - Constraints and feasibility checks

2. **DP Solver** (`dp_solver.py`)
   - Bellman equation solver using backward induction
   - Value function interpolation for continuous states
   - Policy function extraction
   - Forward simulation capabilities

3. **Direct Solver** (`direct_solver.py`) - Optional
   - Direct transcription approach
   - Converts dynamic problem to NLP
   - Alternative to dynamic programming
   - Useful for validation and comparison

4. **Utils** (`utils.py`)
   - I/O helpers for saving/loading solutions
   - Visualization utilities
   - Data formatting and validation
   - Summary statistics

### Problem Formulation

The wealth planning problem is formulated as a discrete-time dynamic optimization:

```
max E[Σ β^t u(c_t)]
```

Subject to:
- Wealth dynamics: W_{t+1} = (W_t - c_t) * (1 + r_t)
- Non-negativity constraints
- Terminal conditions

### Solution Methods

#### Dynamic Programming (Backward Induction)
- Discretize state space (wealth)
- Solve Bellman equation period by period
- Use interpolation for continuous approximation
- Extract optimal policy functions

#### Direct Transcription (Optional)
- Formulate as large-scale NLP
- All periods solved simultaneously
- Use nonlinear optimization solvers
- Good for validation and small problems

## Examples

### Example 1: Consumption Only
Simple problem with only consumption decisions, no portfolio choice.

### Example 2: With Portfolio
Full problem including portfolio allocation between risky and risk-free assets.

## Testing

Unit tests cover:
- Model dynamics and constraints
- DP solver correctness
- Interpolation accuracy
- Solution validation

## Future Extensions

- Stochastic returns and income
- Multiple asset classes
- Labor income dynamics
- Housing decisions
- Healthcare costs
- Bequest motives
