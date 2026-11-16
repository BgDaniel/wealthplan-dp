# wealthplan-dp

Stochastic lifecycle planning: a least-squares Monte Carlo (LSMC) Bellman optimizer for the
monthly **consumption** and the **equity share** of the invested wealth, given salary, fixed
expenses, insurances, pensions incl. a Basisrente (Rürup), German taxes and Gompertz mortality.
Equity (ETF) follows a GBM, the rest earns a risk-free savings rate.

## Setup

```bash
poetry install --with app
poetry run python -m unittest discover -s backend/test -t backend -v
```

## App

```bash
poetry run streamlit run app/streamlit_app.py
```

* **Steering**: the current month: income, fixed costs (itemised), allowed consumption and
  how much to buy / sell of equity and savings account given your current holdings.
* **Runs**: pick a configuration, adjust the number of paths, start now or at a scheduled
  time. Runs compute in background processes (`results/_jobs/`), at most *n* in parallel;
  the table shows progress, remaining time and key figures.
* **Result**: key figures, figures (wealth, consumption, equity share, equity vs. savings
  holdings, monthly buy/sell per asset, saving/withdrawal, equity policy map, survival,
  regression R², depletion risk, taxes per year, Rürup vs. ETF plan, cashflow composition),
  runtimes, downloads.
* **Compare**: median curves, key figures and differing parameters of up to four runs.
* **Configurations**: the YAML files in `params/` as forms (add, copy, edit, delete), incl.
  Rürup plans and the tax parameters.
  Saving from the app rewrites the file without comments.
* **Cache**: cached regression coefficients; clear the cache.

## CLI

```bash
poetry run python -m wealthplan.run params/lifecycle_params.yaml
```

writes `results/<timestamp>_<run_config_id>/` with `summary.json`, `quantiles.csv`,
`cashflows.csv` (items per month), `policy.npz` (for `decide`), `params.yaml` and
`figures/*.png`. In Python:

```python
import yaml
from config.stochastic.lsmc.lsmc_config_mapper import LSMCConfigMapper
from wealthplan.optimizer.stochastic.lsmc.lsmc_bellman_optimizer import LSMCBellmanOptimizer

params = LSMCConfigMapper.map_yaml_to_params(yaml.safe_load(open("params/lifecycle_params.yaml")))
opt = LSMCBellmanOptimizer(**params)
opt.solve()
opt.summary(); opt.quantiles(); opt.plot()
```

## Model

Monthly steps t; W_t wealth, y_t net cashflow (income − fixed expenses), c_t consumption,
X_t = W_t + y_t − c_t ≥ 0 invested amount, a_t equity share, R_t equity gross return,
R_f savings gross return, p_t one-month survival probability, β monthly discount factor,
u CRRA utility with risk aversion γ:

    V_t(w) = max_{c, a} u(c) + β p_t E_t[ V_{t+1}( X (a R_t + (1 − a) R_f) ) ]
    V_{T−1}(w) = u(w + y_{T−1})             (consume everything in the last month)

**LSMC step.** For every regression path s, grid point x_j of the invested amount and equity
share a_k the realised value p_t V_{t+1}(x_j G_{k,s}, s) is regressed on a polynomial in the
path's log equity level (`WealthRegressor`). The best a_k per (x_j, s) gives the equity
share; the consumption is solved in closed form on each grid segment
(u'(c) = β · slope of the continuation value). The forward simulation re-evaluates the
regression on fresh **out-of-sample** paths with simulated death times.

**Regression degree.** With i.i.d. GBM returns the equity level carries no information
about future returns, so degree 0 (constant basis, i.e. the Monte Carlo mean) is the
default. Higher degrees overfit: in a test with 2 000 regression paths the out-of-sample
value dropped from −0.0765 (degree 0) to −0.0769 (1) and −0.0773 (3), with standard error
0.0004. A degree > 0 only makes sense for a market model with a state, e.g. mean reversion.

**Variance reduction.** The normal draws of the regression paths are moment matched per
month (mean 0, variance 1), so the comparison between equity shares is not distorted by
the random drift of a single month.

**Shortfall.** If w + y < 0 the fixed expenses cannot be paid: c = 0, x = 0 and each missing
euro costs `shortfall_penalty` utility.

**Cache.** The regression coefficients per month are stored in `.cache/lsmc_<hash>/`. The
hash covers every parameter of the backward induction plus a solver version, but not
the run name, the number of out-of-sample paths or the mortality seed. Repeated runs
only redo the forward simulation.

## Taxes (`wealthplan.tax.TaxModel`, YAML section `tax`)

* **Capital gains**: Abgeltungsteuer incl. Soli (26.375 %, configurable) on interest and on
  equity returns after the Teilfreistellung (30 %), charged when the return accrues (monthly,
  losses credited at once). Simplification: ignores the deferral advantage of unrealised gains
  and the Sparer-Pauschbetrag (slightly pessimistic for equity).
* **Pensions**: taxable part = Besteuerungsanteil of the retirement year for the public pension
  and the Rürup pension (2023 82.5 %, +0.5 %-points per year, 100 % from 2058),
  `taxable_earnings_share` (Ertragsanteil) for private, 1.0 for company pensions; minus
  102 € Pauschbetrag; tax by the §32a EStG 2025 tariff + Soli or a flat configurable rate.
* **Rürup contributions** are deducted at the configurable marginal tax rate while working,
  up to the remaining Basisvorsorge room (Höchstbetrag minus public pension contributions);
  the refund is booked monthly.
* Not modelled: health / long-term care insurance of pensioners, church tax (add it to the
  capital gains rate), Vorabpauschale.

## Inflation (YAML section `inflation`, `wealthplan.economy.inflation`)

All amounts of the configuration are in euros of the start date. With `rate` > 0 the salary
grows with inflation plus `salary_real_growth`, fixed expenses with inflation
(`expenses_indexed`), the public pension with `public_pension_indexation`; Rürup and private
pensions stay nominal (they lose real value). The optimizer works in **real euros of the start
date**: cashflows are divided by the price index, returns by the monthly inflation factor.
Taxes apply to nominal amounts (capital gains tax also on the inflation part of the return);
tariff brackets, allowances and the Basisvorsorge room grow with inflation if
`tax_brackets_indexed`. Results and figures are in real euros; the Steering page shows nominal
euros of the chosen month.

## Mortality (`stochastic.survival_model`)

`table`: `destatis_unisex` / `destatis_male` / `destatis_female` – Gompertz fits to the remaining
life expectancy at 40 and 65 of the Destatis period life table 2021/23 – or `custom` (b, c).
`age_shift` (years) moves the hazard: −3 ≈ cohort adjustment of the period table.

## Emergency reserve (YAML section `reserve`)

With `enabled`, at least `months_of_fixed_costs` × the month's fixed costs stay in the savings
account: the equity share is limited to 1 − reserve / invested amount (backward induction,
simulation and Steering page).

## Rürup (`ruerup_plans` under `total_pension`)

Constant (or growing) monthly contribution until retirement; each contribution minus the
acquisition costs is invested at the expected fund return minus the running costs
(deterministic); monthly pension = capital / 10 000 × pension factor (Rentenfaktor), lifelong,
no death benefit.

## Rürup vs. ETF (figure 12, `wealthplan.analysis.ruerup_vs_etf`)

Same net cost per month (contribution minus tax refund) into an ETF (equity return minus TER,
taxed as in the model); from retirement a level withdrawal that empties the ETF at the plan
end, against the Rürup pension net of the additional income tax it causes. Flows weighted with
the survival probability; the ETF capital left at death is shown separately (heirs).

## Validation against closed forms (`python -m wealthplan.validation`, app page Validation)

| Case | Closed form |
|---|---|
| deterministic consumption smoothing | c_{t+1} = (βR)^{1/γ} c_t, c_0 = W_0 / Σ (g/R)^t |
| constant income | c_0 = (W_0 + Σ y/R^t) / Σ (g/R)^t |
| tax on interest | as above with R = 1 + r_m (1 − τ) |
| inflation | as above with R = R_nom / (1+π)^{1/12} (real terms) |
| mortality without annuities (Yaari) | c_{t+1}/c_t = (β p_t R)^{1/γ} |
| Merton | equity share = argmax E[(aR + (1−a)R_f)^{1−γ}] (≈ (μ−r)/(γσ²)) |

The checks use 800 wealth grid points; the error of c_0 falls about quadratically with the grid
density (100 points: 2.9 %, 400: 0.2 %, 1600: 0.015 %). Single-month consumption growth
jitters because the continuation value is linear between grid points.

## Layout

```
app/streamlit_app.py            Streamlit app (theme in .streamlit/config.toml)
params/                         YAML configurations
backend/src/config/             YAML -> optimizer arguments (ConfigMapper, LSMCConfigMapper)
backend/src/result_cache/       per-month result cache keyed by parameter hash
backend/src/wealthplan/
  cashflows/                    salary, rent, groceries, insurances, pensions, Rürup
  tax/                          TaxModel: capital gains, pension income tax, Rürup refund
  optimizer/stochastic/lsmc/    LSMCBellmanOptimizer and its Numba kernels
  optimizer/stochastic/...      GBM, Gompertz survival, WealthRegressor
  run.py, plots.py, jobs.py     CLI / result folder, figures, background runs
backend/test/                   unittest suite
```
