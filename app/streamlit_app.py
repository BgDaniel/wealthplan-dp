"""Streamlit front end of wealthplan-dp: configurations, scheduled runs, results.

Start from the repository root:

    poetry install --with app
    poetry run streamlit run app/streamlit_app.py

Pages
* Steering: what to do this month: income, fixed costs, allowed consumption and how much to
  buy / sell of equity and savings account, given the current holdings.
* Runs: pick a stored configuration, optionally override the simulation sizes, start now or
  schedule a start time. Runs compute in background processes (wealthplan.jobs); the table
  shows all runs with progress, remaining time, key figures and a link to the result.
* Compare: median curves and key figures of up to four runs side by side (sensitivity).
* Result: key figures, figures (dark theme), runtimes, parameters, downloads of one run.
* Configurations: the YAML parameter files in params/ as forms: add, edit, copy, delete.
* Cache: size of the regression cache, clear it.
"""

from __future__ import annotations

import copy
import dataclasses
import datetime as dt
import json
import shutil
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import streamlit as st
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "src"))

from config.stochastic.lsmc.lsmc_config_mapper import TECHNICAL_DEFAULTS, LSMCConfigMapper  # noqa: E402
from wealthplan.tax.tax_model import TaxModel  # noqa: E402
from wealthplan.economy.inflation import InflationModel  # noqa: E402
from wealthplan.cashflows.payroll import PayrollModel  # noqa: E402
from wealthplan.catalog import DEFAULT_PARAMS_DIR, delete_config, list_configs, load_config, save_config  # noqa: E402
from wealthplan.jobs import (DEFAULT_JOBS_ROOT, cancel_job, delete_job, list_jobs, read_status,  # noqa: E402
                             submit_job)
from wealthplan.optimizer.stochastic.lsmc.lsmc_bellman_optimizer import decide  # noqa: E402
from wealthplan.plots import BLUE, PURPLE, THEMES, compare_runs  # noqa: E402

PARAMS_DIR = ROOT / DEFAULT_PARAMS_DIR
JOBS_ROOT = ROOT / DEFAULT_JOBS_ROOT
RESULTS_ROOT = ROOT / "results"
CACHE_ROOT = ROOT / ".cache"
STATE_LABELS = {"scheduled": "◷ scheduled", "queued": "◌ queued", "running": "● running", "done": "✓ done",
                "failed": "✕ failed"}
FIGURE_TITLES = {
    "01_wealth": "Wealth",
    "02_consumption": "Consumption",
    "03_equity_share": "Equity share",
    "04_holdings": "Equity vs. savings",
    "05_flows": "Monthly investment",
    "06_saving": "Saving / withdrawal",
    "07_equity_policy": "Equity policy",
    "08_survival": "Survival",
    "09_regression": "Regression R²",
    "10_depletion": "Depletion risk",
    "11_taxes": "Taxes",
    "12_ruerup": "Rürup contract",
}

FIGURE_HELP = {
    "01_wealth": "Liquid wealth (ETF + savings account) at the start of each month over the simulated paths "
                 "alive at that age: median line, 25–75 % and 5–95 % bands. Euros of the start year.",
    "02_consumption": "Optimal monthly consumption on top of the fixed costs. Dashed: net cashflow (net income − "
                      "fixed costs) as monthly average of the year (bonus spread over the year). Consumption below "
                      "the line means saving, above it means living from the holdings.",
    "03_equity_share": "Share of the invested wealth held in equity (ETF) after the monthly decision; the rest is "
                       "in the savings account (at least the emergency reserve).",
    "04_holdings": "Amount in equity and in the savings account after rebalancing (median and 25–75 %).",
    "05_flows": "Money moved into (+) or out of (−) each asset per month, averaged over each year: new savings "
                "while working, withdrawals in retirement. The initial reallocation of the start wealth is left "
                "out.",
    "06_saving": "Net cashflow − consumption, monthly average of each year: positive = saving, negative = "
                 "withdrawal from the holdings.",
    "07_equity_policy": "Optimal equity share as a function of age and invested wealth (median Rürup state). "
                        "Dashed: median invested wealth of the simulation.",
    "08_survival": "Probability to be alive at each age (Gompertz fit of the Destatis life table; simulated "
                   "share dotted).",
    "09_regression": "Quality of the LSMC regression of the continuation value on the regression state "
                     "(Rürup capital / pension); median R² over the wealth grid per month.",
    "10_depletion": "Among the paths alive: share with wealth below 6 months of fixed costs, share where the "
                    "fixed costs cannot be paid this month, and share that has been out of money at least once "
                    "(red dotted). Low wealth rises towards the plan end because the plan consumes the rest. In all "
                    "figures the ages from which ≥ 5 % of the paths have been out of money are hatched red.",
    "11_taxes": "Taxes per year: capital gains tax on interest and ETF returns (median and 25–75 %), income tax "
                "on pensions (incl. the Rürup part), Rürup tax refund as negative tax.",
    "12_ruerup": "Rürup contract capital on the simulated ETF paths until the pension start (median and bands) "
                 "and the resulting net Rürup pension.",
}

# Only what the theme config cannot do; selectors kept few (they may change with Streamlit).
CSS = """
<style>
  .block-container {padding-top: 2.4rem; max-width: 1440px;}
  .page-kicker {color: #b9a0f2; font-size: 0.78rem; font-weight: 600; letter-spacing: 0.08em;
                text-transform: uppercase; margin-bottom: -0.4rem;}
  .page-subtitle {color: #9aa3c7; margin: -0.5rem 0 1.6rem 0;}
  .facts {color: #9aa3c7; font-size: 0.86rem; line-height: 1.55;}
  .facts b {color: #e7e9f5; font-weight: 500;}
  div[data-testid="stMetric"] {background: #131a33; border: 1px solid #1f2747;
                               border-radius: 10px; padding: 0.9rem 1.1rem;}
  div[data-testid="stMetricLabel"] p {color: #9aa3c7; font-size: 0.8rem;}
  .action {background: #131a33; border: 1px solid #1f2747; border-left: 4px solid #a07aea;
           border-radius: 10px; padding: 0.9rem 1.1rem; margin-bottom: 0.6rem;}
  .action.equity {border-left-color: #2c6be0;}
  .action .label {color: #9aa3c7; font-size: 0.8rem;}
  .action .value {color: #e7e9f5; font-size: 1.6rem; font-weight: 600;}
  .action .detail {color: #9aa3c7; font-size: 0.85rem;}
</style>
"""


def _header(kicker: str, title: str, subtitle: str) -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(f'<div class="page-kicker">{kicker}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def _money_input(container: Any, label: str, value: float, help: str | None = None) -> float:
    """Euro amount input with thousands separators ("144,400"); accepts input with or without commas."""
    value = float(value)
    text = container.text_input(label, value=f"{value:,.0f}" if value.is_integer() else f"{value:,.2f}", help=help)
    try:
        return float(text.replace(",", "").replace(" ", "").replace("€", "") or 0.0)
    except ValueError:
        container.error(f"{label}: '{text}' is not a number")
        return float(value)


def _fmt_table(frame: pd.DataFrame, decimals: int = 0) -> pd.DataFrame:
    """Numbers as strings with comma thousands separators (for display tables)."""
    out = frame.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]) and not pd.api.types.is_bool_dtype(out[col]):
            out[col] = out[col].map(lambda v: "" if pd.isna(v) else f"{v:,.{decimals}f}")
    return out


def _fmt_duration(seconds: float | None) -> str:
    if seconds is None or seconds != seconds:
        return "–"
    seconds = int(round(seconds))
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _fmt_time(stamp: str | None) -> str:
    if not stamp:
        return ""
    return datetime.fromisoformat(stamp).strftime("%d.%m. %H:%M")


def _keur(value: float | None) -> str:
    if value is None or value != value:
        return "–"
    return f"{value / 1e3:,.0f} k€"


def _pct(value: float | None) -> str:
    if value is None or value != value:
        return "–"
    return f"{100 * value:.0f} %"


def _config_facts(data: dict[str, Any]) -> str:
    sim, sto = data["simulation_parameters"], data["stochastic"]
    tech = {**TECHNICAL_DEFAULTS, **data.get("technical", {})}
    gamma = data.get("utility_function", {}).get("gamma", 0.5)
    return (f'<div class="facts">start <b>{sim["start_date"]}</b> · end <b>{sim["end_date"]}</b> · retirement '
            f'<b>{sim["retirement_date"]}</b><br>wealth <b>{_keur(sim["initial_wealth"])}</b> · age '
            f'<b>{sim.get("current_age", "–")}</b> · γ <b>{gamma:g}</b><br>equity <b>{100 * sto["yearly_return"]:.1f} %</b>'
            f' ± <b>{100 * sto["sigma"]:.0f} %</b> · savings <b>{100 * sim["yearly_return_savings"]:.1f} %</b><br>'
            f'<b>{tech["n_wealth"]}</b> wealth points · <b>{tech["n_equity_shares"]}</b> equity shares · '
            f'degree <b>{tech["regression_degree"]}</b></div>')


# ------------------------------------------------------------------ page: runs


def _new_run() -> None:
    names = list_configs(PARAMS_DIR)
    with st.container(border=True):
        st.subheader("Schedule run")
        if not names:
            st.info("Create a configuration first.")
            return
        col_c, col_s, col_t = st.columns([1, 1, 1], gap="large")
        with col_c:
            name = st.selectbox("Configuration", names)
            data = load_config(name, PARAMS_DIR)
            st.markdown(_config_facts(data), unsafe_allow_html=True)
        sto = data["stochastic"]
        with col_s:
            c1, c2 = st.columns(2)
            n_sims = c1.number_input("Regression paths", value=int(sto["n_sims"]), min_value=10, step=500)
            n_oos = c2.number_input("Out-of-sample paths", value=int(sto.get("n_sims_oos", sto["n_sims"])),
                                    min_value=10, step=1000)
            seed = c1.number_input("Seed", value=int(sto.get("seed") or 0), min_value=0)
            max_parallel = c2.number_input("Parallel runs", value=2, min_value=1, max_value=8,
                                           help="at most this many runs compute at the same time")
            use_cache = st.toggle("Use cache", value=bool(data.get("technical", {}).get("use_cache", True)),
                                  help="reuse the regression coefficients of an identical earlier run")
        with col_t:
            when = st.radio("Start", ["now", "at"], horizontal=True,
                            format_func={"now": "now", "at": "scheduled"}.get)
            c1, c2 = st.columns(2)
            day = c1.date_input("Date", value=date.today(), disabled=when == "now")
            clock = c2.time_input("Time", value=(datetime.now() + dt.timedelta(hours=1)).time().replace(
                second=0, microsecond=0), disabled=when == "now")
        if st.button("Schedule run", type="primary", icon=":material/play_arrow:"):
            params = copy.deepcopy(data)
            params["stochastic"].update(n_sims=int(n_sims), n_sims_oos=int(n_oos), seed=int(seed))
            params.setdefault("technical", {})["use_cache"] = bool(use_cache)
            try:
                LSMCConfigMapper.map_yaml_to_params(copy.deepcopy(params))  # validate before scheduling
            except Exception as exc:  # noqa: BLE001 - shown to the user
                st.error(f"Invalid configuration: {exc}")
                return
            not_before = datetime.combine(day, clock).isoformat(timespec="minutes") if when == "at" else None
            job_dir = submit_job({"params": params, "config_name": name, "results_root": str(RESULTS_ROOT),
                                  "cache_root": str(CACHE_ROOT), "max_parallel": int(max_parallel),
                                  "not_before": not_before}, JOBS_ROOT)
            st.toast(f"Run {job_dir.name} {'scheduled' if not_before else 'started'}",
                     icon=":material/rocket_launch:")


def _jobs_frame(jobs: list[dict]) -> pd.DataFrame:
    rows = []
    for job in jobs:
        summ = job.get("summary") or {}
        rows.append({
            "Status": STATE_LABELS.get(job["state"], job["state"]),
            "Configuration": job.get("name", ""),
            "Progress": 100.0 * float(job.get("progress") or 0.0),
            "Step": job.get("step") if job["state"] != "failed" else (job.get("error") or "")[:80],
            "Elapsed": _fmt_duration(job.get("elapsed_s")),
            "Remaining": _fmt_duration(job.get("eta_s")) if job["state"] == "running" else "–",
            "Equity share start": _pct(summ.get("equity_share_start")),
            "Equity share retirement": _pct(summ.get("equity_share_retirement")),
            "Consumption start": f"{summ['consumption_start']:,.0f} €" if summ.get("consumption_start") else "–",
            "Wealth at retirement": _keur(summ.get("wealth_retirement_median")),
            "Cache": "hit" if summ.get("from_cache") else "",
            "Start": _fmt_time(job.get("not_before") or job.get("started") or job.get("created")),
            "Result": f"/result?run={job['id']}" if job["state"] == "done" else None,
        })
    return pd.DataFrame(rows)


@st.fragment(run_every="2s")
def _runs_table() -> None:
    jobs = list_jobs(JOBS_ROOT)
    st.subheader("Runs")
    if not jobs:
        st.caption("No runs yet – choose a configuration and click “Schedule run”.")
        return
    st.dataframe(
        _jobs_frame(jobs), hide_index=True, width="stretch",
        column_config={
            "Progress": st.column_config.ProgressColumn("Progress", min_value=0, max_value=100, format="%.0f %%"),
            "Result": st.column_config.LinkColumn("Result", display_text="open →"),
        },
    )


def _manage_runs() -> None:
    jobs = list_jobs(JOBS_ROOT)
    if not jobs:
        return
    with st.expander("Manage runs"):
        labels = {j["id"]: f"{j.get('name', '')} · {j['id']} · {j['state']}" for j in jobs}
        chosen = st.selectbox("Run", list(labels), format_func=labels.get)
        job = next(j for j in jobs if j["id"] == chosen)
        c1, c2, _ = st.columns([1, 1, 4])
        if job["state"] in ("scheduled", "queued", "running"):
            if c1.button("Cancel", icon=":material/stop:"):
                cancel_job(job["job_dir"])
                st.rerun()
        elif c2.button("Delete", icon=":material/delete:", help="removes the run entry, keeps the result folder"):
            delete_job(job["job_dir"])
            st.rerun()
        log = Path(job["job_dir"]) / "log.txt"
        if log.exists():
            st.code(log.read_text(encoding="utf-8", errors="replace")[-4000:] or "(empty)", language=None)


def runs_page() -> None:
    """Schedule a run of a stored configuration; table of all runs."""
    _header("Optimisation", "Runs", "Pick a configuration, adjust the simulation size and start now or at a "
            "scheduled time. Runs compute in the background, also when the browser is closed.")
    _new_run()
    _runs_table()
    _manage_runs()


# ------------------------------------------------------------------ page: compare

COMPARE_METRICS = {
    "wealth_p50": ("Wealth (median)", "€"),
    "consumption_p50": ("Consumption (median)", "€"),
    "equity_share_p50": ("Equity share (median)", "%"),
    "equity_amount_p50": ("Amount in equity (median)", "€"),
    "wealth_p05": ("Wealth (5 % quantile)", "€"),
}
COMPARE_KEYS = ["equity_share_start", "equity_share_retirement", "consumption_start", "consumption_retired_median",
                "wealth_retirement_median", "wealth_retirement_p05", "prob_shortfall", "value_oos"]


def _flatten(data: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(data, dict):
        out: dict[str, Any] = {}
        for k, v in data.items():
            out.update(_flatten(v, f"{prefix}{k}."))
        return out
    return {prefix[:-1]: json.dumps(data, default=str) if isinstance(data, list) else data}


def compare_page() -> None:
    """Median curves, key figures and differing parameters of several runs."""
    done = [j for j in list_jobs(JOBS_ROOT) if j["state"] == "done"]
    _header("Optimisation", "Compare", "Sensitivity: put up to four finished runs side by side, e.g. the same "
            "plan with different risk aversion γ, equity return or Rürup contribution.")
    if len(done) < 2:
        st.info("At least two finished runs are needed – schedule variants on the Runs page.")
        return
    labels = {j["id"]: f"{j['name']} · {j['id']}" for j in done}
    chosen = st.multiselect("Runs (max. 4)", list(labels), default=list(labels)[:2], format_func=labels.get,
                            max_selections=4)
    if len(chosen) < 2:
        st.caption("Select at least two runs.")
        return
    dirs = [Path(read_status(JOBS_ROOT / c)["run_dir"]) for c in chosen]
    params = [_flatten(yaml.safe_load((d / "params.yaml").read_text(encoding="utf-8"))) for d in dirs]
    keys = sorted(set().union(*params))
    diff = [k for k in keys if len({str(p.get(k)) for p in params}) > 1
            and not k.endswith("run_config_id")]
    names = [f"#{i + 1} {labels[c].split(' · ')[0]}" for i, c in enumerate(chosen)]
    if diff:
        short = {k: k.split(".")[-1] for k in diff}
        names = [f"#{i + 1} " + ", ".join(f"{short[k]}={p.get(k)}" for k in diff[:3]) for i, p in enumerate(params)]
    metric = st.radio("Curve", list(COMPARE_METRICS), horizontal=True,
                      format_func=lambda k: COMPARE_METRICS[k][0])
    frames = [pd.read_csv(d / "quantiles.csv", index_col=0) for d in dirs]
    title, unit = COMPARE_METRICS[metric]
    st.pyplot(compare_runs(frames, names, metric, title, unit), width="stretch")

    left, right = st.columns([1.2, 1], gap="large")
    with left:
        st.subheader("Key figures")
        summaries = [json.loads((d / "summary.json").read_text()) for d in dirs]
        def fmt(key: str, v: Any) -> str:
            if v is None or v != v:
                return "–"
            if key.startswith(("equity_share", "prob_")):
                return _pct(v)
            if key.startswith(("consumption", "wealth")):
                return _eur(v)
            return f"{v:.5f}"

        table = pd.DataFrame({n: [fmt(k, s.get(k)) for k in COMPARE_KEYS] for n, s in zip(names, summaries)},
                             index=COMPARE_KEYS)
        st.dataframe(table, width="stretch")
    with right:
        st.subheader("Differing parameters")
        if diff:
            st.dataframe(pd.DataFrame({n: [p.get(k) for k in diff] for n, p in zip(names, params)},
                                      index=diff).astype(str), width="stretch")
        else:
            st.caption("The runs have identical parameters.")


# ------------------------------------------------------------------ page: result


def result_page() -> None:
    """Key figures, figures, runtimes, parameters and downloads of one finished run."""
    done = [j for j in list_jobs(JOBS_ROOT) if j["state"] == "done"]
    if not done:
        _header("Optimisation", "Result", "No finished run yet.")
        return
    ids = [j["id"] for j in done]
    run_id = st.query_params.get("run")
    labels = {j["id"]: f"{j['name']} · {j['id']}" for j in done}
    chosen = st.selectbox("Run", ids, index=ids.index(run_id) if run_id in ids else 0, format_func=labels.get)
    st.query_params["run"] = chosen
    job = read_status(JOBS_ROOT / chosen)
    run_dir = Path(job["run_dir"])
    summary = json.loads((run_dir / "summary.json").read_text())

    st.caption("Amounts in real euros of the start date (today's purchasing power) if inflation is configured.")
    _header("Result", job["name"],
            f'{summary["start"]} – {summary["end"]} · age {summary["age_start"]:g} → retirement at '
            f'{summary["age_retirement"]:g} · {summary["n_sims_regression"]} regression + '
            f'{summary["n_sims_oos"]} out-of-sample paths')
    m = st.columns(6)
    m[0].metric("Equity share now", _pct(summary["equity_share_start"]),
                help="optimal equity share of the invested wealth in the first month")
    m[1].metric("Consumption now", f'{summary["consumption_start"]:,.0f} €',
                help="optimal consumption in the first month (on top of the fixed expenses)")
    m[2].metric("Equity share at retirement", _pct(summary["equity_share_retirement"]), help="median over paths")
    m[3].metric("Wealth at retirement", _keur(summary["wealth_retirement_median"]),
                help=f'median over paths; 5 % quantile {_keur(summary["wealth_retirement_p05"])}')
    m[4].metric("Consumption in retirement", f'{summary["consumption_retired_median"]:,.0f} €',
                help="median monthly consumption after retirement")
    age_oom = summary.get("out_of_money_age_5pct")
    m[5].metric("Out of money", _pct(summary["prob_shortfall"]),
                f"5 % of paths from age {age_oom:.0f}" if age_oom is not None and age_oom == age_oom else None,
                delta_color="off",
                help="share of paths where wealth + net cashflow could not pay the fixed costs at least once "
                     "(the model then sets consumption and holdings to 0 and does not carry the gap as debt); "
                     f"mean unpaid fixed costs {_eur(summary.get('unpaid_fixed_costs_mean', 0.0))}. "
                     "Figures hatch the ages from which ≥ 5 % of the paths have been out of money.")

    fig_dir = run_dir / "figures"
    figures = sorted(fig_dir.glob("*.png"))
    if figures:
        tabs = st.tabs([FIGURE_TITLES.get(f.stem, f.stem) for f in figures])
        for tab, fig in zip(tabs, figures):
            tab.image(str(fig), width="stretch")
            if fig.stem in FIGURE_HELP:
                tab.caption(FIGURE_HELP[fig.stem])

    left, right = st.columns([1.2, 1], gap="large")
    with left:
        st.subheader("Key figures")
        rows = [(k, f"{v:,.4f}" if isinstance(v, float) and abs(v) < 10 else f"{v:,.0f}" if isinstance(v, (int, float))
                 and not isinstance(v, bool) else v) for k, v in summary.items() if not k.startswith("runtime_")]
        st.dataframe(pd.DataFrame(rows, columns=["Quantity", "Value"]).astype({"Value": str}),
                     hide_index=True, width="stretch", height=420)
    with right:
        st.subheader("Runtimes")
        times = [(k[len("runtime_"):-2], v) for k, v in summary.items() if k.startswith("runtime_")]
        st.dataframe(pd.DataFrame(times, columns=["Step", "Seconds"]), hide_index=True, width="stretch",
                     column_config={"Seconds": st.column_config.NumberColumn(format="%.1f")})
        st.subheader("Downloads")
        files = [f for f in ("summary.json", "quantiles.csv", "params.yaml") if (run_dir / f).exists()]
        cols = st.columns(2)
        for k, f in enumerate(files):
            cols[k % 2].download_button(f, (run_dir / f).read_bytes(), f, width="stretch",
                                        icon=":material/download:")
        with st.expander("Run parameters"):
            st.code((run_dir / "params.yaml").read_text(encoding="utf-8"), language="yaml")


# ------------------------------------------------------------------ page: steering


def _eur(value: float) -> str:
    return f"{value:,.0f} €"


def _trade(amount: float) -> str:
    if abs(amount) < 0.5:
        return "no change"
    return f"buy {_eur(amount)}" if amount > 0 else f"sell {_eur(-amount)}"


def _action(label: str, value: str, detail: str, kind: str = "") -> str:
    return (f'<div class="action {kind}"><div class="label">{label}</div><div class="value">{value}</div>'
            f'<div class="detail">{detail}</div></div>')


def _allocation_figure(current: tuple[float, float], target: tuple[float, float]):
    t = THEMES["dark"]
    fig, ax = plt.subplots(figsize=(6.2, 2.5))
    fig.patch.set_facecolor(t["bg"])
    ax.set_facecolor(t["bg"])
    rows = ["target", "now"]
    for i, (eq, sav) in enumerate((target, current)):
        ax.barh(i, eq, color=BLUE, height=0.55, label="equity" if i == 0 else None)
        ax.barh(i, sav, left=eq, color=PURPLE, height=0.55, hatch="//", edgecolor=t["bg"],
                label="savings account" if i == 0 else None)
        total = eq + sav
        ax.text(total, i, f"  {_eur(total)}", va="center", color=t["fg"], fontsize=9)
    ax.set_yticks(range(2), rows, color=t["muted"])
    ax.tick_params(axis="x", colors=t["muted"])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1e3:,.0f} k€"))
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    ax.grid(axis="x", color=t["grid"], lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, labelcolor=t["fg"],
              fontsize=9)
    ax.set_xlim(0, max(sum(current), sum(target), 1.0) * 1.22)
    fig.tight_layout()
    return fig


STEERING_HELP = """
* **Net income** – what arrives on your account this month: salary (in the bonus month plus the
  bonus) or pensions, after income tax and social security, plus tax refunds (Rürup).
* **Fixed costs** – rent, groceries, electricity, telecommunication, insurances and the
  contributions to your pension contracts (incl. Rürup).
* **Net cashflow** – net income minus fixed costs: the money left before consumption.
* **Consumption allowed** – the optimal amount you may spend on top of the fixed costs.
  Net cashflow − consumption is invested (positive) or withdrawn from your holdings (negative).
* **Actions** – how much to buy / sell of equity and savings account so that after this month
  the optimal equity share and the emergency reserve are met.
* All amounts are nominal euros of the chosen month; the line below the figures converts them to
  the purchasing power of the start year.
"""


def _income_and_fixed(items: pd.Series, cats: dict[str, str]) -> tuple[float, float]:
    """Net income (salary / pensions after tax and social security, refunds) and fixed costs."""
    income_cats = ("Salary", "Pensions (gross)", "Tax refund", "Social security", "Income tax")
    if cats:
        income = float(sum(v for k, v in items.items() if cats.get(k) in income_cats))
        fixed = float(-sum(v for k, v in items.items() if cats.get(k) not in income_cats))
    else:
        income = float(items[items > 0].sum())
        fixed = float(-items[items < 0].sum())
    return income, fixed


def _signed(value: float) -> str:
    return f"{value:+,.0f} €"


def steering_page() -> None:
    """What to do this month: cashflows, allowed consumption, buy / sell per asset."""
    done = [j for j in list_jobs(JOBS_ROOT) if j["state"] == "done"
            and (Path(j["run_dir"]) / "policy.npz").exists()]
    if not done:
        _header("Optimisation", "Steering", "No finished run yet – schedule one on the Runs page.")
        return
    labels = {j["id"]: f"{j['name']} · {j['id']}" for j in done}
    c_run, c_month = st.columns([2, 1])
    chosen = c_run.selectbox("Run", list(labels), format_func=labels.get)
    run_dir = Path(read_status(JOBS_ROOT / chosen)["run_dir"])
    with np.load(run_dir / "policy.npz") as f:
        policy = {k: f[k] for k in f.files}
    months = [date.fromisoformat(str(m)) for m in policy["months"]]
    this_month = date.today().replace(day=1)
    # default: the coming month, compared with the current one
    current = months.index(this_month) if this_month in months else None
    default = min(current + 1, len(months) - 1) if current is not None else 0
    month = c_month.selectbox("Month", months, index=default, format_func=lambda d: d.strftime("%m/%Y"))
    t = months.index(month)
    all_items = pd.read_csv(run_dir / "cashflows.csv", index_col=0)
    items = all_items.iloc[t]
    quant = pd.read_csv(run_dir / "quantiles.csv", index_col=0).iloc[t]
    params = yaml.safe_load((run_dir / "params.yaml").read_text(encoding="utf-8"))

    _header("Steering", f"{month:%B %Y}", "What to do this month: given your current holdings the optimal "
            "policy of the run says how much you may consume and how much to buy or sell of equity and "
            "savings account. Fixed costs include insurances and pension contributions. Amounts in "
            "nominal euros of this month.")

    # the policy works in real euros of the start date; this page shows nominal euros of the month
    price = float(policy["price_index"][t]) if "price_index" in policy else 1.0
    if t == 0:
        eq_default = float(params["simulation_parameters"].get("initial_equity", 0.0))
        sav_default = float(params["simulation_parameters"]["initial_wealth"]) - eq_default
    else:
        eq_default = price * float(quant.get("equity_before_p50", 0.0))
        sav_default = price * float(quant.get("savings_before_p50", 0.0))
    with st.container(border=True):
        st.subheader("Your holdings at the start of the month")
        c1, c2, c3 = st.columns(3)
        eq_now = _money_input(c1, "Equity (ETFs) [€]", round(max(eq_default, 0.0), -2))
        sav_now = _money_input(c2, "Savings account [€]", round(max(sav_default, 0.0), -2))
        c3.markdown(f'<div class="facts">Defaults: {"configuration" if t == 0 else "median of the simulated paths"}'
                    f'<br>Total <b>{_eur(eq_now + sav_now)}</b></div>', unsafe_allow_html=True)
        # fund-linked Rürup: its contract value (before) / pension (after the start) is a state of
        # the policy and, in the payout phase, a path-dependent income
        state, extra_nominal = 1.0, 0.0
        if bool(policy.get("has_ruerup", False)):
            r_idx = int(policy["ruerup_index"])
            if t < r_idx:
                cap_default = price * float(quant.get("ruerup_capital_p50", 0.0) or 0.0)
                cap = _money_input(c1, "Rürup contract value [€]", round(max(cap_default, 0.0), -2), help="from your Rürup statement (Vertragsguthaben)")
                state = cap / price + 1.0
            else:
                gross_default = price * float(quant.get("ruerup_gross_pension_p50", 0.0) or 0.0)
                gross = _money_input(c1, "Rürup pension, gross [€/month]", float(round(max(gross_default, 0.0))))
                tp = next(cf for cf in LSMCConfigMapper.map_yaml_to_params(copy.deepcopy(params))["cashflows"]
                          if hasattr(cf, "stochastic_plans"))
                extra_nominal = float(tp.stochastic_net_payout(month, np.array([gross]))[0])
                state = gross / price + 1.0
                items = items.copy()
                items["Rürup pension (net, simulated)"] = extra_nominal
                c2.markdown(f'<div class="facts">net after the additional income tax: <b>{_eur(extra_nominal)}</b>'
                            f'</div>', unsafe_allow_html=True)

    d = decide(policy, t, (eq_now + sav_now) / price, state, extra_nominal / price)
    for key in ("net_cashflow", "resources", "shortfall", "consumption", "invested", "equity", "savings"):
        d[key] *= price  # back to nominal euros of the month
    reserve = price * float(policy["reserve"][t]) if "reserve" in policy else 0.0
    cat_file = run_dir / "cashflow_categories.json"
    cats = json.loads(cat_file.read_text()) if cat_file.exists() else {}
    cats["Rürup pension (net, simulated)"] = "Pensions (gross)"
    income, fixed = _income_and_fixed(items, cats)

    # comparison with the current month (same holdings and Rürup state)
    ref = current if current is not None and current != t else None
    deltas: dict[str, str | None] = {"income": None, "fixed": None, "net": None, "consumption": None}
    if ref is not None:
        ref_items = all_items.iloc[ref].copy()
        r_idx = int(policy["ruerup_index"]) if bool(policy.get("has_ruerup", False)) else len(months)
        same_phase = (ref < r_idx) == (t < r_idx)
        ref_extra = extra_nominal if same_phase and t >= r_idx else 0.0
        if ref_extra:
            ref_items["Rürup pension (net, simulated)"] = ref_extra
        ref_income, ref_fixed = _income_and_fixed(ref_items, cats)
        ref_price = float(policy["price_index"][ref]) if "price_index" in policy else 1.0
        ref_d = decide(policy, ref, (eq_now + sav_now) / ref_price, state if same_phase else 1.0,
                       ref_extra / ref_price)
        deltas = {"income": _signed(income - ref_income), "fixed": _signed(fixed - ref_fixed),
                  "net": _signed(d["net_cashflow"] - ref_d["net_cashflow"] * ref_price),
                  "consumption": _signed(d["consumption"] - ref_d["consumption"] * ref_price) if same_phase else None}

    m = st.columns(4)
    m[0].metric("Net income", _eur(income), deltas["income"],
                help="salary / bonus / pensions after income tax and social security, plus tax refunds")
    m[1].metric("Fixed costs", _eur(fixed), deltas["fixed"], delta_color="inverse",
                help="rent, living costs, insurances, pension contributions")
    m[2].metric("Net cashflow", _eur(d["net_cashflow"]), deltas["net"], help="net income − fixed costs")
    m[3].metric("Consumption allowed", _eur(d["consumption"]), deltas["consumption"],
                help="optimal discretionary consumption on top of the fixed costs")
    if ref is not None:
        st.caption(f"Changes compared with {months[ref]:%B %Y} (current month), same holdings.")
    with st.expander("What do these numbers mean?", expanded=False):
        st.markdown(STEERING_HELP)
    if price > 1.0005:
        start_year = months[0].year
        st.markdown(f'<div class="facts">In purchasing power of {start_year} (÷ price index <b>{price:.2f}</b>): '
                    f'net income <b>{_eur(income / price)}</b> · fixed costs <b>{_eur(fixed / price)}</b> · '
                    f'net cashflow <b>{_eur(d["net_cashflow"] / price)}</b> · consumption allowed '
                    f'<b>{_eur(d["consumption"] / price)}</b></div>', unsafe_allow_html=True)

    left, right = st.columns([1, 1.15], gap="large")
    with left:
        st.subheader("Actions")
        if d["shortfall"] > 0:
            st.error(f"The fixed costs exceed income and wealth by {_eur(d['shortfall'])}.")
        st.markdown(_action("Equity (ETFs)", _trade(d["equity"] - eq_now),
                            f"target {_eur(d['equity'])} · {100 * d['equity_share']:.0f} % of the invested "
                            f"wealth", "equity"), unsafe_allow_html=True)
        reserve_note = f" · incl. emergency reserve {_eur(reserve)}" if reserve > 0 else ""
        st.markdown(_action("Savings account", _trade(d["savings"] - sav_now),
                            f"target {_eur(d['savings'])} · {100 * (1 - d['equity_share']):.0f} % of the invested "
                            f"wealth{reserve_note}"), unsafe_allow_html=True)
        st.markdown(_action("Invested after this month", _eur(d["invested"]),
                            f"holdings {_eur(eq_now + sav_now)} + net cashflow {_eur(d['net_cashflow'])} − "
                            f"consumption {_eur(d['consumption'])}"), unsafe_allow_html=True)
        st.pyplot(_allocation_figure((eq_now, sav_now), (d["equity"], d["savings"])), width="stretch")
    with right:
        st.subheader("Cashflows of the month")
        table = pd.DataFrame({"Item": items.index, "Amount [€]": items.values})
        table = table[table["Amount [€]"].abs() > 0.005].sort_values("Amount [€]", ascending=False)
        st.dataframe(_fmt_table(table, 2), hide_index=True, width="stretch", height=min(38 + 35 * len(table), 640))


# ------------------------------------------------------------------ page: configurations

PRIVATE_COLUMNS = ["name", "monthly_contribution", "monthly_payout_brutto", "taxable_earnings_share",
                   "contribution_growth_rate", "start_date"]


def _as_date(value: Any) -> date:
    return value if isinstance(value, date) else date.fromisoformat(str(value))


def _private_frame(plans: list[dict]) -> pd.DataFrame:
    rows = []
    for p in plans:
        dynamic = "contribution_growth_rate" in p
        rows.append({"name": p.get("name", ""),
                     "monthly_contribution": float(p.get("initial_monthly_contribution" if dynamic
                                                         else "monthly_contribution", 0.0)),
                     "monthly_payout_brutto": float(p.get("monthly_payout_brutto", 0.0)),
                     "taxable_earnings_share": float(p.get("taxable_earnings_share", 1.0)),
                     "contribution_growth_rate": float(p["contribution_growth_rate"]) if dynamic else None,
                     "start_date": _as_date(p["start_date"]) if dynamic and p.get("start_date") else None})
    return pd.DataFrame(rows, columns=PRIVATE_COLUMNS)


def _private_plans(frame: pd.DataFrame, default_start: date) -> list[dict]:
    plans = []
    for r in frame.to_dict("records"):
        if not r.get("name"):
            continue
        growth = r.get("contribution_growth_rate")
        if growth is not None and growth == growth:
            start = r.get("start_date")
            start = default_start if start is None or start != start else _as_date(start)
            plans.append({"name": r["name"], "initial_monthly_contribution": float(r["monthly_contribution"]),
                          "monthly_payout_brutto": float(r["monthly_payout_brutto"]),
                          "taxable_earnings_share": float(r["taxable_earnings_share"]),
                          "contribution_growth_rate": float(growth), "start_date": start})
        else:
            plans.append({"name": r["name"], "monthly_contribution": float(r["monthly_contribution"] or 0.0),
                          "monthly_payout_brutto": float(r["monthly_payout_brutto"]),
                          "taxable_earnings_share": float(r["taxable_earnings_share"])})
    return plans


RUERUP_COLUMNS = ["name", "monthly_contribution", "contribution_growth", "start_date", "fund_linked",
                  "expected_return", "running_costs", "acquisition_costs", "pension_factor", "initial_capital"]
RUERUP_DEFAULTS = {"contribution_growth": 0.0, "expected_return": 0.06, "running_costs": 0.012,
                   "acquisition_costs": 0.04, "pension_factor": 28.0, "initial_capital": 0.0}


def _ruerup_frame(plans: list[dict], default_start: date) -> pd.DataFrame:
    rows = [{"name": p.get("name", "Rürup"), "monthly_contribution": float(p["monthly_contribution"]),
             "fund_linked": bool(p.get("fund_linked", True)),
             **{k: float(p.get(k, v)) for k, v in RUERUP_DEFAULTS.items()},
             "start_date": _as_date(p.get("start_date", default_start))} for p in plans]
    return pd.DataFrame(rows, columns=RUERUP_COLUMNS)


def _ruerup_plans(frame: pd.DataFrame, default_start: date) -> list[dict]:
    plans = []
    for r in frame.to_dict("records"):
        if not r.get("name") or r.get("monthly_contribution") is None or r["monthly_contribution"] != \
                r["monthly_contribution"]:
            continue
        start = r.get("start_date")
        plan = {"name": r["name"], "monthly_contribution": float(r["monthly_contribution"]),
                "start_date": default_start if start is None or start != start else _as_date(start),
                "fund_linked": True if r.get("fund_linked") is None else bool(r["fund_linked"])}
        for k, v in RUERUP_DEFAULTS.items():
            value = r.get(k)
            plan[k] = float(v if value is None or value != value else value)
        plans.append(plan)
    return plans


def _config_form(name: str, data: dict[str, Any], is_new: bool) -> None:
    sim = data["simulation_parameters"]
    life = data.get("lifecycle", {})
    pension = data.get("total_pension", {})
    sto = data["stochastic"]
    surv = sto["survival_model"]
    tech = {**TECHNICAL_DEFAULTS, **data.get("technical", {})}
    util = data.get("utility_function", {})
    tax = {**dataclasses.asdict(TaxModel()), **data.get("tax", {})}
    infl = {**{k: v for k, v in dataclasses.asdict(InflationModel()).items() if k != "start_date"},
            **(data.get("inflation") or {})}
    reserve_cfg = {"enabled": False, "months_of_fixed_costs": 6.0, **(data.get("reserve") or {})}

    with st.form(f"config_{name}", border=True):
        new_name = st.text_input("Name", name, help="file name in params/: letters, digits, _ - .")
        tab_sim, tab_sal, tab_cf, tab_pen, tab_tax, tab_infl, tab_mkt, tab_num = st.tabs(
            ["Person & horizon", "Salary & payroll", "Expenses", "Pensions", "Taxes", "Inflation & reserve",
             "Market & mortality", "Preferences & numerics"])

        with tab_sim:
            c1, c2, c3, c4 = st.columns(4)
            start = c1.date_input("Start", _as_date(sim["start_date"]))
            end = c2.date_input("End", _as_date(sim["end_date"]), max_value=date(2150, 1, 1))
            retirement = c3.date_input("End of working life", _as_date(sim["retirement_date"]),
                                       max_value=date(2150, 1, 1), help="last salary in the month before")
            age = c4.number_input("Age at start", value=float(sim.get("current_age", 40.0)), min_value=0.0,
                                  max_value=110.0, step=0.5)
            wealth0 = _money_input(c1, "Initial wealth [€]", float(sim["initial_wealth"]))
            savings = c2.number_input("Savings rate [%/yr]", value=100 * float(sim["yearly_return_savings"]),
                                      step=0.25, format="%.2f")
            beta = c3.number_input("Monthly discount factor β", value=float(sim["beta"]), min_value=0.9,
                                   max_value=1.0, step=0.0005, format="%.4f")
            equity0 = _money_input(c4, "thereof in equity [€]", float(sim.get("initial_equity", 0.0)), help="part of the initial wealth currently held in equity; rest: savings")

        sal = life.get("salary") or {}
        sal_mode = sal.get("mode", "gross" if "yearly_fixed" in sal else "net")
        pay = {**dataclasses.asdict(PayrollModel()), **(data.get("payroll") or {})}
        with tab_sal:
            c1, c2, c3, c4 = st.columns(4)
            salary_mode = c1.radio("Amounts", ["gross", "net"], index=0 if sal_mode == "gross" else 1, horizontal=True,
                            help="gross: converted to net by the payroll below; net: as paid out")
            months = list(range(1, 13))
            bonus_month = c2.selectbox("Bonus paid in", months, index=int(sal.get("bonus_month", 4)) - 1,
                                       format_func=lambda m: date(2000, m, 1).strftime("%B"))
            fixed_gross = _money_input(c3, "Fixed gross salary [€/year]", float(sal.get("yearly_fixed", 0.0)), help="paid in 12 monthly amounts (mode gross)")
            bonus_gross = _money_input(c4, "Bonus gross [€/year]", float(sal.get("yearly_bonus", 0.0)), help="deterministic (mode gross)")
            net_month = _money_input(c3, "Net salary [€/month]", float(sal.get("monthly_net", sal.get("monthly_amount", 0.0))), help="mode net")
            net_bonus = _money_input(c4, "Net bonus [€/year]", float(sal.get("bonus_net", 0.0)), help="mode net")
            st.caption("Payroll (mode gross): employee contributions up to the ceilings, income tax §32a 2025 + "
                       "Soli + church tax on gross − Werbungskosten − Vorsorgeaufwendungen. Values 2025; ceilings "
                       "and allowances grow with inflation if tax brackets are indexed.")
            c1, c2, c3, c4 = st.columns(4)
            health = c1.selectbox("Health insurance", ["statutory", "private"],
                                  index=0 if pay["health_insurance"] == "statutory" else 1)
            health_rate = c2.number_input("Health rate employee [%]", value=100 * float(pay["health_rate"]),
                                          step=0.05, format="%.2f", help="7.3 % + half the Zusatzbeitrag")
            care_rate = c3.number_input("Care rate employee [%]", value=100 * float(pay["care_rate"]), step=0.1,
                                        format="%.2f", help="1.8 %, childless 2.4 %")
            church = c4.selectbox("Church tax", [0.0, 0.08, 0.09],
                                  index=[0.0, 0.08, 0.09].index(float(pay["church_tax_rate"]))
                                  if float(pay["church_tax_rate"]) in (0.0, 0.08, 0.09) else 0,
                                  format_func=lambda v: "none" if v == 0 else f"{100 * v:.0f} %")
            premium = _money_input(c1, "Private premium [€/month]", float(pay["private_premium_monthly"]))
            subsidy = _money_input(c2, "Employer subsidy [€/month]", float(pay["private_employer_subsidy_monthly"]))
            pension_rate = c3.number_input("Pension + unemployment rate [%]",
                                           value=100 * (float(pay["pension_rate"]) + float(pay["unemployment_rate"])),
                                           step=0.1, format="%.2f", help="9.3 % + 1.3 %")
            werbung = _money_input(c4, "Werbungskosten [€/year]", float(pay["werbungskosten"]))
            ceil_rv = _money_input(c1, "Ceiling RV/AV [€/year]", float(pay["pension_ceiling"]))
            ceil_kv = _money_input(c2, "Ceiling KV/PV [€/year]", float(pay["health_ceiling"]))
            basis_max = _money_input(c3, "Höchstbetrag Basisvorsorge [€/year]", float(pay["basis_pension_max"]))

        with tab_cf:
            c1, c2, c3 = st.columns(3)
            rent = _money_input(c2, "Rent [€/month]", float(life.get("rent", {}).get("monthly_amount", 0.0)))
            groceries = _money_input(c3, "Groceries [€/month]", float(life.get("groceries", {}).get( "monthly_amount", 0.0)))
            electricity = _money_input(c1, "Electricity [€/month]", float(life.get("electricity", {}).get( "monthly_amount", 0.0)))
            telecom = _money_input(c2, "Telecommunication [€/month]", float(life.get( "telecommunication", {}).get("monthly_amount", 0.0)))
            st.caption("Other expenses with a time window (e.g. voluntary health insurance after an early "
                       "retirement); empty dates = whole horizon")
            other = st.data_editor(
                pd.DataFrame([{"name": e.get("name", ""), "monthly_amount": float(e.get("monthly_amount", 0.0)),
                               "start_date": _as_date(e["start_date"]) if e.get("start_date") else None,
                               "end_date": _as_date(e["end_date"]) if e.get("end_date") else None}
                              for e in life.get("other_expenses", []) or []],
                             columns=["name", "monthly_amount", "start_date", "end_date"]),
                num_rows="dynamic", width="stretch", key=f"other_{name}",
                column_config={"monthly_amount": st.column_config.NumberColumn("monthly amount [€]", min_value=0.0),
                               "start_date": st.column_config.DateColumn("start"),
                               "end_date": st.column_config.DateColumn("end")})
            st.caption("Insurances (frequency M = monthly, Y = yearly in January)")
            insurances = st.data_editor(
                pd.DataFrame(life.get("insurances", []), columns=["name", "amount", "frequency"]),
                num_rows="dynamic", width="stretch", key=f"ins_{name}",
                column_config={"amount": st.column_config.NumberColumn("amount [€]", min_value=0.0),
                               "frequency": st.column_config.SelectboxColumn("frequency", options=["M", "Y"])})

        with tab_pen:
            st.caption("Private / company pensions: contribution until retirement, payout after. With a "
                       "growth rate the contribution grows yearly from the start date.")
            private = st.data_editor(_private_frame(pension.get("private_pension_plans", [])), num_rows="dynamic",
                                     width="stretch", key=f"priv_{name}",
                                     column_config={"start_date": st.column_config.DateColumn("start_date")})
            pension_start = st.date_input(
                "Pension start", _as_date(pension.get("retirement_date", sim["retirement_date"])),
                max_value=date(2150, 1, 1), help="first payout month of the public, private and Rürup pensions")
            contrib_end = st.date_input(
                "Contributions end", _as_date(pension.get("contributions_end_date", retirement)),
                max_value=date(2150, 1, 1), help="last month of contributions to all pension contracts (exclusive); "
                                                 "before the pension start for an early retirement")
            public = _money_input(st, "Public pension, gross [€/month]", float(sum(p["monthly_payout_brutto"] for p in pension.get("public_pension_plan", []))))
            st.caption("Basisrente (Rürup): contributions until retirement (tax deductible), lifelong pension "
                       "= capital / 10 000 × pension factor, taxed with the Besteuerungsanteil. fund_linked: the "
                       "capital follows the simulated ETF paths (expected_return is then unused). Rates as "
                       "fractions (0.06 = 6 %).")
            ruerup = st.data_editor(_ruerup_frame(pension.get("ruerup_plans", []), _as_date(sim["start_date"])),
                                    num_rows="dynamic", width="stretch", key=f"rue_{name}",
                                    column_config={"start_date": st.column_config.DateColumn("start_date")})

        with tab_tax:
            c1, c2, c3 = st.columns(3)
            cg_rate = c1.number_input("Capital gains tax [%]", value=100 * float(tax["capital_gains_tax_rate"]),
                                      min_value=0.0, max_value=60.0, step=0.25, format="%.3f",
                                      help="Abgeltungsteuer 25 % + Soli 5.5 % = 26.375 % (+ church tax)")
            exemption = c2.number_input("Partial exemption equity [%]",
                                        value=100 * float(tax["equity_partial_exemption"]), min_value=0.0,
                                        max_value=100.0, step=5.0, help="Teilfreistellung equity ETFs: 30 %")
            tax_int = c3.toggle("Tax interest", value=bool(tax["tax_savings_interest"]))
            tax_eq = c3.toggle("Tax equity returns", value=bool(tax["tax_equity_returns"]),
                               help="taxed when the return accrues (simplification, see README)")
            tax_mode = c1.selectbox("Income tax on pensions", ["tariff", "flat"],
                                index=0 if tax["income_tax_mode"] == "tariff" else 1,
                                format_func={"tariff": "§32a tariff 2025 + Soli", "flat": "flat rate"}.get)
            flat = c2.number_input("Flat tax rate in retirement [%]",
                                   value=100 * float(tax["income_tax_rate_retirement"]), min_value=0.0,
                                   max_value=60.0, step=1.0, help="used with “flat rate”")
            marginal = c1.number_input("Marginal tax rate while working [%]",
                                       value=100 * float(tax["marginal_tax_rate_working"]), min_value=0.0,
                                       max_value=47.5, step=1.0,
                                       help="Grenzsteuersatz: Rürup contributions are deducted at this rate")
            room = _money_input(c2, "Basisvorsorge deduction room [€/year]", float(tax["basis_pension_deduction_room"]), help="Höchstbetrag (2025: 29 344 €) minus employee + employer contributions " "to the public pension")
            allowance = _money_input(c3, "Pension allowance [€/year]", float(tax["pension_allowance_yearly"]), help="Werbungskosten-Pauschbetrag (102 €)")

        with tab_infl:
            st.caption("All amounts of the configuration are in euros of the start date. The optimisation runs in "
                       "real euros (today's purchasing power); taxes apply to nominal amounts.")
            c1, c2, c3 = st.columns(3)
            infl_rate = c1.number_input("Inflation [%/yr]", value=100 * float(infl["rate"]), min_value=-2.0,
                                        max_value=20.0, step=0.25, format="%.2f", help="0 disables inflation")
            real_growth = c2.number_input("Real salary growth [%/yr]", value=100 * float(infl["salary_real_growth"]),
                                          min_value=-5.0, max_value=10.0, step=0.25, format="%.2f",
                                          help="salary growth on top of inflation")
            pens_index = c3.number_input("Public pension adjustment [%/yr]",
                                         value=100 * float(infl["public_pension_indexation"]), min_value=-2.0,
                                         max_value=10.0, step=0.25, format="%.2f",
                                         help="Rentenanpassung; the public pension above is in today's euros")
            exp_indexed = c1.toggle("Expenses grow with inflation", value=bool(infl["expenses_indexed"]),
                                    help="rent, living costs, insurances")
            brackets_indexed = c2.toggle("Tax brackets indexed", value=bool(infl["tax_brackets_indexed"]),
                                         help="tariff brackets, allowances and Basisvorsorge room grow with inflation")
            st.divider()
            c1, c2, _ = st.columns(3)
            reserve_on = c1.toggle("Emergency reserve", value=bool(reserve_cfg["enabled"]),
                                   help="keep a reserve in the savings account (never invested in equity)")
            reserve_months = c2.number_input("Reserve [months of fixed costs]",
                                             value=float(reserve_cfg["months_of_fixed_costs"]), min_value=0.0,
                                             max_value=60.0, step=1.0)

        with tab_mkt:
            c1, c2, c3, c4 = st.columns(4)
            mu = c1.number_input("Equity return [%/yr]", value=100 * float(sto["yearly_return"]), step=0.25,
                                 format="%.2f")
            sigma = c2.number_input("Equity volatility [%/yr]", value=100 * float(sto["sigma"]), min_value=0.0,
                                    step=1.0)
            tables = ["destatis_unisex", "destatis_male", "destatis_female", "custom"]
            table = c3.selectbox("Mortality table", tables, index=tables.index(surv.get("table", "custom")),
                                 format_func={"destatis_unisex": "Destatis 2021/23 unisex",
                                              "destatis_male": "Destatis 2021/23 men",
                                              "destatis_female": "Destatis 2021/23 women",
                                              "custom": "custom Gompertz b, c"}.get,
                                 help="Gompertz fit to the period life table (remaining life expectancy at 40 / 65)")
            age_shift = c4.number_input("Age shift [years]", value=float(surv.get("age_shift", 0.0)), min_value=-15.0,
                                        max_value=15.0, step=1.0,
                                        help="-3 ≈ cohort adjustment: mortality of a 3 years younger person")
            b = c3.number_input("Gompertz b (custom)", value=float(surv.get("b", 9.5e-5)), min_value=1e-8,
                                format="%.2e")
            c = c4.number_input("Gompertz c (custom)", value=float(surv.get("c", 0.085)), min_value=1e-4,
                                format="%.4f")
            n_sims = c1.number_input("Regression paths", value=int(sto["n_sims"]), min_value=10, step=500)
            n_oos = c2.number_input("Out-of-sample paths", value=int(sto.get("n_sims_oos", sto["n_sims"])),
                                    min_value=10, step=1000)
            seed = c3.number_input("Seed", value=int(sto.get("seed") or 0), min_value=0)
            surv_seed = c4.number_input("Mortality seed", value=int(surv.get("seed") or 0), min_value=0)

        with tab_num:
            c1, c2, c3, c4 = st.columns(4)
            gamma = c1.number_input("Risk aversion γ", value=float(util.get("gamma", 0.5)), min_value=0.05,
                                    step=0.5)
            epsilon = c2.number_input("Consumption offset ε [€]", value=float(util.get("epsilon", 1.0)),
                                      min_value=1e-8, format="%.4g")
            penalty = c3.number_input("Shortfall penalty [utility/€]", value=float(tech["shortfall_penalty"]),
                                      min_value=0.0, format="%.4g")
            terminal = c4.toggle("Terminal penalty −w²", value=bool(tech["apply_terminal_penalty"]),
                                 help="off: consume everything in the last month")
            w_max = _money_input(c1, "Wealth grid max [€]", float(tech["w_max"]))
            n_wealth = c2.number_input("Wealth grid points", value=int(tech["n_wealth"]), min_value=20,
                                       max_value=2000)
            n_shares = c3.number_input("Equity share points", value=int(tech["n_equity_shares"]), min_value=2,
                                       max_value=101)
            max_share = c4.number_input("Max equity share", value=float(tech["max_equity_share"]), min_value=0.0,
                                        max_value=1.0, step=0.05)
            degree = c1.number_input("Regression degree", value=int(tech["regression_degree"]), min_value=0,
                                     max_value=5, help="0 = Monte Carlo mean; > 0 only helps if the market "
                                                       "model has a state (GBM has none)")
            use_cache = c2.toggle("Use cache", value=bool(tech.get("use_cache", True)))

        saved = st.form_submit_button("Save", type="primary", icon=":material/save:")

    if saved:
        new = copy.deepcopy(data)
        new["simulation_parameters"] = {
            "run_config_id": new_name.strip(), "initial_wealth": wealth0, "initial_equity": min(equity0, wealth0),
            "start_date": start, "end_date": end, "retirement_date": retirement, "current_age": age,
            "yearly_return_savings": savings / 100.0, "beta": beta}
        lifecycle: dict[str, Any] = {}
        if salary_mode == "gross" and fixed_gross + bonus_gross > 0:
            lifecycle["salary"] = {"mode": "gross", "yearly_fixed": fixed_gross, "yearly_bonus": bonus_gross,
                                   "bonus_month": int(bonus_month), "retirement_date": retirement}
        elif salary_mode == "net" and net_month + net_bonus > 0:
            lifecycle["salary"] = {"mode": "net", "monthly_net": net_month, "bonus_net": net_bonus,
                                   "bonus_month": int(bonus_month), "retirement_date": retirement}
        unemployment = float(pay["unemployment_rate"])
        new["payroll"] = {"pension_rate": pension_rate / 100.0 - unemployment, "unemployment_rate": unemployment,
                          "pension_ceiling": ceil_rv, "health_insurance": health, "health_rate": health_rate / 100.0,
                          "care_rate": care_rate / 100.0, "health_ceiling": ceil_kv,
                          "private_premium_monthly": premium, "private_employer_subsidy_monthly": subsidy,
                          "church_tax_rate": float(church), "werbungskosten": werbung,
                          "basis_pension_max": basis_max}
        for key, value in (("rent", rent), ("electricity", electricity), ("telecommunication", telecom),
                           ("groceries", groceries)):
            if value > 0:
                lifecycle[key] = {"monthly_amount": value}
        lifecycle["insurances"] = [{"name": r["name"], "amount": float(r["amount"]), "frequency": r["frequency"]}
                                   for r in insurances.to_dict("records") if r.get("name")]
        new["lifecycle"] = lifecycle
        lifecycle["other_expenses"] = [
            {"name": r["name"], "monthly_amount": float(r["monthly_amount"]),
             **({"start_date": _as_date(r["start_date"])} if r.get("start_date") not in (None, "") and
                r["start_date"] == r["start_date"] else {}),
             **({"end_date": _as_date(r["end_date"])} if r.get("end_date") not in (None, "") and
                r["end_date"] == r["end_date"] else {})}
            for r in other.to_dict("records") if r.get("name") and r.get("monthly_amount") == r.get("monthly_amount")]
        new["total_pension"] = {"retirement_date": pension_start, "contributions_end_date": min(contrib_end, pension_start),
                                "private_pension_plans": _private_plans(private, start),
                                "public_pension_plan": [{"monthly_payout_brutto": public}] if public > 0 else [],
                                "ruerup_plans": _ruerup_plans(ruerup, start)}
        new["tax"] = {"capital_gains_tax_rate": cg_rate / 100.0, "equity_partial_exemption": exemption / 100.0,
                      "tax_savings_interest": bool(tax_int), "tax_equity_returns": bool(tax_eq),
                      "income_tax_mode": tax_mode, "income_tax_rate_retirement": flat / 100.0,
                      "marginal_tax_rate_working": marginal / 100.0, "basis_pension_deduction_room": room,
                      "pension_allowance_yearly": allowance}
        new["stochastic"] = {"yearly_return": mu / 100.0, "sigma": sigma / 100.0, "n_sims": int(n_sims),
                             "n_sims_oos": int(n_oos), "seed": int(seed),
                             "survival_model": {"table": table, "age_shift": age_shift, "b": b, "c": c,
                                                "seed": int(surv_seed)}}
        new["inflation"] = {"rate": infl_rate / 100.0, "salary_real_growth": real_growth / 100.0,
                            "expenses_indexed": bool(exp_indexed), "public_pension_indexation": pens_index / 100.0,
                            "tax_brackets_indexed": bool(brackets_indexed)}
        new["reserve"] = {"enabled": bool(reserve_on), "months_of_fixed_costs": reserve_months}
        new["technical"] = {"w_max": w_max, "n_wealth": int(n_wealth), "n_equity_shares": int(n_shares),
                            "max_equity_share": max_share, "regression_degree": int(degree),
                            "shortfall_penalty": penalty, "apply_terminal_penalty": bool(terminal),
                            "use_cache": bool(use_cache)}
        new["utility_function"] = {"gamma": gamma, "epsilon": epsilon}
        try:
            if end <= start or not start <= retirement:
                raise ValueError("need start < end and start <= end of working life")
            if pension_start < retirement:
                raise ValueError("the pension start must not be before the end of working life")
            LSMCConfigMapper.map_yaml_to_params(copy.deepcopy(new))
            save_config(new_name.strip(), new, PARAMS_DIR)
            if not is_new and new_name.strip() != name:
                delete_config(name, PARAMS_DIR)
            st.toast(f"Configuration {new_name.strip()} saved", icon=":material/check:")
            st.rerun()
        except Exception as exc:  # noqa: BLE001 - shown to the user
            st.error(f"Invalid parameters: {exc}")


def _salary_label(sal: dict[str, Any]) -> str:
    if "yearly_fixed" in sal or sal.get("mode") == "gross":
        return f'{sal.get("yearly_fixed", 0):,.0f} + {sal.get("yearly_bonus", 0):,.0f} gross'
    monthly = sal.get("monthly_net", sal.get("monthly_amount", 0.0))
    return f'{12 * monthly + sal.get("bonus_net", 0):,.0f} net'


def configurations_page() -> None:
    """Stored YAML configurations as forms: add, edit, copy, delete."""
    _header("Master data", "Configurations", "Parameter files in params/. Edit an existing one or start a new "
            "one as a copy; the name is the file name.")
    names = list_configs(PARAMS_DIR)
    rows = []
    for n in names:
        d = load_config(n, PARAMS_DIR)
        sim, sto = d["simulation_parameters"], d["stochastic"]
        rows.append({"Name": n, "Start": str(sim["start_date"]), "End": str(sim["end_date"]),
                     "Retirement": str(sim["retirement_date"]), "Initial wealth [€]": sim["initial_wealth"],
                     "Salary + bonus [€/yr]": _salary_label(d.get("lifecycle", {}).get("salary") or {}),
                     "Equity μ [%]": 100 * sto["yearly_return"], "Equity σ [%]": 100 * sto["sigma"],
                     "γ": d.get("utility_function", {}).get("gamma"), "Paths": sto["n_sims"]})
    st.dataframe(_fmt_table(pd.DataFrame(rows)) if rows else pd.DataFrame(), hide_index=True, width="stretch")

    if not names:
        st.info("No configuration in params/.")
        return
    options = names + ["➕ New (copy)"]
    choice = st.selectbox("Edit", options)
    if choice == options[-1]:
        source = st.selectbox("Copy of", names)
        _config_form(f"{source}_copy", load_config(source, PARAMS_DIR), is_new=True)
    else:
        _config_form(choice, load_config(choice, PARAMS_DIR), is_new=False)
        if st.button("Delete", icon=":material/delete:"):
            delete_config(choice, PARAMS_DIR)
            st.rerun()


# ------------------------------------------------------------------ page: validation


def validation_page() -> None:
    """Run the closed-form checks of wealthplan.validation."""
    from wealthplan.validation import CHECKS, results_as_dicts

    _header("App", "Validation", "The optimizer against closed-form solutions: deterministic consumption "
            "smoothing (Euler equation and budget), constant income, tax on interest, inflation, mortality "
            "(Yaari) and the Merton equity share. Each check solves a small plan (about 15 s in total).")
    if st.button("Run checks", type="primary", icon=":material/fact_check:"):
        rows = []
        progress = st.progress(0.0)
        for i, (name, check) in enumerate(CHECKS.items()):
            rows.extend(results_as_dicts(check()))
            progress.progress((i + 1) / len(CHECKS), text=name)
        st.session_state["validation"] = rows
    rows = st.session_state.get("validation")
    if rows:
        passed = sum(r["passed"] for r in rows)
        (st.success if passed == len(rows) else st.error)(f"{passed} / {len(rows)} checks passed")
        frame = pd.DataFrame(rows)
        frame.insert(0, "", np.where(frame["passed"], "✓", "✕"))
        st.dataframe(frame.drop(columns="passed"), hide_index=True, width="stretch",
                     column_config={"expected": st.column_config.NumberColumn(format="%.6g"),
                                    "model": st.column_config.NumberColumn(format="%.6g"),
                                    "rel_error": st.column_config.NumberColumn("rel. error", format="%.2e"),
                                    "tolerance": st.column_config.NumberColumn(format="%.1e")})


# ------------------------------------------------------------------ page: cache


def cache_page() -> None:
    """Size of the regression cache; clear it."""
    _header("App", "Cache", "Regression coefficients per month, keyed by a hash of all parameters the "
            "backward induction depends on. A repeated run with the same parameters (only the out-of-sample "
            "paths changed) skips the backward induction.")
    folders = sorted(p for p in CACHE_ROOT.glob("*") if p.is_dir()) if CACHE_ROOT.exists() else []
    rows = [{"Entry": p.name, "Months": len(list(p.glob("*.pkl"))),
             "Size [MB]": sum(f.stat().st_size for f in p.glob("*")) / 1e6,
             "Modified": datetime.fromtimestamp(p.stat().st_mtime).strftime("%d.%m.%Y %H:%M")} for p in folders]
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     column_config={"Size [MB]": st.column_config.NumberColumn(format="%.1f")})
    else:
        st.caption("The cache is empty.")
    if rows and st.button("Clear cache", icon=":material/delete_sweep:"):
        shutil.rmtree(CACHE_ROOT)
        st.toast("Cache cleared", icon=":material/check:")
        st.rerun()


def main() -> None:
    """Page setup and navigation."""
    st.set_page_config(page_title="Wealthplan", page_icon=":material/savings:", layout="wide")
    st.logo(str(Path(__file__).with_name("logo.svg")), size="large")
    with st.sidebar:
        st.caption("Lifecycle consumption and equity share (LSMC Bellman)")
    nav = st.navigation({
        "Optimisation": [
            st.Page(steering_page, title="Steering", icon=":material/explore:", url_path="steering"),
            st.Page(runs_page, title="Runs", icon=":material/play_circle:", url_path="runs", default=True),
            st.Page(result_page, title="Result", icon=":material/insights:", url_path="result"),
            st.Page(compare_page, title="Compare", icon=":material/compare_arrows:", url_path="compare"),
        ],
        "Master data": [
            st.Page(configurations_page, title="Configurations", icon=":material/tune:", url_path="configurations"),
        ],
        "App": [
            st.Page(validation_page, title="Validation", icon=":material/fact_check:", url_path="validation"),
            st.Page(cache_page, title="Cache", icon=":material/storage:", url_path="cache"),
        ],
    })
    nav.run()


if __name__ == "__main__":  # streamlit runs the script as __main__
    main()
