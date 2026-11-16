"""Streamlit front end of wealthplan-dp: configurations, scheduled runs, results.

Start from the repository root:

    poetry install --with app
    poetry run streamlit run app/streamlit_app.py

Pages
* Runs: pick a stored configuration, optionally override the simulation sizes, start now or
  schedule a start time. Runs compute in background processes (wealthplan.jobs); the table
  shows all runs with progress, remaining time, key figures and a link to the result.
* Result: key figures, figures (dark theme), runtimes, parameters, downloads of one run.
* Configurations: the YAML parameter files in params/ as forms: add, edit, copy, delete.
* Cache: size of the regression cache, clear it.
"""

from __future__ import annotations

import copy
import datetime as dt
import json
import shutil
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "src"))

from config.stochastic.lsmc.lsmc_config_mapper import TECHNICAL_DEFAULTS, LSMCConfigMapper  # noqa: E402
from wealthplan.catalog import DEFAULT_PARAMS_DIR, delete_config, list_configs, load_config, save_config  # noqa: E402
from wealthplan.jobs import (DEFAULT_JOBS_ROOT, cancel_job, delete_job, list_jobs, read_status,  # noqa: E402
                             submit_job)

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
    "04_equity_amount": "Equity amount",
    "05_saving": "Saving / withdrawal",
    "06_equity_policy": "Equity policy",
    "07_survival": "Survival",
    "08_regression": "Regression R²",
}

# Only what the theme config cannot do; selectors kept few (they may change with Streamlit).
CSS = """
<style>
  .block-container {padding-top: 2.4rem; max-width: 1440px;}
  .page-kicker {color: #6da7ec; font-size: 0.78rem; font-weight: 600; letter-spacing: 0.08em;
                text-transform: uppercase; margin-bottom: -0.4rem;}
  .page-subtitle {color: #9aa4b2; margin: -0.5rem 0 1.6rem 0;}
  .facts {color: #9aa4b2; font-size: 0.86rem; line-height: 1.55;}
  .facts b {color: #e6e9ef; font-weight: 500;}
  div[data-testid="stMetric"] {background: #121a27; border: 1px solid #1e2633;
                               border-radius: 10px; padding: 0.9rem 1.1rem;}
  div[data-testid="stMetricLabel"] p {color: #9aa4b2; font-size: 0.8rem;}
</style>
"""


def _header(kicker: str, title: str, subtitle: str) -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(f'<div class="page-kicker">{kicker}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)


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
    m[5].metric("Shortfall probability", _pct(summary["prob_shortfall"]),
                help="share of paths where the fixed expenses could not be paid at least once")

    fig_dir = run_dir / "figures"
    figures = sorted(fig_dir.glob("*.png"))
    if figures:
        tabs = st.tabs([FIGURE_TITLES.get(f.stem, f.stem) for f in figures])
        for tab, fig in zip(tabs, figures):
            tab.image(str(fig), width="stretch")

    left, right = st.columns([1.2, 1], gap="large")
    with left:
        st.subheader("Key figures")
        rows = [(k, v) for k, v in summary.items() if not k.startswith("runtime_")]
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


def _config_form(name: str, data: dict[str, Any], is_new: bool) -> None:
    sim = data["simulation_parameters"]
    life = data.get("lifecycle", {})
    pension = data.get("total_pension", {})
    sto = data["stochastic"]
    surv = sto["survival_model"]
    tech = {**TECHNICAL_DEFAULTS, **data.get("technical", {})}
    util = data.get("utility_function", {})

    with st.form(f"config_{name}", border=True):
        new_name = st.text_input("Name", name, help="file name in params/: letters, digits, _ - .")
        tab_sim, tab_cf, tab_pen, tab_mkt, tab_num = st.tabs(
            ["Person & horizon", "Income & expenses", "Pensions", "Market & mortality", "Preferences & numerics"])

        with tab_sim:
            c1, c2, c3, c4 = st.columns(4)
            start = c1.date_input("Start", _as_date(sim["start_date"]))
            end = c2.date_input("End", _as_date(sim["end_date"]), max_value=date(2150, 1, 1))
            retirement = c3.date_input("Retirement", _as_date(sim["retirement_date"]), max_value=date(2150, 1, 1))
            age = c4.number_input("Age at start", value=float(sim.get("current_age", 40.0)), min_value=0.0,
                                  max_value=110.0, step=0.5)
            wealth0 = c1.number_input("Initial wealth [€]", value=float(sim["initial_wealth"]), min_value=0.0,
                                      step=5000.0)
            savings = c2.number_input("Savings rate [%/yr]", value=100 * float(sim["yearly_return_savings"]),
                                      step=0.25, format="%.2f")
            beta = c3.number_input("Monthly discount factor β", value=float(sim["beta"]), min_value=0.9,
                                   max_value=1.0, step=0.0005, format="%.4f")

        with tab_cf:
            c1, c2, c3 = st.columns(3)
            salary = c1.number_input("Net salary [€/month]", value=float(life.get("salary", {}).get(
                "monthly_amount", 0.0)), min_value=0.0, step=100.0)
            rent = c2.number_input("Rent [€/month]", value=float(life.get("rent", {}).get("monthly_amount", 0.0)),
                                   min_value=0.0, step=50.0)
            groceries = c3.number_input("Groceries [€/month]", value=float(life.get("groceries", {}).get(
                "monthly_amount", 0.0)), min_value=0.0, step=50.0)
            electricity = c1.number_input("Electricity [€/month]", value=float(life.get("electricity", {}).get(
                "monthly_amount", 0.0)), min_value=0.0, step=5.0)
            telecom = c2.number_input("Telecommunication [€/month]", value=float(life.get(
                "telecommunication", {}).get("monthly_amount", 0.0)), min_value=0.0, step=5.0)
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
            public = st.number_input("Public pension, gross [€/month]",
                                     value=float(sum(p["monthly_payout_brutto"]
                                                     for p in pension.get("public_pension_plan", []))),
                                     min_value=0.0, step=50.0)

        with tab_mkt:
            c1, c2, c3, c4 = st.columns(4)
            mu = c1.number_input("Equity return [%/yr]", value=100 * float(sto["yearly_return"]), step=0.25,
                                 format="%.2f")
            sigma = c2.number_input("Equity volatility [%/yr]", value=100 * float(sto["sigma"]), min_value=0.0,
                                    step=1.0)
            b = c3.number_input("Gompertz b", value=float(surv["b"]), min_value=1e-7, format="%.2e")
            c = c4.number_input("Gompertz c", value=float(surv["c"]), min_value=1e-4, format="%.4f")
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
            w_max = c1.number_input("Wealth grid max [€]", value=float(tech["w_max"]), min_value=1e4, step=1e5)
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
            "run_config_id": new_name.strip(), "initial_wealth": wealth0, "start_date": start, "end_date": end,
            "retirement_date": retirement, "current_age": age, "yearly_return_savings": savings / 100.0,
            "beta": beta}
        lifecycle: dict[str, Any] = {}
        if salary > 0:
            lifecycle["salary"] = {"monthly_amount": salary, "retirement_date": retirement}
        for key, value in (("rent", rent), ("electricity", electricity), ("telecommunication", telecom),
                           ("groceries", groceries)):
            if value > 0:
                lifecycle[key] = {"monthly_amount": value}
        lifecycle["insurances"] = [{"name": r["name"], "amount": float(r["amount"]), "frequency": r["frequency"]}
                                   for r in insurances.to_dict("records") if r.get("name")]
        new["lifecycle"] = lifecycle
        new["total_pension"] = {"retirement_date": retirement,
                                "private_pension_plans": _private_plans(private, start),
                                "public_pension_plan": [{"monthly_payout_brutto": public}] if public > 0 else []}
        new["stochastic"] = {"yearly_return": mu / 100.0, "sigma": sigma / 100.0, "n_sims": int(n_sims),
                             "n_sims_oos": int(n_oos), "seed": int(seed),
                             "survival_model": {"b": b, "c": c, "seed": int(surv_seed)}}
        new["technical"] = {"w_max": w_max, "n_wealth": int(n_wealth), "n_equity_shares": int(n_shares),
                            "max_equity_share": max_share, "regression_degree": int(degree),
                            "shortfall_penalty": penalty, "apply_terminal_penalty": bool(terminal),
                            "use_cache": bool(use_cache)}
        new["utility_function"] = {"gamma": gamma, "epsilon": epsilon}
        try:
            if end <= start or not start <= retirement:
                raise ValueError("need start < end and start <= retirement")
            LSMCConfigMapper.map_yaml_to_params(copy.deepcopy(new))
            save_config(new_name.strip(), new, PARAMS_DIR)
            if not is_new and new_name.strip() != name:
                delete_config(name, PARAMS_DIR)
            st.toast(f"Configuration {new_name.strip()} saved", icon=":material/check:")
            st.rerun()
        except Exception as exc:  # noqa: BLE001 - shown to the user
            st.error(f"Invalid parameters: {exc}")


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
                     "Salary [€]": d.get("lifecycle", {}).get("salary", {}).get("monthly_amount", 0.0),
                     "Equity μ [%]": 100 * sto["yearly_return"], "Equity σ [%]": 100 * sto["sigma"],
                     "γ": d.get("utility_function", {}).get("gamma"), "Paths": sto["n_sims"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

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
            st.Page(runs_page, title="Runs", icon=":material/play_circle:", url_path="runs", default=True),
            st.Page(result_page, title="Result", icon=":material/insights:", url_path="result"),
        ],
        "Master data": [
            st.Page(configurations_page, title="Configurations", icon=":material/tune:", url_path="configurations"),
        ],
        "App": [
            st.Page(cache_page, title="Cache", icon=":material/storage:", url_path="cache"),
        ],
    })
    nav.run()


if __name__ == "__main__":  # streamlit runs the script as __main__
    main()
