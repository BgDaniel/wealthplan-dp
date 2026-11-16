"""Result figures of an ``LSMCBellmanOptimizer`` (matplotlib; dark navy theme, blue / purple).

Fans show the median with 25–75 % and 5–95 % bands over the out-of-sample paths alive at
that age. One y-axis per chart. The two series colours (blue #2c6be0, purple #a07aea) pass
the palette checks on the navy surface; as blue / purple stay close for colour-blind readers,
two-series charts also differ in line style (solid / dashed) and carry direct labels.
"""

from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

BLUE, PURPLE = "#2c6be0", "#a07aea"
THEMES: Dict[str, Dict[str, str]] = {
    "dark": {"bg": "#0c1226", "fg": "#e7e9f5", "muted": "#9aa3c7", "grid": "#1f2747", "seq_low": "#151d38"},
    "light": {"bg": "#ffffff", "fg": "#1b2040", "muted": "#5b6285", "grid": "#e3e6f2", "seq_low": "#eef1fb"},
}


def _sequential(t: Dict[str, str]) -> LinearSegmentedColormap:
    """One hue from the surface (low) to the accent blue (high)."""
    return LinearSegmentedColormap.from_list("seq", [t["seq_low"], BLUE])


def _style(theme: str) -> Dict[str, str]:
    t = THEMES[theme]
    plt.rcParams.update({
        "figure.facecolor": t["bg"], "axes.facecolor": t["bg"], "savefig.facecolor": t["bg"],
        "axes.edgecolor": t["grid"], "axes.labelcolor": t["muted"], "text.color": t["fg"],
        "xtick.color": t["muted"], "ytick.color": t["muted"], "grid.color": t["grid"],
        "axes.grid": True, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "lines.linewidth": 2.0,
    })
    return t


def _euro(ax) -> None:
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v / 1e3:,.0f} k€" if abs(v) >= 1e4 else f"{v:,.0f} €"))


def _retirement(ax, opt, t: Dict[str, str]) -> None:
    r = opt.retirement_index
    if r < opt.n_months:
        ax.axvline(opt.age_grid[r], color=t["muted"], lw=1.0, ls="--")
        ax.text(opt.age_grid[r], ax.get_ylim()[1], " retirement", color=t["muted"], va="top", fontsize=9)


def _quantiles(data: np.ndarray) -> np.ndarray:
    """5/25/50/75/95 % quantiles per month over the alive paths (NaN where < 20 paths)."""
    q = np.full((5, data.shape[0]), np.nan)
    valid = np.isfinite(data).sum(axis=1) >= 20
    if valid.any():
        q[:, valid] = np.nanpercentile(data[valid], (5, 25, 50, 75, 95), axis=1)
    return q


def _fan(ax, age: np.ndarray, data: np.ndarray, color: str, label: str, outer: bool = True, ls: str = "-") -> None:
    """Median and bands of data (T, n_paths); NaN (dead) ignored, ages with < 20 paths skipped."""
    q = _quantiles(data)
    if outer:
        ax.fill_between(age, q[0], q[4], color=color, alpha=0.16, lw=0, label="5–95 %")
    ax.fill_between(age, q[1], q[3], color=color, alpha=0.32, lw=0, label=f"{label}: 25–75 %" if not outer
                    else "25–75 %")
    ax.plot(age, q[2], color=color, ls=ls, label=f"{label} (median)")


def _label_end(ax, age: np.ndarray, data: np.ndarray, text: str, color: str, t) -> None:
    """Direct label at the median's peak (secondary encoding next to the colour)."""
    med = _quantiles(data)[2]
    if np.isfinite(med).any():
        i = int(np.nanargmax(np.abs(med)))
        ax.annotate(text, (age[i], med[i]), xytext=(6, 6), textcoords="offset points", color=t["fg"], fontsize=9)


def wealth(opt, t) -> Figure:
    fig, ax = plt.subplots(figsize=(11, 4.6))
    _fan(ax, opt.age_grid, opt.opt_wealth.values, BLUE, "Wealth")
    ax.set(title="Wealth over age", xlabel="Age")
    ax.set_ylim(bottom=0)
    _euro(ax)
    _retirement(ax, opt, t)
    ax.legend(loc="upper left")
    return fig


def consumption(opt, t) -> Figure:
    fig, ax = plt.subplots(figsize=(11, 4.6))
    _fan(ax, opt.age_grid, opt.opt_consumption.values, PURPLE, "Consumption")
    ax.plot(opt.age_grid, opt.cf, color=t["muted"], lw=1.3, ls="--", label="Net cashflow (income − fixed costs)")
    ax.set(title="Monthly consumption", xlabel="Age")
    _euro(ax)
    _retirement(ax, opt, t)
    ax.legend(loc="upper left", ncol=2)
    return fig


def equity_share(opt, t) -> Figure:
    fig, ax = plt.subplots(figsize=(11, 4.6))
    _fan(ax, opt.age_grid, 100.0 * opt.opt_equity_share.values, BLUE, "Equity share")
    ax.set(title="Equity share of the invested wealth", xlabel="Age", ylabel="%")
    ax.set_ylim(-2, 102)
    _retirement(ax, opt, t)
    ax.legend(loc="lower left")
    return fig


def holdings(opt, t) -> Figure:
    """Amount held in equity and in the savings account after rebalancing (median, 25–75 %)."""
    eq = opt.opt_equity_share.values * opt.opt_invested.values
    sav = opt.opt_invested.values - eq
    fig, ax = plt.subplots(figsize=(11, 4.6))
    _fan(ax, opt.age_grid, eq, BLUE, "Equity", outer=False)
    _fan(ax, opt.age_grid, sav, PURPLE, "Savings account", outer=False, ls="--")
    _label_end(ax, opt.age_grid, eq, "equity", BLUE, t)
    _label_end(ax, opt.age_grid, sav, "savings", PURPLE, t)
    ax.set(title="Holdings: equity vs. savings account", xlabel="Age")
    ax.set_ylim(bottom=0)
    _euro(ax)
    _retirement(ax, opt, t)
    ax.legend(loc="upper left", ncol=2)
    return fig


def flows(opt, t) -> Figure:
    """Monthly purchases (+) / sales (−) per asset: target holding minus holding before rebalancing."""
    eq = opt.opt_equity_share.values * opt.opt_invested.values
    sav = opt.opt_invested.values - eq
    buy_eq = eq - opt.opt_equity_before.values
    buy_sav = sav - opt.opt_savings_before.values
    fig, ax = plt.subplots(figsize=(11, 4.6))
    _fan(ax, opt.age_grid[1:], buy_eq[1:], BLUE, "Equity", outer=False)
    _fan(ax, opt.age_grid[1:], buy_sav[1:], PURPLE, "Savings account", outer=False, ls="--")
    ax.axhline(0.0, color=t["muted"], lw=1.0)
    ax.set(title="Monthly investment per asset: buy (+) / sell (−)", xlabel="Age")
    _euro(ax)
    _retirement(ax, opt, t)
    ax.legend(loc="lower left", ncol=2)
    return fig


def investment(opt, t) -> Figure:
    fig, ax = plt.subplots(figsize=(11, 4.6))
    _fan(ax, opt.age_grid, opt.cf[:, None] - opt.opt_consumption.values, PURPLE, "Saving")
    ax.axhline(0.0, color=t["muted"], lw=1.0)
    ax.set(title="Monthly saving (+) / withdrawal (−)", xlabel="Age")
    _euro(ax)
    _retirement(ax, opt, t)
    ax.legend(loc="lower left")
    return fig


def equity_policy(opt, t) -> Figure:
    """Optimal equity share by age and invested wealth (market level S_t = 1)."""
    invested = opt.opt_invested.values
    top = float(np.nanpercentile(invested, 97)) if np.isfinite(invested).any() else opt.wealth_grid[-1]
    top = min(max(top, 1e4), opt.wealth_grid[-1])
    w_axis = np.linspace(0.0, top, 160)
    months = np.arange(opt.n_months - 1)
    z = np.array([np.interp(w_axis, opt.wealth_grid, opt.policy_grid(m)[0]) for m in months]).T
    fig, ax = plt.subplots(figsize=(11, 4.6))
    mesh = ax.pcolormesh(opt.age_grid[months], w_axis, 100.0 * z, shading="auto", cmap=_sequential(t), vmin=0,
                         vmax=100, rasterized=True)
    ax.plot(opt.age_grid, np.nanmedian(invested, axis=1), color=PURPLE, lw=1.6, ls="--",
            label="median invested wealth")
    cb = fig.colorbar(mesh, ax=ax, pad=0.01)
    cb.set_label("equity share [%]")
    cb.outline.set_visible(False)
    ax.set(title="Optimal equity share by age and invested wealth", xlabel="Age")
    ax.grid(False)
    ax.set_ylim(0, top)
    _euro(ax)
    ax.legend(loc="upper left", frameon=True, facecolor=t["bg"], edgecolor=t["grid"], framealpha=0.9)
    return fig


def survival(opt, t) -> Figure:
    fig, ax = plt.subplots(figsize=(11, 4.0))
    ax.plot(opt.age_grid, 100.0 * opt.survival_probabilities, color=PURPLE, label="model")
    ax.plot(opt.age_grid, 100.0 * opt.survival_paths.mean(axis=0), color=t["muted"], lw=1.0, ls=":",
            label="simulated")
    ax.set(title="Probability to be alive", xlabel="Age", ylabel="%")
    ax.set_ylim(0, 102)
    ax.legend(loc="lower left")
    return fig


def regression(opt, t) -> Figure:
    months = [m for m in opt.months if m in opt.r_squared]
    ages = [opt.age_grid[i] for i, m in enumerate(opt.months) if m in opt.r_squared]
    fig, ax = plt.subplots(figsize=(11, 4.0))
    ax.set(title="LSMC regression R² (median over grid points)", xlabel="Age")
    ax.set_ylim(-0.05, 1.05)
    if opt.regression_degree == 0:
        ax.text(0.5, 0.5, "regression degree 0: constant basis = Monte Carlo mean (no R²)",
                transform=ax.transAxes, ha="center", va="center", color=t["muted"])
        return fig
    ax.plot(ages, [opt.r_squared[m] for m in months], color=BLUE)
    return fig


def depletion(opt, t) -> Figure:
    """Probability (among the living) of low wealth and of unpaid fixed costs."""
    from wealthplan.analysis import depletion_risk

    risk = depletion_risk(opt)
    ok = risk["alive_paths"].values >= 20
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.plot(risk["age"][ok], 100.0 * risk["low_wealth"][ok], color=BLUE,
            label="wealth below 6 months of fixed costs")
    ax.plot(risk["age"][ok], 100.0 * risk["shortfall"][ok], color=PURPLE, ls="--",
            label="fixed costs cannot be paid")
    ax.set(title="Depletion risk (share of the paths alive at that age)", xlabel="Age", ylabel="%")
    ax.set_ylim(bottom=0)
    ax.text(0.99, 0.5, "the plan consumes the remaining\nwealth towards its end date", transform=ax.transAxes,
            ha="right", va="center", color=t["muted"], fontsize=9)
    _retirement(ax, opt, t)
    ax.legend(loc="upper left")
    return fig


def taxes(opt, t) -> Figure:
    """Taxes per year: capital gains tax (fan over paths), income tax on pensions, Rürup refund."""
    from wealthplan.analysis import yearly_taxes

    det, cap = yearly_taxes(opt)
    age = det["age"].values
    fig, ax = plt.subplots(figsize=(11, 4.6))
    _fan(ax, age, cap, BLUE, "Capital gains tax", outer=False)
    ax.plot(age, det["income_tax"].values, color=PURPLE, ls="--", label="Income tax on pensions")
    if det["rurup_refund"].abs().sum() > 0:
        ax.plot(age, -det["rurup_refund"].values, color=t["muted"], ls=":", lw=1.6,
                label="Rürup tax refund (negative tax)")
    ax.axhline(0.0, color=t["muted"], lw=0.8)
    ax.set(title="Taxes per year", xlabel="Age")
    _euro(ax)
    _retirement(ax, opt, t)
    ax.legend(loc="upper left")
    return fig


def ruerup_vs_etf_plot(opt, t) -> Figure:
    """Expected cumulative net cash of the Rürup plan vs. an ETF plan with the same net cost."""
    from wealthplan.analysis import ruerup_vs_etf

    frame, info = ruerup_vs_etf(opt)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set(title="Rürup vs. ETF savings plan (same net cost) – expected cumulative net cash", xlabel="Age")
    if frame.empty:
        ax.text(0.5, 0.5, "no Rürup plan configured", transform=ax.transAxes, ha="center", va="center",
                color=t["muted"])
        return fig
    ax.plot(frame["age"], frame["ruerup_cum"], color=PURPLE, label="Rürup (paid while alive)")
    ax.plot(frame["age"], frame["etf_cum"], color=BLUE, ls="--", label="ETF withdrawals (paid while alive)")
    ax.plot(frame["age"], frame["etf_cum_with_heirs"], color=BLUE, ls=":", lw=1.4,
            label="ETF incl. capital left to heirs")
    ax.axhline(0.0, color=t["muted"], lw=0.8)
    be = info["break_even_age"]
    if be is not None:
        ax.axvline(be, color=t["muted"], lw=1.0, ls="-.")
        ax.text(be, ax.get_ylim()[0], f" break-even {be:.0f}", color=t["fg"], va="bottom", fontsize=9)
    text = (f"net cost {info['net_monthly_cost']:,.0f} €/month\n"
            f"capital at retirement: Rürup {info['ruerup_capital_at_retirement'] / 1e3:,.0f} k€, "
            f"ETF {info['etf_capital_at_retirement'] / 1e3:,.0f} k€\n"
            f"Rürup net pension {info['ruerup_net_pension']:,.0f} €/month · "
            f"ETF withdrawal {info['etf_monthly_withdrawal']:,.0f} €/month")
    ax.text(0.99, 0.03, text, transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color=t["fg"],
            bbox=dict(facecolor=t["bg"], edgecolor=t["grid"], boxstyle="round,pad=0.5"))
    _euro(ax)
    _retirement(ax, opt, t)
    ax.legend(loc="upper left")
    return fig


COMPOSITION = (("Salary", BLUE, ""), ("Pensions (gross)", "#5b8ef0", ""), ("Tax refund", "#9bbaf7", ""),
               ("Living costs", PURPLE, ""), ("Insurances", "#c3a8f3", "//"),
               ("Pension contributions", "#7d55cf", ""), ("Income tax", "#5b3aa6", "xx"))


def composition(opt, t) -> Figure:
    """Deterministic cashflows by category, yearly average per month (income +, outflows −)."""
    items = opt.cashflow_items()
    cats = opt.cashflow_categories()
    grouped = items.T.groupby(lambda c: cats.get(c, "Living costs")).sum().T
    years = np.array([m.year for m in opt.months])
    yearly = grouped.groupby(years).mean()
    age = np.array([opt.age_grid[np.argmax(years == y)] for y in yearly.index])
    fig, ax = plt.subplots(figsize=(11, 4.8))
    pos_base = np.zeros(len(yearly))
    neg_base = np.zeros(len(yearly))
    for name, color, hatch in COMPOSITION:
        if name not in yearly or not yearly[name].abs().sum():
            continue
        v = yearly[name].values
        base = np.where(v >= 0, pos_base, neg_base)
        ax.fill_between(age, base, base + v, step="post", color=color, hatch=hatch or None, edgecolor=t["bg"],
                        lw=0.4, label=name)
        pos_base = np.where(v >= 0, pos_base + v, pos_base)
        neg_base = np.where(v < 0, neg_base + v, neg_base)
    ax.step(age, yearly.sum(axis=1).values, where="post", color=t["fg"], lw=1.4, label="net cashflow")
    ax.axhline(0.0, color=t["muted"], lw=0.8)
    ax.set(title="Cashflow composition (average per month of each year)", xlabel="Age")
    _euro(ax)
    _retirement(ax, opt, t)
    ax.legend(loc="lower left", ncol=4, fontsize=8.5)
    return fig


COMPARE_STYLES = ((BLUE, "-"), (PURPLE, "--"), ("#8fb3f5", ":"), ("#cdb6f7", "-."))


def compare_runs(frames: List[pd.DataFrame], labels: List[str], column: str, title: str, unit: str,
                 theme: str = "dark") -> Figure:
    """Median curves of several runs (quantiles.csv frames) in one chart; at most 4 runs."""
    with plt.rc_context():
        _style(theme)
        fig, ax = plt.subplots(figsize=(11, 4.6))
        for (color, ls), frame, label in zip(COMPARE_STYLES, frames, labels):
            scale = 100.0 if unit == "%" else 1.0
            ax.plot(frame["age"], scale * frame[column], color=color, ls=ls, label=label)
        ax.set(title=title, xlabel="Age")
        if unit == "%":
            ax.set_ylabel("%")
            ax.set_ylim(-2, 102)
        else:
            _euro(ax)
        ax.legend(loc="best")
        fig.tight_layout()
    return fig


FIGURES: Dict[str, Tuple[str, Callable]] = {
    "01_wealth": ("Wealth", wealth),
    "02_consumption": ("Consumption", consumption),
    "03_equity_share": ("Equity share", equity_share),
    "04_holdings": ("Equity vs. savings", holdings),
    "05_flows": ("Monthly investment", flows),
    "06_saving": ("Saving / withdrawal", investment),
    "07_equity_policy": ("Equity policy", equity_policy),
    "08_survival": ("Survival", survival),
    "09_regression": ("Regression R²", regression),
    "10_depletion": ("Depletion risk", depletion),
    "11_taxes": ("Taxes", taxes),
    "12_ruerup_vs_etf": ("Rürup vs. ETF", ruerup_vs_etf_plot),
    "13_composition": ("Cashflow composition", composition),
}


def save_figures(opt, folder: str | Path, theme: str = "dark") -> List[Path]:
    """Render all figures into ``folder`` as PNG; returns the files."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    files = []
    with plt.rc_context():
        t = _style(theme)
        for stem, (_, make) in FIGURES.items():
            fig = make(opt, t)
            fig.tight_layout()
            path = folder / f"{stem}.png"
            fig.savefig(path, dpi=130)
            plt.close(fig)
            files.append(path)
    return files


def show_figures(opt, theme: str = "light") -> None:
    """Show all figures interactively."""
    with plt.rc_context():
        t = _style(theme)
        for _, make in FIGURES.values():
            make(opt, t).tight_layout()
        plt.show()
