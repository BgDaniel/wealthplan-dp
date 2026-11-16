"""Result figures of an ``LSMCBellmanOptimizer`` (matplotlib; dark theme like the BESS app).

Fans show the median with 25–75 % and 5–95 % bands over the out-of-sample paths alive at
that age. One y-axis per chart.
"""

from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

BLUE, ORANGE, GREEN, YELLOW, PINK, PURPLE = "#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#9085e9"
THEMES: Dict[str, Dict[str, str]] = {
    "dark": {"bg": "#0b0f17", "fg": "#e6e9ef", "muted": "#9aa4b2", "grid": "#1e2633", "seq_low": "#141c2b"},
    "light": {"bg": "#ffffff", "fg": "#1b2330", "muted": "#5b6573", "grid": "#e3e7ee", "seq_low": "#eef3fb"},
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


def _fan(ax, age: np.ndarray, data: np.ndarray, color: str, label: str) -> None:
    """Median and bands of data (T, n_paths); NaN (dead) ignored, ages with < 20 paths skipped."""
    q = np.full((5, data.shape[0]), np.nan)
    valid = np.isfinite(data).sum(axis=1) >= 20
    if valid.any():
        q[:, valid] = np.nanpercentile(data[valid], (5, 25, 50, 75, 95), axis=1)
    ax.fill_between(age, q[0], q[4], color=color, alpha=0.18, lw=0, label="5–95 %")
    ax.fill_between(age, q[1], q[3], color=color, alpha=0.35, lw=0, label="25–75 %")
    ax.plot(age, q[2], color=color, label=f"{label} (median)")


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
    _fan(ax, opt.age_grid, opt.opt_consumption.values, GREEN, "Consumption")
    ax.plot(opt.age_grid, opt.cf, color=ORANGE, lw=1.5, label="Net cashflow (income − fixed costs)")
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


def equity_amount(opt, t) -> Figure:
    fig, ax = plt.subplots(figsize=(11, 4.6))
    _fan(ax, opt.age_grid, opt.opt_equity_share.values * opt.opt_invested.values, ORANGE, "Amount in equity")
    ax.set(title="Amount invested in equity", xlabel="Age")
    ax.set_ylim(bottom=0)
    _euro(ax)
    _retirement(ax, opt, t)
    ax.legend(loc="upper left")
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
    ax.plot(opt.age_grid, np.nanmedian(invested, axis=1), color=ORANGE, lw=1.5, label="median invested wealth")
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
    ax.plot(opt.age_grid, 100.0 * opt.survival_probabilities, color=PINK, label="model")
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
    ax.plot(ages, [opt.r_squared[m] for m in months], color=YELLOW)
    return fig


FIGURES: Dict[str, Tuple[str, Callable]] = {
    "01_wealth": ("Wealth", wealth),
    "02_consumption": ("Consumption", consumption),
    "03_equity_share": ("Equity share", equity_share),
    "04_equity_amount": ("Equity amount", equity_amount),
    "05_saving": ("Saving / withdrawal", investment),
    "06_equity_policy": ("Equity policy", equity_policy),
    "07_survival": ("Survival", survival),
    "08_regression": ("Regression R²", regression),
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
