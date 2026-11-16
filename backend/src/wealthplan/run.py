"""Run a plan from a YAML parameter file and save the results.

    python -m wealthplan.run params/lifecycle_params.yaml [--results results] [--no-figures]

Output folder ``<results>/<timestamp>_<run_config_id>/``:
summary.json, quantiles.csv, cashflows.csv (items per month), policy.npz (for ``decide``),
params.yaml (input), figures/*.png.
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import yaml

from config.stochastic.lsmc.lsmc_config_mapper import LSMCConfigMapper
from wealthplan.optimizer.stochastic.lsmc.lsmc_bellman_optimizer import LSMCBellmanOptimizer

logger = logging.getLogger(__name__)

ProgressFn = Callable[[str, float], None]


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_plan(data: Dict[str, Any], results_root: str | Path = "results", figures: bool = True,
             cache_root: Optional[str] = None, progress: Optional[ProgressFn] = None) -> Path:
    """
    Solve a plan given as YAML dictionary and write its result folder.

    Args:
        data: Parsed YAML parameters.
        results_root: Folder for the result folders.
        figures: Render the PNG figures (dark theme).
        cache_root: Cache folder for the regression coefficients (default $CACHE_FOLDER or ./.cache).
        progress: Callback (step label, fraction done in [0, 1]).

    Returns:
        The result folder.
    """
    params = LSMCConfigMapper.map_yaml_to_params(data)
    optimizer = LSMCBellmanOptimizer(**params, cache_root=cache_root)
    optimizer.solve(progress=progress)

    if progress is not None:
        progress("saving results", 0.95)
    t0 = time.perf_counter()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(results_root) / f"{stamp}_{optimizer.run_config_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "params.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    optimizer.quantiles().to_csv(run_dir / "quantiles.csv")
    optimizer.cashflow_items().to_csv(run_dir / "cashflows.csv")
    with open(run_dir / "cashflow_categories.json", "w", encoding="utf-8") as f:
        json.dump(optimizer.cashflow_categories(), f, indent=2)
    np.savez_compressed(run_dir / "policy.npz", **optimizer.policy_dict())
    if figures:
        from wealthplan.plots import save_figures

        save_figures(optimizer, run_dir / "figures")
    optimizer.runtimes["save"] = time.perf_counter() - t0
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(optimizer.summary(), f, indent=2, default=str)
    logger.info("Results written to %s", run_dir)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="LSMC lifecycle optimisation (consumption and equity share)")
    parser.add_argument("params", help="YAML parameter file")
    parser.add_argument("--results", default="results", help="results folder (default: results)")
    parser.add_argument("--no-figures", action="store_true", help="skip the PNG figures")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_dir = run_plan(load_yaml(args.params), args.results, figures=not args.no_figures)
    print(run_dir)


if __name__ == "__main__":
    main()
