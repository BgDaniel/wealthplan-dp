"""Background optimisation runs for the app: schedule, run, report status.

A job is a folder ``<jobs_root>/<job id>/`` with

* ``job.json``: the YAML parameters of the plan plus run options;
* ``status.json``: state ("scheduled", "queued", "running", "done", "failed"), current step,
  overall progress (0..1), timestamps, estimated remaining seconds, the result folder and key
  figures when done, the error message if failed;
* ``log.txt``: log output of the worker process.

:func:`submit_job` starts ``python -m wealthplan.jobs <job folder>`` as a separate
process, so a run survives a restart of the app and the app only reads files. A job may carry
a start time (``not_before``): it waits as "scheduled" until then. At most ``max_parallel``
jobs compute at the same time, the others wait as "queued" (oldest first).
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_JOBS_ROOT = Path("results") / "_jobs"
ACTIVE = ("scheduled", "queued", "running")
SUMMARY_KEYS = ("equity_share_start", "equity_share_retirement", "consumption_start", "wealth_retirement_median",
                "prob_shortfall", "from_cache")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    tmp.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def submit_job(spec: dict[str, Any], jobs_root: str | Path = DEFAULT_JOBS_ROOT,
               start: bool = True) -> Path:
    """Create a job folder and start the worker process.

    Args:
        spec: ``{"params": {...YAML parameters}, "config_name": "...", "results_root": "results",
            "cache_root": ".cache", "max_parallel": 2, "not_before": ISO datetime or None}``.
        jobs_root: Folder holding all job folders.
        start: Start the worker process (False: only create the folder, e.g. in tests).

    Returns:
        The job folder.
    """
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
    job_dir = Path(jobs_root) / job_id
    job_dir.mkdir(parents=True)
    _write_json(job_dir / "job.json", spec)
    scheduled = bool(spec.get("not_before"))
    status = {"id": job_id, "name": spec.get("config_name", ""),
              "state": "scheduled" if scheduled else "queued",
              "step": f"start at {spec['not_before']}" if scheduled else "waiting", "progress": 0.0,
              "created": _now(), "not_before": spec.get("not_before"), "started": None, "finished": None,
              "eta_s": None, "run_dir": None, "summary": None, "error": None, "pid": None}
    _write_json(job_dir / "status.json", status)
    if start:
        with open(job_dir / "log.txt", "w", encoding="utf-8") as log:  # the child keeps its own copy
            proc = subprocess.Popen([sys.executable, "-m", "wealthplan.jobs", str(job_dir)],
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                    cwd=os.getcwd(), env=_child_env())
        status["pid"] = proc.pid
        _write_json(job_dir / "status.json", status)
    return job_dir


def _child_env() -> dict[str, str]:
    """Environment of the worker: make ``backend/src`` importable."""
    env = dict(os.environ)
    src = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    stat = Path(f"/proc/{pid}/stat")
    if stat.exists():  # a finished but not yet reaped child is a zombie ("Z")
        try:
            return stat.read_text().split(")")[-1].split()[0] != "Z"
        except OSError:
            return True
    return True


def read_status(job_dir: str | Path) -> dict[str, Any]:
    """Status of a job; an active job whose worker process has died is reported as failed."""
    job_dir = Path(job_dir)
    status = _read_json(job_dir / "status.json")
    if status["state"] in ACTIVE and status.get("pid") and not _alive(status["pid"]):
        status = _read_json(job_dir / "status.json")  # it may have finished meanwhile
        if status["state"] in ACTIVE:
            status["state"] = "failed"
            status["error"] = "worker process ended unexpectedly (see log.txt)"
    status["job_dir"] = str(job_dir)
    return status


def list_jobs(jobs_root: str | Path = DEFAULT_JOBS_ROOT) -> list[dict[str, Any]]:
    """Status of all jobs, newest first."""
    root = Path(jobs_root)
    if not root.exists():
        return []
    jobs = [read_status(d) for d in root.iterdir() if (d / "status.json").exists()]
    return sorted(jobs, key=lambda s: s["created"], reverse=True)


def cancel_job(job_dir: str | Path) -> None:
    """Stop a scheduled, queued or running job (terminates the worker process)."""
    job_dir = Path(job_dir)
    status = read_status(job_dir)
    if status["state"] not in ACTIVE:
        return
    if status.get("pid") and _alive(status["pid"]):
        try:
            os.kill(status["pid"], signal.SIGTERM)
        except OSError:
            pass
    status.pop("job_dir", None)
    status.update(state="failed", step="cancelled", error="cancelled by user", finished=_now())
    _write_json(job_dir / "status.json", status)


def delete_job(job_dir: str | Path) -> None:
    """Remove a finished or failed job folder (its result folder is kept)."""
    job_dir = Path(job_dir)
    if read_status(job_dir)["state"] in ACTIVE:
        raise ValueError("cancel the job before deleting it")
    for f in job_dir.iterdir():
        f.unlink()
    job_dir.rmdir()


def _wait_for_start(job_dir: Path, status: dict[str, Any], spec: dict[str, Any]) -> None:
    """Wait for the start time, then until fewer than ``max_parallel`` older jobs are running."""
    not_before = spec.get("not_before")
    if not_before:
        while datetime.now() < datetime.fromisoformat(not_before):
            time.sleep(5.0)
        status.update(state="queued", step="waiting")
        _write_json(job_dir / "status.json", status)
    max_parallel = int(spec.get("max_parallel", 2))
    while True:
        running = [j for j in list_jobs(job_dir.parent) if j["state"] == "running" and j["id"] != job_dir.name]
        older_queued = [j for j in list_jobs(job_dir.parent)
                        if j["state"] == "queued" and j["id"] < job_dir.name]
        if len(running) + len(older_queued) < max_parallel:
            return
        time.sleep(2.0)


def run_job(job_dir: str | Path) -> dict[str, Any]:
    """Run the plan of a job folder and keep ``status.json`` up to date.

    Returns:
        The final status.
    """
    from wealthplan.run import run_plan

    job_dir = Path(job_dir)
    spec = _read_json(job_dir / "job.json")
    status = _read_json(job_dir / "status.json")
    status["pid"] = os.getpid()
    _write_json(job_dir / "status.json", status)
    _wait_for_start(job_dir, status, spec)

    t_start = time.time()
    last_write = [0.0]

    def update(force: bool = False, **fields: Any) -> None:
        status.update(fields)
        now = time.time()
        if force or now - last_write[0] >= 1.0:
            elapsed = now - t_start
            p = status["progress"]
            status["elapsed_s"] = elapsed
            status["eta_s"] = elapsed * (1.0 - p) / p if p > 0.01 and status["state"] == "running" else None
            _write_json(job_dir / "status.json", status)
            last_write[0] = now

    update(force=True, state="running", step="preparing", started=_now())
    try:
        run_dir = run_plan(spec["params"], spec.get("results_root", "results"),
                           cache_root=spec.get("cache_root"), progress=lambda step, p: update(step=step, progress=p))
        summary = _read_json(run_dir / "summary.json")
        update(force=True, state="done", step="done", progress=1.0, finished=_now(), run_dir=str(run_dir),
               summary={k: summary.get(k) for k in SUMMARY_KEYS})
    except Exception as exc:  # noqa: BLE001 - reported to the app, re-raised for the log
        update(force=True, state="failed", finished=_now(), error=f"{type(exc).__name__}: {exc}",
               traceback=traceback.format_exc())
        raise
    return status


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_job(sys.argv[1])
