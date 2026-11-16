import tempfile
import unittest
from pathlib import Path

from wealthplan.jobs import list_jobs, read_status, submit_job


class TestJobs(unittest.TestCase):
    """Job folders and status files (without starting worker processes)."""

    def test_submit_and_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            now = submit_job({"params": {}, "config_name": "a"}, tmp, start=False)
            later = submit_job({"params": {}, "config_name": "b", "not_before": "2099-01-01T00:00"}, tmp,
                               start=False)
            self.assertEqual(read_status(now)["state"], "queued")
            self.assertEqual(read_status(later)["state"], "scheduled")
            self.assertEqual({j["name"] for j in list_jobs(tmp)}, {"a", "b"})
            self.assertTrue((Path(now) / "job.json").exists())


if __name__ == "__main__":
    unittest.main()
