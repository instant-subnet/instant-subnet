from __future__ import annotations

import unittest
from pathlib import Path


class CronTests(unittest.TestCase):
    def test_five_minute_run_once_entry_is_exact(self) -> None:
        line = (Path(__file__).parents[1] / "deploy/instant-validator.cron").read_text()

        self.assertEqual(
            line,
            "*/5 * * * * flock -n /var/lock/instant-validator.lock "
            "instant-validator run-once\n",
        )


if __name__ == "__main__":
    unittest.main()
