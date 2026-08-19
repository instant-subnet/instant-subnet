from __future__ import annotations

import importlib.util
import subprocess
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).parents[1]


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DeploymentTests(unittest.TestCase):
    def test_updater_fast_forwards_main_and_syncs_only_when_changed(self) -> None:
        updater = _load("validator_updater", "scripts/update_validator.py")
        outputs = iter(["a" * 40, "", "b" * 40, "", ""])
        commands: list[list[str]] = []

        def run(command, **kwargs):
            commands.append(command)
            return subprocess.CompletedProcess(command, 0, next(outputs), "")

        with (
            mock.patch.object(updater, "LOCK_PATH", ROOT / ".test-update.lock"),
            mock.patch.object(updater.subprocess, "run", side_effect=run),
        ):
            self.assertTrue(updater.update_once())

        self.assertEqual(
            commands,
            [
                ["git", "rev-parse", "HEAD"],
                ["git", "fetch", "origin", "main"],
                ["git", "rev-parse", "origin/main"],
                ["git", "merge", "--ff-only", "origin/main"],
                [str(updater.PYTHON), "-m", "pip", "install", "-e", "."],
            ],
        )
        (ROOT / ".test-update.lock").unlink(missing_ok=True)

    def test_installed_cron_uses_the_shared_lock_and_run_once(self) -> None:
        starter = _load("validator_starter", "scripts/start_validator.py")

        self.assertIn(
            "*/5 * * * * flock -n /var/lock/instant-validator.lock",
            starter.CRON_LINE,
        )
        self.assertIn("run_validator.sh", starter.CRON_LINE)
        self.assertEqual(starter.PM2_NAME, "instant-validator-updater")
        self.assertEqual(
            (ROOT / "deploy/instant-validator.cron").read_text(),
            "*/5 * * * * flock -n /var/lock/instant-validator.lock "
            "instant-validator run-once\n",
        )


if __name__ == "__main__":
    unittest.main()
