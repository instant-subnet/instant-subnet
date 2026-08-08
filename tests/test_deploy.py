from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_pm2_ecosystem_files_use_detectable_commonjs_suffix():
    pm2_dir = ROOT / "deploy" / "pm2"

    for role in ("miner", "validator", "platform", "mock-vllm"):
        assert (pm2_dir / f"{role}.ecosystem.config.cjs").is_file()
        assert not (pm2_dir / f"{role}.ecosystem.cjs").exists()
