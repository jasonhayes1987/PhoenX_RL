"""Resuming a saved run: ``phoenx.run_info`` and ``phoenx-train --agent_dir``.

Covers:
    * ``read_run_info`` reports the saved step / stop limit without building
      the run, and works as ``python -m phoenx.run_info`` (how the GUI calls it).
    * ``phoenx-train --agent_dir DIR --log_level X`` no longer crashes
      (``config`` used to be undefined when a log level was given).
    * ``--stop_units`` trains a resumed run past its original limit, and
      refuses a target the checkpoint has already reached.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

FIXTURE = Path(__file__).parent / "fixtures" / "smoke_ppo_cartpole.yml"


@pytest.fixture()
def saved_run(tmp_path) -> Path:
    """A tiny PPO CartPole run trained for 2048 steps and saved."""
    from phoenx.builder import build_trainer_from_config

    cfg = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    cfg["save_dir"] = str(tmp_path / "run")
    cfg["schedule"]["stop_units"] = 2048
    cfg["env"]["config"]["num_envs"] = 2
    trainer = build_trainer_from_config(cfg)
    trainer.train()
    trainer.save()
    return Path(cfg["save_dir"])


def test_read_run_info(saved_run):
    from phoenx.run_info import is_run_dir, read_run_info

    assert is_run_dir(saved_run)
    info = read_run_info(saved_run)
    assert info["agent_type"] == "PPO"
    assert info["env"] == "CartPole-v1"
    assert info["stop_unit"] == "timestep"
    assert info["stop_units"] == 2048
    assert info["progress"] == info["step"] >= 2048
    assert info["has_trainer_state"] is True


def test_run_info_module_prints_json(saved_run):
    out = subprocess.run([sys.executable, "-m", "phoenx.run_info", str(saved_run)],
                         capture_output=True, text=True, check=True)
    assert json.loads(out.stdout.strip().splitlines()[-1])["agent_type"] == "PPO"


def test_run_info_missing_dir(tmp_path):
    from phoenx.run_info import read_run_info

    with pytest.raises(FileNotFoundError):
        read_run_info(tmp_path)


def _train_cli(monkeypatch, *args):
    from phoenx.cli import train

    monkeypatch.setattr(sys, "argv", ["phoenx-train", *args])
    train.main()


def test_resume_with_log_level_and_stop_units_trains_further(saved_run, monkeypatch):
    from phoenx.run_info import read_run_info
    from phoenx.trainer import Trainer

    before = read_run_info(saved_run)["step"]
    target = before + 2048
    seen = {}
    real_train = Trainer.train

    def spy(self, *a, **k):
        out = real_train(self, *a, **k)
        seen["stop_units"] = self.schedule.stop_units
        seen["step"] = self._step
        return out

    monkeypatch.setattr(Trainer, "train", spy)
    # --log_level together with --agent_dir used to raise NameError.
    _train_cli(monkeypatch, "--agent_dir", str(saved_run), "--log_level", "ERROR",
               "--stop_units", str(target))
    assert seen["stop_units"] == target
    assert seen["step"] >= target > before


def test_stop_units_at_or_below_checkpoint_is_refused(saved_run, monkeypatch):
    from phoenx.run_info import read_run_info

    reached = read_run_info(saved_run)["progress"]
    with pytest.raises(ValueError, match="not past the saved checkpoint"):
        _train_cli(monkeypatch, "--agent_dir", str(saved_run), "--stop_units", str(reached))
