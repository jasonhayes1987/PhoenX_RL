"""``env_log_metrics``: Isaac Lab ``extras["log"]`` numbers become ``env/`` metrics."""

from __future__ import annotations

import numpy as np
import torch as T

from phoenx.trainer import env_log_metrics


def test_numbers_are_prefixed_and_converted():
    infos = {
        "log": {
            "Curriculum/robustness/level": 0.35,
            "Curriculum/robustness/landing_rate": T.tensor(0.82),
            "Episode_Reward/step": T.tensor([12.5]),
            "Episode_Termination/landed": 37,
            "flag": True,
            "np_scalar": np.float32(2.5),
        },
        "time_outs": T.zeros(4, dtype=T.bool),
    }
    out = env_log_metrics(infos)
    assert out == {
        "env/Curriculum/robustness/level": 0.35,
        "env/Curriculum/robustness/landing_rate": float(T.tensor(0.82)),
        "env/Episode_Reward/step": 12.5,
        "env/Episode_Termination/landed": 37.0,
        "env/flag": 1.0,
        "env/np_scalar": 2.5,
    }
    assert all(isinstance(v, float) for v in out.values())


def test_non_scalars_and_non_finite_are_skipped():
    infos = {"log": {"vector": T.ones(3), "text": "hi", "nested": {"a": 1}, "nan": float("nan"), "ok": 1}}
    assert env_log_metrics(infos) == {"env/ok": 1.0}


def test_envs_without_a_log_dict_give_nothing():
    assert env_log_metrics({}) == {}
    assert env_log_metrics({"log": None}) == {}
    assert env_log_metrics({"is_success": np.array([True, False])}) == {}
    assert env_log_metrics(None) == {}
