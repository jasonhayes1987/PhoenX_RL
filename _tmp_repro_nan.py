"""Offline repro of the camera-PPO NaN: same model config, synthetic data."""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

import gymnasium as gym
import numpy as np
import torch as T
import yaml

from app.models import ModularModel, modular_parts_from_config

DEV = "cuda" if T.cuda.is_available() else "cpu"
T.manual_seed(0)

raw = yaml.safe_load((SRC / "Configs/IsaacSim/franka/cube_lift/dense/ppo_camera.yml").read_text())
model_cfg = raw["agent"]["config"]["model"]


class FakeEnv:
    single_observation_space = gym.spaces.Dict({
        "rgb": gym.spaces.Box(0, 255, (84, 84, 3), np.uint8),
        "policy": gym.spaces.Box(-np.inf, np.inf, (36,), np.float32),
    })
    observation_space = single_observation_space
    single_action_space = gym.spaces.Box(-1, 1, (8,), np.float32)
    action_space = single_action_space
    obs_key = None
    goal_key = None


env = FakeEnv()
parts = modular_parts_from_config({**model_cfg, "device": DEV}, env)
model = ModularModel(env=env, roots=parts["roots"], trunk=parts["trunk"],
                     branches=parts["branches"], optimizer_params=parts["optimizer_params"],
                     shared_update=parts["shared_update"], device=DEV)

B = 384
def batch():
    return {"rgb": T.randint(0, 255, (B, 84, 84, 3), dtype=T.uint8, device=DEV),
            "policy": T.randn(B, 36, device=DEV).clamp(-5, 5)}

# Simulate PPO-like combined updates: surrogate losses with realistic scales.
for step in range(30):
    obs = batch()
    out, _ = model(obs)
    dist, value = out["policy"], out["value"]
    actions = dist.sample()
    logp = dist.log_prob(actions)
    if logp.dim() > 1:
        logp = logp.sum(-1)
    adv = T.randn(B, device=DEV)
    returns = T.randn(B, 1, device=DEV) * 50  # lift returns are O(50)
    pol_loss = -(logp * adv).mean() - 0.006 * dist.entropy().mean()
    val_loss = ((value - returns) ** 2).mean()
    loss = pol_loss + 1.0 * val_loss
    model.zero_grad()
    loss.backward()
    model.clip(1.0, modules=model.branch_module_names("policy"))
    model.clip(1.0, modules=model.branch_module_names("value"))
    shared = model.shared_module_names()
    if shared:
        model.clip(1.0, modules=shared)
    model.step()
    with T.no_grad():
        mu_chk, _ = model(batch())
        m = mu_chk["policy"].sample()
    bad = not T.isfinite(m).all()
    print(f"step {step:02d}  pol_loss={pol_loss.item():.4f} val_loss={val_loss.item():.2f} "
          f"finite={not bad}", flush=True)
    if bad:
        print("NaN reproduced offline"); break
else:
    print("NO NaN offline with synthetic data")
