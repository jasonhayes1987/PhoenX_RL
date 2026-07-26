"""Step the camera env with random [-1,1] actions and report the first
non-finite value in any observation group / reward. Deleted after use."""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

import torch as T  # noqa: E402

from app.env_wrapper import IsaacSimWrapper  # noqa: E402

wrapper = IsaacSimWrapper(
    cfg="Configs.IsaacSim.franka.cube_lift.custom_franka_cube_lift_cfg:FrankaCubeLiftCameraEnvCfg",
    num_envs=64, obs_key=None, goal_key=None, ach_goal_key=None,
    wrappers=[], render_mode="headless", seed=42, enable_cameras=True,
)
obs = wrapper.reset(seed=42)
act_dim = wrapper.single_action_space.shape[-1]

bad_steps = 0
for step in range(300):
    action = (T.rand((64, act_dim), device="cuda") * 2 - 1)
    obs = wrapper.step(action)
    msgs = []
    for k, v in obs.states.items():
        if v.dtype.is_floating_point and not T.isfinite(v).all():
            n = (~T.isfinite(v)).sum().item()
            envs = T.nonzero((~T.isfinite(v.reshape(64, -1))).any(dim=1)).squeeze(-1)
            msgs.append(f"states[{k}]: {n} non-finite in envs {envs.tolist()[:8]}")
    if not T.isfinite(obs.rewards).all():
        msgs.append(f"rewards non-finite: {obs.rewards[~T.isfinite(obs.rewards)][:4].tolist()}")
    pol = obs.states["policy"]
    big = pol.abs().amax()
    if msgs:
        bad_steps += 1
        print(f"step {step:3d}: " + "; ".join(msgs), flush=True)
        if bad_steps > 5:
            break
    elif step % 50 == 0:
        print(f"step {step:3d}: all finite. |policy|max={big.item():.2f} "
              f"reward range=({obs.rewards.min().item():.2f},{obs.rewards.max().item():.2f})",
              flush=True)

print(f"DONE bad_steps={bad_steps}", flush=True)
