"""Run the smoke config under torch autograd anomaly mode + finite watchdogs."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import torch as T  # noqa: E402

T.autograd.set_detect_anomaly(True)

from app.logging_config import configure_logging  # noqa: E402
from scripts.agent import build_trainer_from_config, load_config  # noqa: E402
from app.obs_utils import tree_map  # noqa: E402

config = load_config(ROOT / "_tmp_smoke_cam.yml")
configure_logging("INFO", log_dir=config.get("save_dir"))

trainer = build_trainer_from_config(config)
agent = trainer.agent

# Watchdog 1: check every sampled learn batch for non-finite tensors.
orig_learn = agent.learn
def learn_watch(*args, **kwargs):
    def chk(name, x):
        if x is None or isinstance(x, (int, float, bool, str)):
            return
        if isinstance(x, dict):
            for k, v in x.items():
                chk(f"{name}[{k}]", v)
            return
        if T.is_tensor(x) and x.dtype.is_floating_point and not T.isfinite(x).all():
            print(f"!!! NON-FINITE INPUT TO LEARN: {name} "
                  f"({(~T.isfinite(x)).sum().item()} bad of {x.numel()})", flush=True)
    for i, a in enumerate(args):
        chk(f"arg{i}", a)
    for k, v in kwargs.items():
        chk(k, v)
    return orig_learn(*args, **kwargs)
agent.learn = learn_watch

# Watchdog 2: after every optimizer step, verify weights stay finite.
orig_step = agent.model.step
_step_n = [0]
def step_watch(modules=None):
    orig_step(modules)
    _step_n[0] += 1
    for n, p in agent.model.named_parameters():
        if not T.isfinite(p).all():
            grads_msg = []
            for nn_, pp in agent.model.named_parameters():
                if pp.grad is not None and not T.isfinite(pp.grad).all():
                    grads_msg.append(nn_)
            raise RuntimeError(
                f"param '{n}' went non-finite after optimizer step #{_step_n[0]};"
                f" params with non-finite grads this step: {grads_msg[:10]}")
agent.model.step = step_watch

trainer.train()
print("SMOKE TRAIN COMPLETE", flush=True)
