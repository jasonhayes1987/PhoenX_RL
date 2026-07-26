"""Probe: run smoke config, log sigma/log-ratio stats around the exp() blow-up."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import torch as T  # noqa: E402

from app.logging_config import configure_logging  # noqa: E402
from scripts.agent import build_trainer_from_config, load_config  # noqa: E402

config = load_config(ROOT / "_tmp_smoke_cam.yml")
configure_logging("WARNING", log_dir=None)

trainer = build_trainer_from_config(config)
agent = trainer.agent

# --- wrap torch.exp to catch huge/inf log-ratios with context ---------------
_orig_exp = T.exp
_exp_events = []
def exp_probe(x, *a, **k):
    if T.is_tensor(x) and x.is_floating_point():
        with T.no_grad():
            mx = x.max().item() if x.numel() else 0.0
        if mx > 50:
            _exp_events.append(mx)
            print(f"!!! exp() input max={mx:.1f}  numel={x.numel()}  "
                  f"n>50: {(x > 50).sum().item()}", flush=True)
    return _orig_exp(x, *a, **k)
T.exp = exp_probe

# --- wrap model forward: track policy sigma + mu ranges ----------------------
orig_fwd = agent.model.forward
_fwd_n = [0]
def fwd_probe(*args, **kwargs):
    out, hid = orig_fwd(*args, **kwargs)
    _fwd_n[0] += 1
    if isinstance(out, dict) and "policy" in out:
        try:
            d = out["policy"]
            base = d
            while hasattr(base, "base_dist"):
                base = base.base_dist
            while hasattr(base, "dist"):
                base = base.dist
            loc, scale = base.loc, base.scale
            with T.no_grad():
                bad = (~T.isfinite(loc)).sum().item()
                print(f"fwd#{_fwd_n[0]:04d} mu[{loc.abs().max().item():9.3f}] "
                      f"sigma[{scale.min().item():.3e},{scale.max().item():.3e}] "
                      f"nonfinite_mu={bad}", flush=True)
        except Exception as e:
            print(f"fwd probe err: {e}", flush=True)
    return out, hid
agent.model.forward = fwd_probe

# --- discriminator: stash exact rollout model inputs; compare at learn -------
_stash = []
orig_act = agent.act
def act_stash(states, *a, **k):
    if isinstance(states, dict) and len(_stash) < 30:
        _stash.append({k2: v2.clone() for k2, v2 in states.items()})
    return orig_act(states, *a, **k)
agent.act = act_stash

orig_learn3 = agent.learn
def learn_compare(*args, **kwargs):
    samples = args[1] if len(args) > 1 else kwargs.get("samples")
    states = samples.get("states") if isinstance(samples, dict) else None
    if isinstance(states, dict) and _stash:
        # states: (T, N, ...) raw from buffer. Normalize like agent.learn does.
        from app.obs_utils import flatten_leading, tree_index
        flat = flatten_leading(states, 2)
        flat_n = agent.state_normalizer.normalize(flat)
        n_envs = states["policy"].shape[1]
        for t in (0, 1, 2):
            buf_t = tree_index(flat_n, T.arange(t * n_envs, (t + 1) * n_envs, device="cuda"))
            roll_t = _stash[t]
            diffs = {k2: (buf_t[k2].float() - roll_t[k2].float()).abs().max().item()
                     for k2 in roll_t}
            with T.no_grad():
                out_b, _ = agent.model(buf_t, branches=("policy",))
                out_r, _ = agent.model(roll_t, branches=("policy",))
                # per-root bisect
                for rname, root in agent.model.roots.items():
                    kb = agent.model._run_root(rname, root, buf_t, None, "step")
                    kr = agent.model._run_root(rname, root, roll_t, None, "step")
                    d = (kb - kr).abs().max().item()
                    print(f"  ROOT {rname}: out_diff={d:.4e} "
                          f"buf_norm={kb.norm().item():.2f} roll_norm={kr.norm().item():.2f}",
                          flush=True)
                for k2 in roll_t:
                    tb, tr = buf_t[k2], roll_t[k2]
                    print(f"  KEY {k2}: buf dtype={tb.dtype} shape={tuple(tb.shape)} "
                          f"strides={tb.stride()} | roll dtype={tr.dtype} "
                          f"shape={tuple(tr.shape)} strides={tr.stride()}", flush=True)
            def smin(o):
                b = o["policy"]
                while hasattr(b, "base_dist"): b = b.base_dist
                while hasattr(b, "dist"): b = b.dist
                return b.scale.min().item()
            print(f"CMP t={t} max|buf-roll| {diffs} "
                  f"sigma_min buf={smin(out_b):.3e} roll={smin(out_r):.3e}", flush=True)
    return orig_learn3(*args, **kwargs)
agent.learn = learn_compare

# --- param watchdog after each optimizer step --------------------------------
orig_step = agent.model.step
_step_n = [0]
def step_watch(modules=None):
    orig_step(modules)
    _step_n[0] += 1
    for n, p in agent.model.named_parameters():
        if not T.isfinite(p).all():
            bad_g = [nn_ for nn_, pp in agent.model.named_parameters()
                     if pp.grad is not None and not T.isfinite(pp.grad).all()]
            raise RuntimeError(
                f"param '{n}' non-finite after step #{_step_n[0]}; "
                f"non-finite grads: {bad_g[:8]}")
agent.model.step = step_watch

try:
    trainer.train()
    print("SMOKE TRAIN COMPLETE", flush=True)
except Exception as e:
    print(f"TRAIN FAILED: {type(e).__name__}: {e}", flush=True)
