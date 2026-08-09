"""Reinforcement-learning agents and the factory that rebuilds them from config.

``Agent`` is the abstract base: composite-model plumbing (roots → trunk →
branches), serialization, and recurrent rollout state. Concrete algorithms
(``Reinforce``, ``ActorCritic``, ``PPO``, ``DDPG``, ``TD3``, ``SAC``) subclass
it. ``build_agent`` reconstructs a fresh-tensor agent from a ``{"type",
"config"}`` mapping via ``AGENT_REGISTRY``.
"""

# imports
from abc import ABC, abstractmethod
from typing import Protocol, Optional, Dict, Any, runtime_checkable
from pathlib import Path
from collections import deque
import copy

from .logging_config import get_logger

from .intrinsic_motivation import IntrinsicMotivation
from .models import (
    SubNetwork, Head, ModularModel,
    ValueHead, StochasticDiscreteHead, StochasticContinuousHead,
    DeterministicActorHead, ContinuousQHead, DiscreteQHead,
    build_head, head_from_legacy_model_config, modular_parts_from_config,
    map_legacy_state_dict,
)
from .schedulers import ScheduleWrapper
from .adaptive_kl import AdaptiveKL
from .normalizer import BaseNormalizer, RewardNorm, create_normalizer
from .noise import Noise
from .torch_utils import get_device, move_to_device
from .env_wrapper import EnvWrapper, Action
from .utils import *

import numpy as np

import torch as T
# import torch.nn.functional as F

from phoenx.agent_utils import compute_n_step_return, compute_advantages_and_returns, compute_monte_carlo_returns, compute_q_retrace, grad_norm_from_optimizer, setup_auto_entropy, soft_update
from phoenx.obs_utils import flatten_leading, flatten_obs, tree_index, unflatten_leading


## Base Agent Class ##
class Agent(ABC):
    """Base class for all RL agents.

    Serialization contract (uniform across every agent):
        - ``get_config()``  -> ``{"type", "config"}`` architecture description.
        - ``from_config(config, env)`` -> rebuild architecture (fresh tensors),
          the env injected as a live object.
        - ``save_state(dir)`` / ``load_state(dir)`` -> dump/restore every tensor
          (model weights + optimizers + schedule progress + normalizer stats +
          entropy temperature + intrinsic motivation), driven entirely by the
          class-level component-attribute declarations below.

    Subclasses only declare *which* attributes hold each kind of component; the
    base class handles the (de)serialization uniformly.
    """

    # Attribute names that hold trainable Models (weights + optimizer + schedule).
    MODEL_ATTRS: tuple[str, ...] = ()
    # Target/EMA networks (weights are saved; rebuilt as clones on construction).
    TARGET_ATTRS: tuple[str, ...] = ()
    # BaseNormalizer attributes (running statistics).
    NORMALIZER_ATTRS: tuple[str, ...] = (
        "state_normalizer", "goal_normalizer", "reward_normalizer", "advantage_normalizer",
    )
    # Agent-level ScheduleWrapper attributes (progress persisted via get_state).
    SCHEDULE_ATTRS: tuple[str, ...] = ()
    # IntrinsicMotivation attributes (self-contained sub-artifacts).
    IM_ATTRS: tuple[str, ...] = ("intrinsic_motivation",)

    def __init__(self,
                 save_dir: str = "models/",
                 device: Optional[str | T.device] = None,
                 log_level: str = 'INFO',
                 name: str | None = None,
                 **kwargs
    ):
        """Initialize shared agent bookkeeping and save-path / device setup.

        Args:
            save_dir: Directory used as the agent's default save root.
            device: Torch device; ``None`` resolves via ``get_device``.
            log_level: Logger level name (uppercased before use).
            name: Logger / display name; defaults to the class name.
            **kwargs (Any): Extra attributes set on the instance via ``setattr``.
        """
        self.name = name if name else self.__class__.__name__
        self.logger = get_logger(self.name, level=log_level.upper())
        self.kwargs = kwargs
        try:
            self.save_dir = self._setup_save_dir(save_dir)
            self.device = get_device(device)


            self._diag_freq = None
            self._learn_count = 0
            self._nstep_retrace_stats = deque(maxlen=2048)

            # Recurrent rollout state (used when the composite model has a
            # temporal trunk): the live hidden carried across act() calls, the
            # snapshot taken at the start of the current rollout window, the
            # pre-step hidden of the latest act (R2D2 stored state), and the
            # rolling context window for causal-transformer trunks.
            self._hidden = None
            self._rollout_start_hidden = None
            self._last_pre_hidden = None
            self._ctx_obs = None
            self._ctx_goals = None
            self._ctx_starts = None

            if self.kwargs is not None:
                for key, value in self.kwargs.items():
                    setattr(self, key, value)
           
        except Exception as e:
            self.logger.error(f"Error in Agent init: {e}", exc_info=True)

    def _setup_save_dir(self, save_dir: str):
        """Return the path used as this agent's save directory.

        The base implementation returns ``save_dir`` unchanged.

        Args:
            save_dir: Requested save directory path.

        Returns:
            path (str): Save directory path stored on the agent.
        """
        return save_dir

    def clone(self, device: Optional[str | T.device] = None) -> 'Agent':
        """Create a deep copy of the agent, optionally moving it to a new device.

        Args:
            device: Target device for the clone; ``None`` keeps the current device.

        Returns:
            Cloned agent with components copied and, when requested, moved.
        """
        # Perform a deep copy of the agent
        clone = copy.deepcopy(self)

        if clone.__class__.__name__ == 'HER':
            cloned_agent = clone.agent
        else:
            cloned_agent = clone

        if device:
            # Determine the target device
            target_device = get_device(device)
            # Update the cloned agent's device attribute
            cloned_agent.device = target_device

            # Move composite models (params + optimizer state + device attrs).
            for attr_name in ('model', 'target_model'):
                model = getattr(cloned_agent, attr_name, None)
                if isinstance(model, ModularModel):
                    model.set_device(target_device)

            # Now use move_to_device to handle all remaining tensors/components
            cloned_agent = move_to_device(cloned_agent, target_device)

        if clone.__class__.__name__ == 'HER':
            clone.agent = cloned_agent
        else:
            clone = cloned_agent

        return clone

    def get_nstep_diagnostics(self) -> dict:
        """Return and clear accumulated n-step + retrace boundary diagnostics."""
        if not self._nstep_retrace_stats:
            return {}

        final_cum_c = []
        max_leakage = []

        for stats in self._nstep_retrace_stats:
            final_cum_c.extend(stats.get("done_window_final_cum_c", []))
            max_leakage.extend(stats.get("done_window_max_leakage", []))

        self._nstep_retrace_stats.clear()

        out = {}
        if final_cum_c:
            out["nstep/avg_final_cum_c_on_done_windows"] = float(sum(final_cum_c) / len(final_cum_c))
        if max_leakage:
            out["nstep/max_leakage_in_mask_after_done"] = float(max(max_leakage))

        return out

    def get_config(self):
        """Return the architecture description ``{"type", "config"}``.

        The inner ``config`` holds base fields (``save_dir``, ``name``,
        ``device``); subclasses extend it with algorithm-specific entries.

        Returns:
            payload (dict): Mapping with ``type`` (class name) and ``config``
                (constructor kwargs suitable for ``from_config`` after env
                injection).
        """
        return {
            "type": self.__class__.__name__,
            "config":{
                "save_dir": self.save_dir,
                "name": self.name,
                "device": self.device.type,
            }
        }

    @abstractmethod
    def act(
        self,
        states: np.ndarray | T.Tensor,
        goals: np.ndarray | T.Tensor | None = None,
        context: str = 'train',
        **kwargs: Any
    ) -> Action:
        """Select an action for the given observation(s).

        Args:
            states: Current observation(s).
            goals: Optional goal vector(s) for goal-conditioned envs.
            context: Call context, ``'train'`` or ``'test'``.
            **kwargs: Algorithm-specific extras forwarded by subclasses.

        Returns:
            Action for the environment step.
        """
        raise NotImplementedError("Subclasses must implement act.")

    @abstractmethod
    def learn(self, step:int, sample:dict, **kwargs: Any)->dict:
        """Apply one learning update from a sampled batch.

        Args:
            step: Global training step (used by schedules and logging).
            sample: Batch dict from the replay or rollout buffer.
            **kwargs: Algorithm-specific extras forwarded by subclasses.

        Returns:
            Metrics dict for logging (loss terms and related scalars).
        """
        raise NotImplementedError("Subclasses must implement learn.")

    # ------------------------------------------------------------------ #
    # Composite-model plumbing (roots -> trunk -> branches architecture)
    # ------------------------------------------------------------------ #
    #: shared_update default used when the config doesn't specify one.
    #: On-policy agents override with 'combined'; off-policy with 'critic'.
    DEFAULT_SHARED_UPDATE: str = 'combined'

    def _assemble_model(
        self,
        branches: Dict[str, Head],
        roots: Dict[str, SubNetwork] | None = None,
        trunk: SubNetwork | None = None,
        optimizer_params: dict | None = None,
        lr_scheduler: ScheduleWrapper | None = None,
        shared_update: str | None = None,
    ) -> ModularModel:
        """Wire the branch heads (+ optional roots/trunk) into one composite."""
        env = next(iter(branches.values())).env
        return ModularModel(
            env=env,
            roots=roots,
            trunk=trunk,
            branches=branches,
            optimizer_params=optimizer_params,
            lr_scheduler=lr_scheduler,
            shared_update=shared_update or self.DEFAULT_SHARED_UPDATE,
            device=self.device,
        )

    @staticmethod
    def _check_head(role: str, head, *expected_types, optional: bool = False):
        """Type-enforce an algorithm's required head class(es)."""
        if head is None:
            if optional:
                return
            raise TypeError(f"'{role}' head is required")
        if not isinstance(head, expected_types):
            names = " | ".join(t.__name__ for t in expected_types)
            raise TypeError(
                f"'{role}' must be a {names} (got {type(head).__name__}). "
                f"Legacy Model classes are no longer accepted directly; build the "
                f"matching Head (see app.models.HEAD_REGISTRY) or use the config adapter."
            )

    # Convenience branch access so diagnostics/scripts keep reading
    # ``agent.policy`` etc. (the heads live inside ``self.model``).
    def _branch(self, role: str):
        model = getattr(self, 'model', None)
        if model is None or role not in model.branches:
            return None
        return model.branches[role]

    @property
    def policy(self):
        """Policy branch head, or ``None`` if that role is absent."""
        return self._branch('policy')

    @property
    def value(self):
        """Value branch head, or ``None`` if that role is absent."""
        return self._branch('value')

    @property
    def critic(self):
        """Primary critic branch head, or ``None`` if that role is absent."""
        return self._branch('critic')

    @property
    def critic_b(self):
        """Secondary critic branch head, or ``None`` if that role is absent."""
        return self._branch('critic_b')

    # ------------------------------------------------------------------ #
    # Recurrent rollout support
    # ------------------------------------------------------------------ #
    def reset_hidden(self) -> None:
        """Clear the carried recurrent/context state (called at run start)."""
        self._hidden = None
        self._rollout_start_hidden = None
        self._last_pre_hidden = None
        self._ctx_obs = None
        self._ctx_goals = None
        self._ctx_starts = None

    def _rollout_forward(self, states, goals=None, branches=('policy',), dones=None):
        """Rollout-time forward that carries recurrent/context state.

        ``dones`` (bool (num_envs,) — the previous step's done flags) resets
        the hidden state of freshly reset envs before it is consumed. The
        pre-step hidden actually used is kept in ``self._last_pre_hidden`` so
        off-policy agents can attach it to Actions (R2D2 stored state). For
        feedforward models this is a plain forward.
        """
        if self.model.is_causal and not self.model.is_recurrent:
            return self._context_forward(states, goals, branches, dones)
        if not self.model.is_temporal:
            outputs, _ = self.model(states, goal=goals, branches=branches)
            return outputs
        from phoenx.obs_utils import obs_batch_size
        batch_size = obs_batch_size(states)
        if self._hidden is None:
            self._hidden = self.model.init_hidden(batch_size)
        if dones is not None:
            # Apply the episode-start reset eagerly so the stored pre-step
            # hidden is exactly the state the forward consumes.
            self._hidden = self.model.mask_hidden(self._hidden, dones)
        if self._rollout_start_hidden is None:
            self._rollout_start_hidden = self.model.detach_hidden(self._hidden)
        self._last_pre_hidden = self._hidden
        outputs, new_hidden = self.model(
            states, goal=goals, branches=branches,
            hidden=self._hidden, mode='step',
        )
        self._hidden = self.model.detach_hidden(new_hidden)
        return outputs

    def _context_forward(self, states, goals, branches, dones):
        """Run rolling-window inference for causal-transformer trunks.

        Keeps the last ``context_length`` observations per env and runs the
        heads on the window's final position.
        """
        from phoenx.obs_utils import tree_detach_clone, tree_stack
        window = int(getattr(self, 'context_length', 16))
        if self._ctx_obs is None:
            self._ctx_obs, self._ctx_goals, self._ctx_starts = [], [], []
        num_envs = (next(iter(states.values())).shape[0]
                    if isinstance(states, dict) else states.shape[0])
        start = (dones.bool().to(self.device) if dones is not None
                 else T.zeros(num_envs, dtype=T.bool, device=self.device))
        if not self._ctx_obs:
            start = T.ones(num_envs, dtype=T.bool, device=self.device)
        self._ctx_obs.append(tree_detach_clone(states))
        self._ctx_goals.append(goals.detach().clone() if goals is not None else None)
        self._ctx_starts.append(start)
        if len(self._ctx_obs) > window:
            self._ctx_obs = self._ctx_obs[-window:]
            self._ctx_goals = self._ctx_goals[-window:]
            self._ctx_starts = self._ctx_starts[-window:]

        obs_window = tree_stack(self._ctx_obs, dim=1)              # (E, W, ...)
        goals_window = (T.stack([g for g in self._ctx_goals], dim=1)
                        if self._ctx_goals[0] is not None else None)
        start_mask = T.stack(self._ctx_starts, dim=1)              # (E, W)
        return self.model.forward_context(
            obs_window, goal=goals_window, branches=branches, start_mask=start_mask)

    def _advance_rollout_window(self) -> None:
        """Snapshot the live hidden as the next learn window's start state.

        Called at the end of an on-policy learn when the rollout buffer resets.
        """
        if self.model.is_temporal:
            self._rollout_start_hidden = (
                self.model.detach_hidden(self._hidden) if self._hidden is not None else None
            )

    def _live_env(self) -> EnvWrapper:
        """Return the single live env instance shared by the agent's models."""
        for name in self.MODEL_ATTRS or ("model",):
            model = getattr(self, name, None)
            if model is not None:
                return model.env
        raise AttributeError(f"{self.__class__.__name__} has no model to source env from.")

    @classmethod
    def from_config(cls, config: dict, env: EnvWrapper) -> "Agent":
        """Rebuild an agent (architecture + fresh tensors) from an inner config.

        Every sub-component is reconstructed and the single live ``env`` is
        injected into all models. Tensor state (weights, optimizers, stats,
        entropy temperature) and intrinsic-motivation modules are restored
        separately by [load_state][phoenx.rl_agents.Agent.load_state].

        Accepts both schemas:

        - new: a ``model`` entry (ModularModel config) that is decomposed into
          ``roots`` / ``trunk`` / per-role branch heads;
        - legacy: per-model entries (``policy`` / ``value`` / ``critic`` /
          ``critic_b``) holding legacy Model or Head configs, adapted into
          heads (branches-only composite).

        Args:
            config: Inner agent config (the ``config`` field of ``get_config``).
            env: Live environment injected into every rebuilt model.

        Returns:
            New agent instance with fresh tensors.
        """
        cfg = dict(config)
        if cfg.get("model") is not None:
            model_cfg = cfg.pop("model")
            inner = model_cfg.get("config", model_cfg) if isinstance(model_cfg, dict) else model_cfg
            parts = modular_parts_from_config(inner, env)
            cfg["roots"] = parts["roots"]
            cfg["trunk"] = parts["trunk"]
            for role, head in parts["branches"].items():
                cfg[role] = head
            if parts["optimizer_params"] is not None:
                cfg.setdefault("optimizer_params", parts["optimizer_params"])
            if parts["lr_scheduler"] is not None:
                cfg.setdefault("lr_scheduler", parts["lr_scheduler"])
            if parts["shared_update"]:
                cfg.setdefault("shared_update", parts["shared_update"])
        else:
            for key in ("policy", "critic", "critic_b", "value"):
                if cfg.get(key) is not None:
                    cfg[key] = head_from_legacy_model_config(cfg[key], env)
        for key in ("state_normalizer", "goal_normalizer",
                    "reward_normalizer", "advantage_normalizer"):
            if cfg.get(key) is not None:
                cfg[key] = create_normalizer(cfg[key])
        for key in ("noise", "target_noise"):
            if cfg.get(key) is not None:
                cfg[key] = Noise.create_instance(cfg[key]["type"], **cfg[key]["config"])
        if cfg.get("kl_adapter") is not None:
            cfg["kl_adapter"] = AdaptiveKL(**cfg["kl_adapter"])
        for key in list(cfg):
            if key.endswith("_schedule") and isinstance(cfg.get(key), dict):
                cfg[key] = ScheduleWrapper.from_config(cfg[key])
        # Intrinsic motivation is a self-contained artifact rebuilt in load_state.
        for key in ("intrinsic_motivation"):
            if key in cfg:
                cfg[key] = None
        return cls(**cfg)

    def save_state(self, save_dir: str | Path) -> None:
        """Dump every tensor of the agent under ``save_dir`` (mirrors the tree)."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        for name in self.MODEL_ATTRS + self.TARGET_ATTRS:
            model = getattr(self, name, None)
            if model is not None:
                model.save_state(save_dir / f"{name}.pt")

        for name in self.NORMALIZER_ATTRS:
            normalizer = getattr(self, name, None)
            if normalizer is not None:
                normalizer.save_state(save_dir / "normalizers" / f"{name}.pt")

        for name in self.IM_ATTRS:
            intrinsic = getattr(self, name, None)
            if intrinsic is not None:
                intrinsic.save(save_dir)  # writes save_dir/intrinsic_motivation/...

        extra: dict = {}
        for name in self.SCHEDULE_ATTRS:
            schedule = getattr(self, name, None)
            if schedule is not None:
                extra[name] = schedule.get_state()
        if getattr(self, "auto_entropy_tuning", False):
            extra["log_alpha"] = self.log_alpha.detach().cpu()
            extra["entropy_optimizer"] = self.entropy_optimizer.state_dict()
        kl_adapter = getattr(self, "kl_adapter", None)
        if kl_adapter is not None and hasattr(kl_adapter, "get_state"):
            extra["kl_adapter"] = kl_adapter.get_state()
        if extra:
            T.save(extra, save_dir / "agent_state.pt")

    def _load_legacy_checkpoint(self, model: "ModularModel", attr_name: str,
                                save_dir: Path, load_weights: bool) -> bool:
        """Map legacy per-model checkpoint files onto a composite ``ModularModel``.

        Looks for ``policy.pt``, ``value.pt``, ``critic.pt``, ``critic_b.pt``
        and their ``target_*`` variants under ``save_dir``.

        Args:
            model: Composite model whose branch roles are loaded.
            attr_name: Model attribute being restored (``model`` /
                ``target_model``); a name starting with ``target`` selects the
                ``target_*`` file prefix.
            save_dir: Checkpoint directory.
            load_weights: When ``False``, skip weight tensors and still restore
                optimizer / schedule state where present.

        Returns:
            found (bool): ``True`` if any legacy file was found and applied.
        """
        prefix = "target_" if attr_name.startswith("target") else ""
        found = False
        for role in model.branches.keys():
            path = save_dir / f"{prefix}{role}.pt"
            if not path.exists():
                continue
            found = True
            state = T.load(path, map_location=self.device, weights_only=False)
            if load_weights and state.get("model") is not None:
                mapped = map_legacy_state_dict(state["model"], role)
                missing, unexpected = model.load_state_dict(mapped, strict=False)
                if unexpected:
                    self.logger.warning(
                        f"Legacy checkpoint {path.name}: unexpected keys {unexpected}"
                    )
            opt_state = state.get("optimizer")
            opt = model.optimizers.get(f"branches.{role}")
            if opt_state is not None and opt is not None:
                try:
                    opt.load_state_dict(opt_state)
                except Exception as e:  # param layout drift — weights still loaded
                    self.logger.warning(
                        f"Legacy checkpoint {path.name}: optimizer state not restored ({e})"
                    )
            sched_state = state.get("lr_scheduler")
            sched = model.lr_schedulers.get(f"branches.{role}")
            if sched_state is not None and sched is not None:
                sched.set_state(sched_state)
            ts_state = state.get("temperature_schedule")
            head = model.branches[role] if role in model.branches else None
            ts = getattr(head, "temperature_schedule", None) if head is not None else None
            if ts_state is not None and ts is not None:
                ts.set_state(ts_state)
        if found:
            self.logger.warning(
                f"Loaded legacy per-model checkpoint files into '{attr_name}' "
                f"(deprecated format; re-save to write the composite {attr_name}.pt)."
            )
        return found

    def load_state(self, save_dir: str | Path, load_weights: bool = True) -> None:
        """Restore every tensor written by ``save_state`` (in place).

        Restores the artifacts produced by
        [save_state][phoenx.rl_agents.Agent.save_state]. Falls back to the
        legacy per-model checkpoint layout (``policy.pt`` / ``value.pt`` / ...)
        when the composite ``model.pt`` is absent.

        Args:
            save_dir: Checkpoint directory previously passed to ``save_state``.
            load_weights: When ``False``, skip model weight tensors where the
                composite loader supports that flag.
        """
        save_dir = Path(save_dir)

        for name in self.MODEL_ATTRS + self.TARGET_ATTRS:
            model = getattr(self, name, None)
            path = save_dir / f"{name}.pt"
            if model is not None and path.exists():
                model.load_state(path, load_weights=load_weights)
            elif isinstance(model, ModularModel):
                self._load_legacy_checkpoint(model, name, save_dir, load_weights)

        for name in self.NORMALIZER_ATTRS:
            normalizer = getattr(self, name, None)
            path = save_dir / "normalizers" / f"{name}.pt"
            if normalizer is not None and path.exists():
                normalizer.load_state(path)

        if (save_dir / "intrinsic_motivation" / "config.json").is_file():
            intrinsic = IntrinsicMotivation.load(save_dir, env=self._live_env())
            for name in self.IM_ATTRS:
                setattr(self, name, intrinsic)

        extra_path = save_dir / "agent_state.pt"
        if extra_path.exists():
            extra = T.load(extra_path, map_location=self.device, weights_only=False)
            for name in self.SCHEDULE_ATTRS:
                schedule = getattr(self, name, None)
                if schedule is not None and extra.get(name) is not None:
                    schedule.set_state(extra[name])
            if getattr(self, "auto_entropy_tuning", False) and "log_alpha" in extra:
                with T.no_grad():
                    self.log_alpha.data.copy_(extra["log_alpha"].to(self.device))
                if extra.get("entropy_optimizer") is not None:
                    self.entropy_optimizer.load_state_dict(extra["entropy_optimizer"])
            kl_adapter = getattr(self, "kl_adapter", None)
            if kl_adapter is not None and hasattr(kl_adapter, "set_state") and "kl_adapter" in extra:
                kl_adapter.set_state(extra["kl_adapter"])

class Reinforce(Agent):
    """REINFORCE (Monte Carlo policy gradient) on-policy agent.

    Assembles a composite ``ModularModel`` with a required stochastic discrete
    policy head and an optional value baseline. Temporal trunks are rejected at
    construction; use `ActorCritic` or `PPO` for recurrent / causal policies.

    Learning uses Monte Carlo returns via
    [compute_monte_carlo_returns][phoenx.agent_utils.compute_monte_carlo_returns].
    Policy and value losses share one combined backward and a coordinated
    ``model.step()`` (on-policy gradient ownership).
    """

    MODEL_ATTRS = ("model",)
    SCHEDULE_ATTRS = ("entropy_schedule",)
    IM_ATTRS = ()
    DEFAULT_SHARED_UPDATE = 'combined'

    def __init__(
        self,
        roots: Dict[str, SubNetwork] | None = None,
        trunk: SubNetwork | None = None,
        policy: StochasticDiscreteHead | None = None,
        value: ValueHead | None = None,
        optimizer_params: dict | None = None,
        lr_scheduler: ScheduleWrapper | None = None,
        shared_update: str | None = None,
        discount: float = 0.99,
        state_normalizer: BaseNormalizer|None = None,
        advantage_normalizer: BaseNormalizer|None = None,
        reward_normalizer: RewardNorm|None = None,
        entropy_coefficient: float = 0.01,
        entropy_schedule: ScheduleWrapper|None = None,
        auto_entropy_tuning: bool=True,
        entropy_lr: float=3e-4, # Only used if auto entropy = True
        target_entropy_scale: float=0.98, # Only used if auto entropy = True and discrete action space
        save_dir: str = "models",
        device: str = None,
        **kwargs,
    ):
        """Initialize the REINFORCE agent and assemble its composite model.

        Args:
            roots: Optional per-modality encoder SubNetworks (name -> SubNetwork).
            trunk: Optional shared fusion SubNetwork.
            policy: Stochastic discrete policy head used for action selection
                (required).
            value: Optional value head for a learned baseline; when present,
                advantages are ``return - value``, otherwise returns weight the
                policy gradient directly.
            optimizer_params: Model-wide default optimizer spec.
            lr_scheduler: Model-wide default LR scheduler.
            shared_update: Gradient-ownership rule for shared modules; defaults
                to ``'combined'`` via ``DEFAULT_SHARED_UPDATE``.
            discount: Discount factor for Monte Carlo returns.
            state_normalizer: Optional observation normalizer applied in
                ``learn``.
            advantage_normalizer: Optional normalizer for the policy-weight
                tensor (advantages or returns).
            reward_normalizer: Optional reward normalizer applied before return
                computation.
            entropy_coefficient: Fixed entropy bonus weight when
                ``auto_entropy_tuning`` is ``False``.
            entropy_schedule: Optional multiplier schedule for
                ``entropy_coefficient`` when auto-tuning is off.
            auto_entropy_tuning: When ``True``, learn ``log_alpha`` toward a
                target entropy.
            entropy_lr: Learning rate for the entropy temperature optimizer.
            target_entropy_scale: Scale used when building the target entropy
                (mainly for discrete action spaces).
            save_dir: Default save directory passed to the base agent.
            device: Torch device string; ``None`` resolves via ``get_device``.
            **kwargs (Any): Extra attributes set on the instance via the base
                ``Agent`` constructor.
        """
        super().__init__(save_dir, device, **kwargs)
        self._check_head('policy', policy, StochasticDiscreteHead)
        self._check_head('value', value, ValueHead, optional=True)
        branches: Dict[str, Head] = {'policy': policy}
        if value is not None:
            branches['value'] = value
        self.model = self._assemble_model(
            branches, roots=roots, trunk=trunk,
            optimizer_params=optimizer_params, lr_scheduler=lr_scheduler,
            shared_update=shared_update,
        )
        if self.model.is_temporal:
            raise NotImplementedError(
                "Reinforce does not support temporal (recurrent/causal) trunks; "
                "use ActorCritic or PPO for recurrent policies."
            )
        self.discount = discount
        self.state_normalizer = state_normalizer
        self.advantage_normalizer = advantage_normalizer
        self.reward_normalizer = reward_normalizer
        self.entropy_coefficient = entropy_coefficient
        self.entropy_schedule = entropy_schedule
        self.auto_entropy_tuning = auto_entropy_tuning
        self.entropy_lr = entropy_lr
        self.target_entropy_scale = target_entropy_scale
        if self.auto_entropy_tuning:
            self.target_entropy, self.log_alpha, self.entropy_optimizer = setup_auto_entropy(
                self.policy,
                target_entropy_scale=target_entropy_scale,
                lr=entropy_lr,
                device=self.device,
            )

    def act(
        self,
        states: np.ndarray | T.Tensor,
        goals: np.ndarray | T.Tensor | None = None,
        context: str = 'train',
        **kwargs: Any
    ) -> Action:
        """Select an action from the current policy distribution.

        In ``'train'`` context samples from the policy; in ``'test'`` uses the
        policy mean / mode via ``get_mean_actions``. Forwards optional episode
        done flags into `_rollout_forward` for recurrent reset (unused for this
        feedforward-only agent, but accepted for API parity).

        Args:
            states: Current observation(s).
            goals: Optional goal vector(s) for goal-conditioned envs.
            context: ``'train'`` to sample, ``'test'`` for deterministic actions.
            **kwargs: May include ``dones`` (previous-step done flags) forwarded
                to `_rollout_forward`.

        Returns:
            action (`Action`): Package with ``actions`` and ``log_probs``;
                ``raw_actions`` and ``hidden`` are always ``None``.
        """
        with T.no_grad():
            outputs = self._rollout_forward(
                states, goals=goals, branches=('policy',), dones=kwargs.get('dones'))
            dist = outputs['policy']
            if context == 'train':
                actions = dist.sample()
                log_probs = dist.log_prob(actions)

            elif context == 'test':
                actions = self.policy.get_mean_actions(dist)
                log_probs = dist.log_prob(actions)

            else:
                raise ValueError(f"Invalid context: {context}")

        return Action(actions, log_probs=log_probs)

    def learn(self, step: int, sample: list[dict], **kwargs: Any)->dict:
        """Apply one REINFORCE update from completed trajectories.

        Concatenates trajectory tensors, computes Monte Carlo returns, runs a
        single shared forward through policy (and value when present), then
        combines policy and value losses into one backward plus ``model.step()``.
        Auto-entropy, when enabled, updates ``log_alpha`` in a separate step.

        Args:
            step: Global training step (diagnostics / logging).
            sample: List of trajectory dicts with ``states``, ``actions``, and
                ``rewards`` tensors.
            **kwargs: Unused; accepted for ``Agent.learn`` API compatibility.

        Returns:
            Metrics dict (policy/value loss, advantages, returns, entropy, LRs).
        """
        self._learn_count += 1
        if self._diag_freq is not None:
            should_log_diag = (self._learn_count % self._diag_freq == 0)
        else:
            should_log_diag = False
        
        learn_metrics = {}

        all_states = [trajectory['states'] for trajectory in sample]
        all_actions = [trajectory['actions'] for trajectory in sample]
        all_rewards = [trajectory['rewards'] for trajectory in sample]
        # all_terminations = [trajectory['terminations'] for trajectory in completed_trajectories]
        # all_truncations = [trajectory['truncations'] for trajectory in completed_trajectories]

        for i, rewards in enumerate(all_rewards):
            if self.reward_normalizer:
                rewards = self.reward_normalizer.normalize(rewards)
            all_rewards[i] = rewards
        
        all_returns = [compute_monte_carlo_returns(rewards, self.discount, device=self.device) for rewards in all_rewards]

        # # Iterate over completed trajectories
        # for trajectory in completed_trajectories:
        #     all_states.append(trajectory['states'])
        #     all_actions.append(trajectory['actions'])
        #     # _return = compute_monte_carlo_returns(trajectory['rewards'], self.discount, device=self.device)
        #     # all_returns.append(_return)
        #     all_rewards.append(trajectory['rewards'])
            

        # Use T.cat to concatenate all tensors in list into single tensor of shape [total_steps, obs_dim]
        states = T.cat(all_states, dim=0)
        actions = T.cat(all_actions, dim=0)
        returns = T.cat(all_returns, dim=0).unsqueeze(-1)

        # Normalize states if using normalizer
        if self.state_normalizer:
            states = self.state_normalizer.normalize(states)

        # Clear gradients on every module optimizer
        self.model.zero_grad()

        # Single shared forward through roots/trunk into the requested heads
        branch_roles = ('policy', 'value') if self.value is not None else ('policy',)
        outputs, _ = self.model(states, branches=branch_roles)
        dist = outputs['policy']

        # Calculate advantages and value loss if using value function
        if self.value is not None:
            values = outputs['value']
            advantages = returns.detach() - values
            value_loss = advantages.pow(2).mean()
        else:
            values = T.zeros_like(returns)
            advantages = T.zeros_like(returns)
            value_loss = 0

        # Calculate policy loss
        # Create policy_weight value based on advantages (if value) or returns
        if self.value is not None:
            policy_weight = advantages.detach()
        else:
            policy_weight = returns.detach()

        if self.advantage_normalizer:
            if getattr(self.advantage_normalizer, 'add', None):
                self.advantage_normalizer.add(policy_weight)
            policy_weight = self.advantage_normalizer.normalize(policy_weight)

        log_probs = dist.log_prob(actions.squeeze(-1)).unsqueeze(-1)
        entropies = dist.entropy().unsqueeze(-1)

        # Get entropy coefficient
        if self.auto_entropy_tuning:
            entropy_coefficient = self.log_alpha.exp()
        else:
            entropy_coefficient = self.entropy_coefficient
            if self.entropy_schedule:
                entropy_coefficient *= self.entropy_schedule.get_factor()

        # Get policy loss
        policy_loss = -(log_probs * policy_weight + entropy_coefficient * entropies).mean()

        # ONE combined backward; every module optimizer then steps exactly once
        # (equivalent to a single optimizer over the whole composite).
        total_loss = policy_loss + value_loss
        total_loss.backward()

        if should_log_diag:
            self.logger.debug(
                "ac_diag step=%d learn_count=%d %s %s %s %s %s %s %s %s %s",
                step,
                self._learn_count,
                summarize_tensor(states, "states"),
                summarize_tensor(actions, "actions"),
                summarize_tensor(rewards, "rewards"),
                # summarize_tensor(terminations, "terminations"),
                # summarize_tensor(truncations, "truncations"),
                summarize_tensor(values, "values"),
                summarize_tensor(advantages, "advantages"),
                summarize_tensor(returns, "returns"),
                summarize_tensor(log_probs, "log_probs"),
                summarize_tensor(entropies, "entropies"),
                f"entropy_coef={float(entropy_coefficient)}",
            )
            value_opt = self.model.optimizers.get('branches.value')
            value_grad_norm = grad_norm_from_optimizer(value_opt) if value_opt else 0.0
            policy_grad_norm = grad_norm_from_optimizer(self.model.optimizers['branches.policy'])
            self.logger.debug(
                "ac_grads step=%d learn_count=%d value_grad_norm=%.6f policy_grad_norm=%.6f "
                "value_loss=%.6f policy_loss=%.6f",
                step,
                self._learn_count,
                value_grad_norm,
                policy_grad_norm,
                float(value_loss.item()),
                float(policy_loss.item()),
            )

        # Update weights (all module optimizers step once on the combined grads)
        self.model.step()

        if self.auto_entropy_tuning:
            self.entropy_optimizer.zero_grad()
            if self.policy.distribution in ['normal', 'beta']:
                alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            else: # Discrete actor
                alpha_loss = -(self.log_alpha * ((dist.probs * log_probs).sum(dim=-1) + self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.entropy_optimizer.step()

        policy_learning_rate = self.model.learning_rate('branches.policy')
        value_learning_rate = self.model.learning_rate('branches.value')

        # Get temperature value from policy if categorical
        if self.policy.distribution == 'categorical':
            temperature = self.policy.temperature
            if self.policy.temperature_schedule:
                temperature *= self.policy.temperature_schedule.get_factor()
            learn_metrics.update({'temperature': temperature})

        learn_metrics.update({
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'advantages': advantages.mean().item(),
            'returns': returns.mean().item(),
            'entropy': entropies.mean().item(),
            'entropy_coefficient': entropy_coefficient,
            'policy_learning_rate': policy_learning_rate,
            'value_learning_rate': value_learning_rate,
        })

        return learn_metrics

    def get_config(self):
        """Return the architecture description ``{"type", "config"}``.

        Extends the base payload with the composite model config, discount,
        normalizers, and entropy-tuning fields.

        Returns:
            payload (dict): Mapping with ``type`` (class name) and ``config``
                (constructor kwargs suitable for ``from_config`` after env
                injection).
        """
        config = super().get_config()
        config["config"].update({
            "model": self.model.get_config(),
            "discount": self.discount,
            "state_normalizer": self.state_normalizer.get_config() if self.state_normalizer is not None else None,
            "advantage_normalizer": self.advantage_normalizer.get_config() if self.advantage_normalizer is not None else None,
            "reward_normalizer": self.reward_normalizer.get_config() if self.reward_normalizer is not None else None,
            "entropy_coefficient": self.entropy_coefficient,
            "entropy_schedule": self.entropy_schedule.get_config() if self.entropy_schedule is not None else None,
            "auto_entropy_tuning": self.auto_entropy_tuning,
            "entropy_lr": self.entropy_lr,
            "target_entropy_scale": self.target_entropy_scale,
        })
        return config

class ActorCritic(Agent):
    """Advantage actor-critic (A2C-style) on-policy agent.

    Assembles a composite ``ModularModel`` with required stochastic policy and
    value heads (roots → trunk → branches). Supports feedforward and temporal
    trunks; the temporal path uses `_recurrent_update` with the rollout-start
    hidden and GAE via
    [compute_advantages_and_returns][phoenx.agent_utils.compute_advantages_and_returns].

    Policy and value losses share one combined backward and a coordinated
    ``model.step()`` (on-policy gradient ownership).
    """

    MODEL_ATTRS = ("model",)
    SCHEDULE_ATTRS = ("entropy_schedule",)
    IM_ATTRS = ()
    DEFAULT_SHARED_UPDATE = 'combined'

    def __init__(
        self,
        roots: Dict[str, SubNetwork] | None = None,
        trunk: SubNetwork | None = None,
        policy: StochasticDiscreteHead | StochasticContinuousHead | None = None,
        value: ValueHead | None = None,
        optimizer_params: dict | None = None,
        lr_scheduler: ScheduleWrapper | None = None,
        shared_update: str | None = None,
        discount: float=0.99,
        state_normalizer: BaseNormalizer|None = None,
        goal_normalizer: BaseNormalizer|None = None,
        advantage_normalizer: BaseNormalizer|None = None,
        reward_normalizer: RewardNorm|None = None,
        entropy_coefficient: float=0.01, # Only used if auto entropy = False
        entropy_schedule: ScheduleWrapper|None = None, # Only used if auto entropy = False
        auto_entropy_tuning: bool=True,
        entropy_lr: float=3e-4, # Only used if auto entropy = True
        target_entropy_scale: float=0.98, # Only used if auto entropy = True and discrete action space
        gae_coefficient: float=0.95,
        policy_grad_clip: float=1.0,
        value_grad_clip: float=1.0,
        shared_grad_clip: float|None = None, # None -> max(policy, value) clip
        value_coef: float=0.5,
        bootstrap_truncations: bool=True,
        save_dir: str = "models/",
        device: Optional[str | T.device] = None,
        **kwargs,
    ):
        """Initialize the actor-critic agent and assemble its composite model.

        Args:
            roots: Optional per-modality encoder SubNetworks (name -> SubNetwork).
            trunk: Optional shared fusion SubNetwork (may hold temporal layers).
            policy: Stochastic discrete or continuous policy head (required).
            value: Value head for state-value / GAE baselines (required).
            optimizer_params: Model-wide default optimizer spec.
            lr_scheduler: Model-wide default LR scheduler.
            shared_update: Gradient-ownership rule for shared modules; defaults
                to ``'combined'`` via ``DEFAULT_SHARED_UPDATE``.
            discount: Discount factor ``gamma`` for TD / GAE.
            state_normalizer: Optional observation normalizer.
            goal_normalizer: Optional goal normalizer for goal-conditioned envs.
            advantage_normalizer: Optional normalizer for policy advantages.
            reward_normalizer: Optional reward normalizer.
            entropy_coefficient: Fixed entropy bonus weight when
                ``auto_entropy_tuning`` is ``False``.
            entropy_schedule: Optional multiplier schedule for
                ``entropy_coefficient`` when auto-tuning is off.
            auto_entropy_tuning: When ``True``, learn ``log_alpha`` toward a
                target entropy.
            entropy_lr: Learning rate for the entropy temperature optimizer.
            target_entropy_scale: Scale used when building the target entropy
                (mainly for discrete action spaces).
            gae_coefficient: GAE lambda for advantage estimation.
            policy_grad_clip: Max grad norm for policy-branch modules; falsy
                skips clipping.
            value_grad_clip: Max grad norm for value-branch modules; falsy
                skips clipping.
            shared_grad_clip: Max grad norm for shared roots/trunk modules;
                ``None`` uses ``max(policy_grad_clip, value_grad_clip)`` (with
                missing clips treated as infinity).
            value_coef: Multiplier on the squared value loss.
            bootstrap_truncations: Whether truncated episodes bootstrap from
                next-state values in GAE / TD targets.
            save_dir: Default save directory passed to the base agent.
            device: Torch device; ``None`` resolves via ``get_device``.
            **kwargs (Any): Extra attributes set on the instance via the base
                ``Agent`` constructor.
        """
        super().__init__(save_dir, device, **kwargs)
        self._check_head('policy', policy, StochasticDiscreteHead, StochasticContinuousHead)
        self._check_head('value', value, ValueHead)
        self.model = self._assemble_model(
            {'policy': policy, 'value': value}, roots=roots, trunk=trunk,
            optimizer_params=optimizer_params, lr_scheduler=lr_scheduler,
            shared_update=shared_update,
        )
        self.discount = discount
        self.state_normalizer = state_normalizer
        self.goal_normalizer = goal_normalizer
        self.advantage_normalizer = advantage_normalizer
        self.reward_normalizer = reward_normalizer
        self.entropy_coefficient = entropy_coefficient
        self.entropy_schedule = entropy_schedule
        self.auto_entropy_tuning = auto_entropy_tuning
        self.entropy_lr = entropy_lr
        self.target_entropy_scale = target_entropy_scale
        if self.auto_entropy_tuning:
            self.target_entropy, self.log_alpha, self.entropy_optimizer = setup_auto_entropy(
                self.policy,
                target_entropy_scale=target_entropy_scale,
                lr=entropy_lr,
                device=self.device,
            )
        self.gae_coefficient = gae_coefficient
        self.policy_grad_clip = policy_grad_clip
        self.value_grad_clip = value_grad_clip
        self.shared_grad_clip = shared_grad_clip
        self.value_coef = value_coef
        self.bootstrap_truncations = bootstrap_truncations

    def act(
        self,
        states: np.ndarray | T.Tensor,
        goals: np.ndarray | T.Tensor | None = None,
        context: str = 'train',
        **kwargs: Any
    ) -> Action:
        """Select an action from the current policy distribution.

        In ``'train'`` context samples from the policy; in ``'test'`` uses the
        policy mean / mode: ``get_mean_actions`` on the categorical branch, or
        ``dist.mean_with_z()`` on the continuous branch. Optional ``dones`` in
        ``kwargs`` reset recurrent / context state for freshly finished envs.

        Args:
            states: Current observation(s).
            goals: Optional goal vector(s) for goal-conditioned envs.
            context: ``'train'`` to sample, ``'test'`` for deterministic actions.
            **kwargs: May include ``dones`` (previous-step done flags) forwarded
                to `_rollout_forward`.

        Returns:
            action (`Action`): Package with ``actions`` and ``log_probs``;
                ``raw_actions`` is populated on the continuous branch and
                ``None`` on the categorical branch. ``hidden`` is always
                ``None``.
        """
        raw_actions = None
        with T.no_grad():
            outputs = self._rollout_forward(
                states, goals=goals, branches=('policy',), dones=kwargs.get('dones'))
            dist = outputs['policy']
            if context == 'train':
                if self.policy.distribution == 'categorical':
                    actions = dist.sample()
                    log_probs = dist.log_prob(actions)
                else: # Continuous
                    actions, raw_actions = dist.sample_with_z()
                    log_probs = dist.log_prob_from_z(raw_actions)
            elif context == 'test':
                if self.policy.distribution == 'categorical':
                    actions = self.policy.get_mean_actions(dist)
                    log_probs = dist.log_prob(actions)
                else: # Continuous
                    actions, raw_actions = dist.mean_with_z()
                    log_probs = dist.log_prob_from_z(raw_actions)

            else:
                raise ValueError(f"Invalid context: {context}")

        return Action(actions, raw_actions=raw_actions, log_probs=log_probs)

    def learn(self, step:int, sample:dict, **kwargs: Any)->dict:
        """Apply one actor-critic update from a rollout-buffer sample.

        Normalizes observations / goals / rewards when configured, estimates
        GAE advantages and returns, then combines value and policy losses into
        one backward plus ``model.step()``. Temporal models delegate to
        `_recurrent_update` instead of the flat feedforward path.

        Args:
            step: Global training step (diagnostics / logging).
            sample: Rollout batch with ``states``, ``actions``, ``rewards``,
                ``next_states``, terminations / truncations, ``valid_indices``,
                and goal fields.
            **kwargs: Unused; accepted for ``Agent.learn`` API compatibility.

        Returns:
            Metrics dict (losses, TD error, advantages, returns, entropy, LRs).
        """
        self._learn_count += 1
        if self._diag_freq is not None:
            should_log_diag = (self._learn_count % self._diag_freq == 0)
        else:
            should_log_diag = False

        learn_metrics = {}

        self.model.zero_grad()

        # Extract trajectories from buffer
        states = sample['states']
        actions = sample['actions']
        rewards = sample['rewards']
        next_states = sample['next_states']
        terminations = sample['terminations']
        truncations = sample['truncations']
        # first_steps = sample["first_steps"]
        valid_indices = sample["valid_indices"]
        ach_goals = sample["state_achieved_goals"]
        next_ach_goals = sample["next_state_achieved_goals"]
        goals = sample["desired_goals"]

        # Normalize states and goals
        if self.state_normalizer:
            states = self.state_normalizer.normalize(states)
            next_states = self.state_normalizer.normalize(next_states)
        if self.goal_normalizer:
            goals = self.goal_normalizer.normalize(goals)
            ach_goals = self.goal_normalizer.normalize(ach_goals)
        if self.reward_normalizer:
            rewards = self.reward_normalizer.normalize(rewards)

        # Get entropy coefficient
        if self.auto_entropy_tuning:
            entropy_coefficient = self.log_alpha.exp()
        else:
            entropy_coefficient = self.entropy_coefficient
            if self.entropy_schedule:
                entropy_coefficient *= self.entropy_schedule.get_factor()

        # Get trajectory length, num_envs, and feature dims
        # trajectory_length, num_envs, obs_dim = states.shape
        # action_dim = actions.shape[-1]
        traj_len, num_envs = rewards.shape
        total_samples = traj_len * num_envs

        # Flatten trajectory data (feature shapes preserved; dict obs per key)
        states_flat = flatten_leading(states, 2)
        next_states_flat = flatten_leading(next_states, 2)
        actions_flat = actions.reshape(total_samples, -1)
        goals_flat = goals.reshape(total_samples, -1) if goals is not None else None

        # Recurrent path: hidden-aware full-batch sequence update
        if self.model.is_temporal:
            learn_metrics.update(self._recurrent_update(
                step=step,
                states_flat=states_flat, next_states_flat=next_states_flat,
                actions_flat=actions_flat, goals_flat=goals_flat,
                rewards=rewards, terminations=terminations, truncations=truncations,
                first_steps=sample["first_steps"], traj_len=traj_len, num_envs=num_envs,
                entropy_coefficient=entropy_coefficient,
            ))
            return learn_metrics

        value_out, _ = self.model(states_flat, goal=goals_flat, branches=('value',))
        state_values = value_out['value'].reshape(traj_len, num_envs)
        with T.no_grad():
            next_value_out, _ = self.model(next_states_flat, goal=goals_flat, branches=('value',))
            next_state_values = next_value_out['value'].reshape(traj_len, num_envs)

        advantages, returns, td_errors = compute_advantages_and_returns(
            rewards,
            state_values,
            next_state_values,
            terminations,
            truncations,
            self.discount,
            self.gae_coefficient,
            self.bootstrap_truncations,
            device=self.device
        )

        # Filter phantom steps
        valid_idx = valid_indices.squeeze(-1)
        states_flat = tree_index(states_flat, valid_idx)
        next_states_flat = tree_index(next_states_flat, valid_idx)
        actions_flat = actions_flat[valid_idx]
        goals_flat = goals_flat[valid_idx] if goals is not None else None
        state_values_flat = state_values.reshape(total_samples)[valid_idx]
        next_state_values_flat = next_state_values.reshape(total_samples)[valid_idx]
        advantages_flat = advantages.reshape(total_samples)[valid_idx]
        returns_flat = returns.reshape(total_samples)[valid_idx]
        td_errors_flat = td_errors.reshape(total_samples)[valid_idx]

        # Calculate value loss
        value_loss = self.value_coef * (state_values_flat - returns_flat.detach()).pow(2).mean()

        # Create separate policy advantage in case using advantage normalizer
        policy_advantages = advantages_flat.detach()
        if self.advantage_normalizer:
            policy_advantages = policy_advantages.reshape(-1,1)
            if getattr(self.advantage_normalizer, 'add', None):
                self.advantage_normalizer.add(policy_advantages)
            policy_advantages = self.advantage_normalizer.normalize(policy_advantages).reshape(-1)

        # Get log probs and entropy values from current policy dist
        policy_out, _ = self.model(states_flat, goal=goals_flat, branches=('policy',))
        dist = policy_out['policy']
        # reshape flat actions to be vector if categorical distribution
        if self.policy.distribution == 'categorical':
            actions_flat = actions_flat.squeeze(-1)
        log_probs = dist.log_prob(actions_flat).flatten()#.reshape(traj_len, num_envs)

        entropies = dist.entropy().flatten()#.reshape(traj_len, num_envs)

        # Calculate policy loss
        policy_loss = -(log_probs * policy_advantages + entropy_coefficient * entropies).mean()

        # ONE combined backward over both losses; shared modules (if any)
        # accumulate gradients from both, applied exactly once by step().
        (value_loss + policy_loss).backward()
        # Clip gradients if grad clips (per branch, plus shared modules)
        value_grad_norm = None
        policy_grad_norm = None
        if self.value_grad_clip:
            value_grad_norm = self.model.clip(self.value_grad_clip, modules=self.model.branch_module_names('value'))
        if self.policy_grad_clip:
            policy_grad_norm = self.model.clip(self.policy_grad_clip, modules=self.model.branch_module_names('policy'))
        shared_modules = self.model.shared_module_names()
        if shared_modules:
            shared_clip = self.shared_grad_clip
            if shared_clip is None:
                shared_clip = max(self.policy_grad_clip or float('inf'),
                                  self.value_grad_clip or float('inf'))
            self.model.clip(shared_clip, modules=shared_modules)

        nonfinite_values = (
            count_nonfinite(state_values)
            + count_nonfinite(next_state_values)
            + count_nonfinite(td_errors)
            + count_nonfinite(advantages)
            + count_nonfinite(returns)
            + count_nonfinite(log_probs)
            + count_nonfinite(entropies)
        )

        if should_log_diag or nonfinite_values > 0:
            self.logger.debug(
                "ac_diag step=%d learn_count=%d %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s",
                step,
                self._learn_count,
                summarize_tensor(states, "states"),
                summarize_tensor(actions, "actions"),
                summarize_tensor(rewards, "rewards"),
                summarize_tensor(next_states, "next_states"),
                summarize_tensor(goals, "goals"),
                summarize_tensor(next_ach_goals, "next_ach_goals"),
                summarize_tensor(terminations, "terminations"),
                summarize_tensor(truncations, "truncations"),
                summarize_tensor(state_values_flat, "values"),
                summarize_tensor(next_state_values_flat, "next_values"),
                summarize_tensor(td_errors_flat, "td_errors"),
                summarize_tensor(advantages_flat, "advantages"),
                summarize_tensor(returns_flat, "returns"),
                summarize_tensor(log_probs, "log_probs"),
                summarize_tensor(entropies, "entropies"),
                f"entropy_coef={float(entropy_coefficient)}",
            )
        
            self.logger.debug(
                "ac_grads step=%d learn_count=%d value_grad_norm=%.6f policy_grad_norm=%.6f "
                "value_loss=%.6f policy_loss=%.6f",
                step,
                self._learn_count,
                float(value_grad_norm) if value_grad_norm is not None else -1.0,
                float(policy_grad_norm) if policy_grad_norm is not None else -1.0,
                float(value_loss.item()),
                float(policy_loss.item()),
            )

        self.model.step()

        if self.auto_entropy_tuning:
            self.entropy_optimizer.zero_grad()
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.entropy_optimizer.step()

        policy_learning_rate = self.model.learning_rate('branches.policy')
        value_learning_rate = self.model.learning_rate('branches.value')

        # Get temperature value from policy if categorical
        if self.policy.distribution == 'categorical':
            temperature = self.policy.temperature
            if self.policy.temperature_schedule:
                temperature *= self.policy.temperature_schedule.get_factor()
            learn_metrics.update({'temperature': temperature})

        learn_metrics.update({
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'temporal_difference': td_errors_flat.mean().item(),
            'advantages': advantages_flat.mean().item(),
            'returns': returns_flat.mean().item(),
            'entropy': entropies.mean().item(),
            'entropy_coefficient': entropy_coefficient,
            'policy_learning_rate': policy_learning_rate,
            'value_learning_rate': value_learning_rate,
        })

        return learn_metrics

    def _recurrent_update(
        self,
        *,
        step: int,
        states_flat, next_states_flat, actions_flat, goals_flat,
        rewards, terminations, truncations, first_steps,
        traj_len: int, num_envs: int,
        entropy_coefficient,
    ) -> dict:
        """Hidden-aware full-batch ActorCritic update over env sequences.

        Values and the policy distribution come from one sequence forward with
        the rollout-start hidden (episode boundaries reset it mid-sequence).
        Next-state values for TD targets are the shifted values from the same
        stream (exact within episodes); the final step's next-value uses the
        post-sequence hidden. Losses share one combined backward and
        ``model.step()``.

        Args:
            step (int): Global training step (diagnostics / logging).
            states_flat: Flattened observations ``(T * E, ...)``.
            next_states_flat: Flattened next observations.
            actions_flat: Flattened actions ``(T * E, action_dim)``.
            goals_flat: Flattened goals, or ``None``.
            rewards: Rewards shaped ``(T, E)``.
            terminations: Termination flags ``(T, E)``.
            truncations: Truncation flags ``(T, E)``.
            first_steps: Episode-start / phantom-step mask ``(T, E)``.
            traj_len (int): Rollout length ``T``.
            num_envs (int): Parallel environment count ``E``.
            entropy_coefficient: Scalar entropy weight (tensor or float).

        Returns:
            Metrics dict matching the feedforward ``learn`` path, after
            advancing the rollout-start hidden via `_advance_rollout_window`.
        """
        learn_metrics: dict = {}
        from phoenx.obs_utils import tree_map as _tree_map

        def to_seq(flat):
            seq = unflatten_leading(flat, (traj_len, num_envs))
            return _tree_map(lambda x: x.transpose(0, 1).contiguous(), seq)

        states_seq = to_seq(states_flat)
        next_states_seq = to_seq(next_states_flat)
        actions_seq = actions_flat.reshape(traj_len, num_envs, -1).transpose(0, 1).contiguous()
        goals_seq = (goals_flat.reshape(traj_len, num_envs, -1).transpose(0, 1).contiguous()
                     if goals_flat is not None else None)
        start_mask = first_steps.transpose(0, 1).bool().to(self.device)
        valid_mask = (~start_mask).float()
        valid_count = valid_mask.sum().clamp(min=1.0)

        hidden0 = self._rollout_start_hidden
        if hidden0:
            hb = next(iter(hidden0.values()))
            hb = hb[0] if isinstance(hb, tuple) else hb
            if hb.shape[1] != num_envs:
                hidden0 = None
        if not hidden0:
            hidden0 = self.model.init_hidden(num_envs)

        outputs, final_hidden = self.model(
            states_seq, goal=goals_seq, branches=('policy', 'value'),
            hidden=hidden0, start_mask=start_mask, mode='sequence')
        dist = outputs['policy']
        values_et = outputs['value'].squeeze(-1)                       # (E, T), grad

        with T.no_grad():
            last_next = tree_index(next_states_seq, (slice(None), -1))
            goals_last = goals_seq[:, -1] if goals_seq is not None else None
            last_out, _ = self.model(last_next, goal=goals_last, branches=('value',),
                                     hidden=self.model.detach_hidden(final_hidden), mode='step')
            next_values_et = T.cat(
                [values_et.detach()[:, 1:], last_out['value'].reshape(num_envs, 1)], dim=1)

        advantages_tm, returns_tm, td_errors_tm = compute_advantages_and_returns(
            rewards, values_et.transpose(0, 1), next_values_et.transpose(0, 1),
            terminations, truncations, self.discount, self.gae_coefficient,
            self.bootstrap_truncations, self.device)
        advantages_et = advantages_tm.transpose(0, 1)
        returns_et = returns_tm.transpose(0, 1)

        value_loss = self.value_coef * (
            ((values_et - returns_et.detach()).pow(2) * valid_mask).sum() / valid_count)

        policy_advantages = advantages_et.detach()
        if self.advantage_normalizer:
            valid_flat = policy_advantages[valid_mask.bool()].reshape(-1, 1)
            if getattr(self.advantage_normalizer, 'add', None):
                self.advantage_normalizer.add(valid_flat)
            policy_advantages = self.advantage_normalizer.normalize(
                policy_advantages.reshape(-1, 1)).reshape(num_envs, traj_len)

        if self.policy.distribution == 'categorical':
            log_probs = dist.log_prob(actions_seq.squeeze(-1))          # (E, T)
        else:
            log_probs = dist.log_prob(actions_seq)                     # (E, T)
        entropies = dist.entropy()

        policy_loss = -(((log_probs * policy_advantages
                          + entropy_coefficient * entropies) * valid_mask).sum() / valid_count)

        self.model.zero_grad()
        (value_loss + policy_loss).backward()
        if self.value_grad_clip:
            self.model.clip(self.value_grad_clip, modules=self.model.branch_module_names('value'))
        if self.policy_grad_clip:
            self.model.clip(self.policy_grad_clip, modules=self.model.branch_module_names('policy'))
        shared_modules = self.model.shared_module_names()
        if shared_modules:
            shared_clip = self.shared_grad_clip
            if shared_clip is None:
                shared_clip = max(self.policy_grad_clip or float('inf'),
                                  self.value_grad_clip or float('inf'))
            self.model.clip(shared_clip, modules=shared_modules)
        self.model.step()

        if self.auto_entropy_tuning:
            self.entropy_optimizer.zero_grad()
            masked_log_probs = (log_probs.detach() * valid_mask).sum() / valid_count
            alpha_loss = -(self.log_alpha * (masked_log_probs + self.target_entropy)).mean()
            alpha_loss.backward()
            self.entropy_optimizer.step()

        if self.policy.distribution == 'categorical':
            temperature = self.policy.temperature
            if self.policy.temperature_schedule:
                temperature *= self.policy.temperature_schedule.get_factor()
            learn_metrics.update({'temperature': temperature})

        learn_metrics.update({
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'temporal_difference': (td_errors_tm.transpose(0, 1) * valid_mask).sum().item() / valid_count.item(),
            'advantages': (advantages_et * valid_mask).sum().item() / valid_count.item(),
            'returns': (returns_et * valid_mask).sum().item() / valid_count.item(),
            'entropy': ((entropies * valid_mask).sum() / valid_count).item(),
            'entropy_coefficient': entropy_coefficient,
            'policy_learning_rate': self.model.learning_rate('branches.policy'),
            'value_learning_rate': self.model.learning_rate('branches.value'),
        })

        self._advance_rollout_window()
        return learn_metrics

    def get_config(self):
        """Return the architecture description ``{"type", "config"}``.

        Extends the base payload with the composite model config, GAE / clip
        settings, normalizers, and entropy-tuning fields.

        Returns:
            payload (dict): Mapping with ``type`` (class name) and ``config``
                (constructor kwargs suitable for ``from_config`` after env
                injection).
        """
        config = super().get_config()
        config["type"] = self.__class__.__name__
        config["config"].update({
            "model": self.model.get_config(),
            "discount": self.discount,
            "state_normalizer": self.state_normalizer.get_config() if self.state_normalizer is not None else None,
            "goal_normalizer": self.goal_normalizer.get_config() if self.goal_normalizer is not None else None,
            "advantage_normalizer": self.advantage_normalizer.get_config() if self.advantage_normalizer is not None else None,
            "reward_normalizer": self.reward_normalizer.get_config() if self.reward_normalizer is not None else None,
            "entropy_coefficient": self.entropy_coefficient,
            "entropy_schedule": self.entropy_schedule.get_config() if self.entropy_schedule is not None else None,
            "auto_entropy_tuning": self.auto_entropy_tuning,
            "entropy_lr": self.entropy_lr,
            "target_entropy_scale": self.target_entropy_scale,
            "gae_coefficient": self.gae_coefficient,
            "policy_grad_clip": self.policy_grad_clip,
            "value_grad_clip": self.value_grad_clip,
            "shared_grad_clip": self.shared_grad_clip,
            "value_coef": self.value_coef,
            "bootstrap_truncations": self.bootstrap_truncations,
        })
        return config

class PPO(Agent):
    """Proximal Policy Optimization (clipped surrogate) on-policy agent.

    Assembles a composite ``ModularModel`` with required stochastic policy and
    value heads. Supports multi-epoch minibatch updates with optional KL
    penalty / [AdaptiveKL][phoenx.adaptive_kl.AdaptiveKL], value clipping,
    intrinsic motivation, and temporal trunks via `_recurrent_update`.

    Each minibatch combines policy and value losses into one backward and a
    coordinated ``model.step()`` (on-policy gradient ownership).
    """

    MODEL_ATTRS = ("model",)
    SCHEDULE_ATTRS = ("entropy_schedule", "policy_clip_schedule", "value_clip_schedule")
    DEFAULT_SHARED_UPDATE = 'combined'

    def __init__(
        self,
        roots: Dict[str, SubNetwork] | None = None,
        trunk: SubNetwork | None = None,
        policy: StochasticContinuousHead | StochasticDiscreteHead | None = None,
        value: ValueHead | None = None,
        optimizer_params: dict | None = None,
        lr_scheduler: ScheduleWrapper | None = None,
        shared_update: str | None = None,
        discount: float = 0.99,
        gae_coefficient: float = 0.95,
        state_normalizer: BaseNormalizer|None = None,
        goal_normalizer: BaseNormalizer|None = None,
        advantage_normalizer: BaseNormalizer|None = None,
        reward_normalizer: RewardNorm|None = None,
        entropy_coefficient: float = 0.01,
        entropy_schedule: ScheduleWrapper|None = None,
        auto_entropy_tuning: bool = True,
        entropy_lr: float = 3e-4,
        target_entropy_scale: float=0.98, # Only used if auto entropy = True and discrete action space
        kl_coefficient: float = 0.0,
        kl_adapter: AdaptiveKL|None = None,
        policy_clip: float = 0.2,
        policy_clip_schedule: ScheduleWrapper|None = None,
        policy_grad_clip: float = 40.0,
        value_clip: float = 0.2,
        value_clip_schedule: ScheduleWrapper|None = None,
        value_grad_clip: float = 40.0,
        shared_grad_clip: float|None = None, # None -> max(policy, value) clip
        value_coef: float = 0.5,
        reward_clip: float = float('inf'),
        intrinsic_motivation: IntrinsicMotivation|None = None,
        bootstrap_truncations: bool=True,
        save_dir: str = 'models',
        device: str | T.device | None = None,
        **kwargs: Any
    ) -> None:
        """Initialize the PPO agent and assemble its composite model.

        Args:
            roots: Optional per-modality encoder SubNetworks (name -> SubNetwork).
            trunk: Optional shared fusion SubNetwork (may hold temporal layers).
            policy: Stochastic continuous or discrete policy head (required).
            value: Value head for GAE baselines and the clipped value loss
                (required).
            optimizer_params: Model-wide default optimizer spec.
            lr_scheduler: Model-wide default LR scheduler.
            shared_update: Gradient-ownership rule for shared modules; defaults
                to ``'combined'`` via ``DEFAULT_SHARED_UPDATE``.
            discount: Discount factor ``gamma`` for TD / GAE.
            gae_coefficient: GAE lambda for advantage estimation.
            state_normalizer: Optional observation normalizer.
            goal_normalizer: Optional goal normalizer for goal-conditioned envs.
            advantage_normalizer: Optional normalizer for advantages before the
                policy update.
            reward_normalizer: Optional reward normalizer.
            entropy_coefficient: Fixed entropy bonus weight when
                ``auto_entropy_tuning`` is ``False``.
            entropy_schedule: Optional multiplier schedule for
                ``entropy_coefficient`` when auto-tuning is off.
            auto_entropy_tuning: When ``True``, learn ``log_alpha`` toward a
                target entropy.
            entropy_lr: Learning rate for the entropy temperature optimizer.
            target_entropy_scale: Scale used when building the target entropy
                (mainly for discrete action spaces).
            kl_coefficient: Weight on the approximate KL penalty term when no
                ``kl_adapter`` is supplied.
            kl_adapter: Optional adaptive controller that supplies beta and may
                early-stop epochs when KL exceeds ``1.5 * target_kl``.
            policy_clip: PPO ratio clip epsilon (before optional schedule).
            policy_clip_schedule: Optional multiplier schedule for
                ``policy_clip``.
            policy_grad_clip: Max grad norm for policy-branch modules; falsy
                skips clipping.
            value_clip: Clip range around old values in the value loss (before
                optional schedule).
            value_clip_schedule: Optional multiplier schedule for
                ``value_clip``.
            value_grad_clip: Max grad norm for value-branch modules; falsy
                skips clipping.
            shared_grad_clip: Max grad norm for shared roots/trunk modules;
                ``None`` uses ``max(policy_grad_clip, value_grad_clip)``.
            value_coef: Multiplier on the clipped value loss.
            reward_clip: Absolute clamp applied to extrinsic rewards when finite
                and ``reward_normalizer`` is ``None``.
            intrinsic_motivation: Optional intrinsic motivation module trained
                inside ``learn`` and added to the reward used for GAE.
            bootstrap_truncations: Whether truncated episodes bootstrap from
                next-state values in GAE / TD targets.
            save_dir: Default save directory passed to the base agent.
            device: Torch device; ``None`` resolves via ``get_device``.
            **kwargs: Extra attributes set on the instance via the base
                ``Agent`` constructor.
        """
        super().__init__(save_dir, device, **kwargs)
        self._check_head('policy', policy, StochasticContinuousHead, StochasticDiscreteHead)
        self._check_head('value', value, ValueHead)
        self.model = self._assemble_model(
            {'policy': policy, 'value': value}, roots=roots, trunk=trunk,
            optimizer_params=optimizer_params, lr_scheduler=lr_scheduler,
            shared_update=shared_update,
        )
        self.discount = discount
        self.state_normalizer = state_normalizer
        self.goal_normalizer = goal_normalizer
        self.advantage_normalizer = advantage_normalizer
        self.reward_normalizer = reward_normalizer
        self.entropy_coefficient = entropy_coefficient
        self.entropy_schedule = entropy_schedule
        self.auto_entropy_tuning = auto_entropy_tuning
        self.entropy_lr = entropy_lr
        self.target_entropy_scale = target_entropy_scale
        if self.auto_entropy_tuning:
            self.target_entropy, self.log_alpha, self.entropy_optimizer = setup_auto_entropy(
                self.policy,
                target_entropy_scale=target_entropy_scale,
                lr=entropy_lr,
                device=self.device,
            )
        self.gae_coefficient = gae_coefficient
        self.kl_coefficient = kl_coefficient
        self.kl_adapter = kl_adapter
        self.policy_clip = policy_clip
        self.policy_clip_schedule = policy_clip_schedule
        self.policy_grad_clip = policy_grad_clip
        self.value_clip = value_clip
        self.value_clip_schedule = value_clip_schedule
        self.value_grad_clip = value_grad_clip
        self.shared_grad_clip = shared_grad_clip
        self.value_coef = value_coef
        self.reward_clip = reward_clip
        self.intrinsic_motivation = intrinsic_motivation
        self.bootstrap_truncations = bootstrap_truncations

    def act(
        self,
        states: np.ndarray | T.Tensor,
        goals: np.ndarray | T.Tensor | None = None,
        context: str = 'train',
        **kwargs: Any
    ) -> Action:
        """Select an action and package log-probs for the PPO rollout.

        Always runs under ``torch.no_grad``. In ``'train'`` samples from the
        policy (continuous paths also return the pre-squash ``z`` via
        ``sample_with_z``); in ``'test'`` uses the mean action. Optional
        ``dones`` in ``kwargs`` reset recurrent / context state.

        Args:
            states: Current observation(s).
            goals: Optional goal vector(s) for goal-conditioned envs.
            context: ``'train'`` to sample, ``'test'`` for deterministic actions.
            **kwargs: May include ``dones`` (previous-step done flags) forwarded
                to `_rollout_forward`.

        Returns:
            `Action` with ``actions``, optional ``raw_actions``, and
            ``log_probs`` for the importance-ratio baseline in ``learn``.
        """
        raw_actions = None
        with T.no_grad():
            outputs = self._rollout_forward(
                states, goals=goals, branches=('policy',), dones=kwargs.get('dones'))
            dist = outputs['policy']
            if context == 'train':
                if self.policy.distribution == 'categorical':
                    actions = dist.sample()
                    log_probs = dist.log_prob(actions)
                else: # Continuous
                    actions, raw_actions = dist.sample_with_z()
                    log_probs = dist.log_prob_from_z(raw_actions)
            elif context == 'test':
                if self.policy.distribution == 'categorical':
                    actions = self.policy.get_mean_actions(dist)
                    log_probs = dist.log_prob(actions)
                else: # Continuous
                    actions, raw_actions = dist.mean_with_z()
                    log_probs = dist.log_prob_from_z(raw_actions)
            else:
                raise ValueError(f"Invalid context: {context}")

        return Action(actions, raw_actions=raw_actions, log_probs=log_probs)

    def learn(self, step:int, sample:dict, learning_epochs:int, mini_batch_size:int, **kwargs: Any)->dict:
        """Apply PPO clipped-surrogate updates over collected rollouts.

        Computes GAE advantages once, then runs ``learning_epochs`` of shuffled
        minibatches. Each minibatch combines clipped policy and value losses
        into one backward plus ``model.step()``. Temporal models use
        `_recurrent_update` (env-subset sequence minibatches). When a
        ``kl_adapter`` is set, epochs may stop early if mean KL exceeds
        ``1.5 * target_kl``.

        Args:
            step: Global training step (schedules, IM extrinsic gate, logging).
            sample: Rollout batch including ``states``, ``actions``,
                ``raw_actions``, rewards, ``intrinsic_rewards``, next states,
                terminations / truncations, ``first_steps``, ``valid_indices``,
                and goal fields.
            learning_epochs: Number of passes over the valid samples.
            mini_batch_size: Minibatch size in transition units (feedforward) or
                env units (recurrent path).
            **kwargs: Unused; accepted for ``Agent.learn`` API compatibility.

        Returns:
            Metrics dict (losses, entropy, KL, clips, coefficients, LRs, and
            optional intrinsic-motivation scalars).
        """
        self._learn_count += 1
        if self._diag_freq is not None:
            should_log_diag = (self._learn_count % self._diag_freq == 0)
        else:
            should_log_diag = False
        
        learn_metrics = {}

        # Unpack trajectory
        states = sample["states"]
        actions = sample["actions"]
        raw_actions = sample["raw_actions"]
        extrinsic_rewards = sample["rewards"]
        im_rollout_rewards = sample["intrinsic_rewards"]
        next_states = sample["next_states"]
        terminations = sample["terminations"]
        truncations = sample["truncations"]
        first_steps = sample["first_steps"]
        valid_indices = sample["valid_indices"]
        ach_goals = sample["state_achieved_goals"]
        next_ach_goals = sample["next_state_achieved_goals"]
        goals = sample["desired_goals"]

        # Get current values of policy/value clip and entropy/kl coefficients
        policy_clip = self.policy_clip
        if self.policy_clip_schedule:
            policy_clip *= self.policy_clip_schedule.get_factor()

        value_clip = self.value_clip
        if self.value_clip_schedule:
            value_clip *= self.value_clip_schedule.get_factor()

        if self.auto_entropy_tuning:
            entropy_coefficient = self.log_alpha.exp()
        else:
            entropy_coefficient = self.entropy_coefficient
            if self.entropy_schedule:
                entropy_coefficient *= self.entropy_schedule.get_factor()

        kl_coefficient = self.kl_coefficient
        if self.kl_adapter:
            kl_coefficient = self.kl_adapter.get_beta()

        # Get trajectory length, num envs, and total samples for reshaping
        traj_len, num_envs = extrinsic_rewards.shape
        total_samples = traj_len * num_envs

        # Flatten trajectory data (feature shapes preserved; dict obs per key)
        states_flat = flatten_leading(states, 2)
        next_states_flat = flatten_leading(next_states, 2)
        actions_flat = actions.reshape(total_samples, -1)
        raw_actions_flat = raw_actions.reshape(total_samples, -1)
        extrinsic_rewards_flat = extrinsic_rewards.reshape(total_samples, -1)
        goals_flat = goals.reshape(total_samples, -1) if goals is not None else None



        # Normalize states and goals
        if self.state_normalizer:
            states_flat = self.state_normalizer.normalize(states_flat)
            next_states_flat = self.state_normalizer.normalize(next_states_flat)
        if self.goal_normalizer:
            # ach_goals = self.goal_normalizer.normalize(ach_goals)
            # next_ach_goals = self.goal_normalizer.normalize(next_ach_goals)
            goals_flat = self.goal_normalizer.normalize(goals_flat)
        if self.reward_normalizer:
            extrinsic_rewards_flat = self.reward_normalizer.normalize(extrinsic_rewards_flat)

        # Clip rewards if finite and not using reward normalizer
        if T.isfinite(T.tensor(self.reward_clip)) and self.reward_normalizer is None:
            extrinsic_rewards_flat = T.clamp(extrinsic_rewards_flat, min=-self.reward_clip, max=self.reward_clip)

        # Train Intrinsic Motivation and get intrinsic rewards (IM consumes a
        # single flat vector view of the — possibly multi-modal — observation)
        if self.intrinsic_motivation:
            im_states = flatten_obs(states_flat)
            im_next_states = flatten_obs(next_states_flat)
            im_loss = self.intrinsic_motivation.train(im_states, im_next_states, actions_flat)
            # Compute intrinsic reward
            im_learn_rewards = self.intrinsic_motivation.compute_learn_reward(
                im_states,
                im_next_states,
                actions_flat
            )
            # Add intrinsic learn rewards to intrinsic rollout rewards
            im_rewards = im_learn_rewards.reshape(traj_len, num_envs) + im_rollout_rewards
            # Add extrinsic reward if past step threshold
            if self.intrinsic_motivation.use_extrinsic_reward(step):
                rewards = extrinsic_rewards_flat.reshape(traj_len, num_envs) + im_rewards
            else:
                rewards = im_rewards
        else:
            rewards = extrinsic_rewards_flat.reshape(traj_len, num_envs)
            im_learn_rewards = T.zeros_like(rewards)
            im_rewards = T.zeros_like(rewards)

        # Recurrent path: hidden-aware sequence updates over full env rollouts
        if self.model.is_temporal:
            learn_metrics.update(self._recurrent_update(
                step=step,
                states_flat=states_flat, next_states_flat=next_states_flat,
                actions_flat=actions_flat, raw_actions_flat=raw_actions_flat,
                goals_flat=goals_flat, rewards=rewards,
                terminations=terminations, truncations=truncations,
                first_steps=first_steps, traj_len=traj_len, num_envs=num_envs,
                learning_epochs=learning_epochs, mini_batch_size=mini_batch_size,
                policy_clip=policy_clip, value_clip=value_clip,
                entropy_coefficient=entropy_coefficient, kl_coefficient=kl_coefficient,
            ))
            if self.intrinsic_motivation:
                learn_metrics.update({
                    "intrinsic_loss": im_loss,
                    "learn_intrinsic_reward": im_learn_rewards.mean().item(),
                    "intrinsic_reward": im_rewards.mean().item(),
                    "reward_weight": self.intrinsic_motivation.reward_weight * self.intrinsic_motivation.reward_scheduler.get_factor() \
                        if self.intrinsic_motivation.reward_scheduler else self.intrinsic_motivation.reward_weight
                })
            return learn_metrics

        # Get current log probs and values (one shared forward for both heads)
        with T.no_grad():
            cur_out, _ = self.model(states_flat, goal=goals_flat, branches=('policy', 'value'))
            cur_dist = cur_out['policy']
            if self.policy.distribution == 'categorical':
                cur_log_probs = cur_dist.log_prob(actions_flat.view(-1)).unsqueeze(-1)
            else:
                cur_log_probs = cur_dist.log_prob_from_z(raw_actions_flat).unsqueeze(-1)

            cur_values = cur_out['value'].reshape(traj_len, num_envs)
            next_out, _ = self.model(next_states_flat, goal=goals_flat, branches=('value',))
            cur_next_values = next_out['value'].reshape(traj_len, num_envs)

        # Calculate advantages and returns
        advantages, returns, td_errors = compute_advantages_and_returns(
            rewards,
            cur_values,
            cur_next_values,
            terminations,
            truncations,
            self.discount,
            self.gae_coefficient,
            self.bootstrap_truncations,
            self.device
        )

        # Filter phantom steps
        valid_idx = valid_indices.squeeze(-1)
        num_valid = valid_idx.numel()
        states_flat = tree_index(states_flat, valid_idx)
        next_states_flat = tree_index(next_states_flat, valid_idx)
        actions_flat = actions_flat[valid_idx]
        raw_actions_flat = raw_actions_flat[valid_idx]
        goals_flat = goals_flat[valid_idx] if goals is not None else None
        cur_log_probs = cur_log_probs[valid_idx]
        cur_values_flat = cur_values.reshape(total_samples, 1)[valid_idx]
        advantages_flat = advantages.reshape(total_samples, 1)[valid_idx]
        returns_flat = returns.reshape(total_samples, 1)[valid_idx]

        # Normalize advantages
        if self.advantage_normalizer:
            if getattr(self.advantage_normalizer, 'add', None):
                self.advantage_normalizer.add(advantages_flat)
            advantages_flat = self.advantage_normalizer.normalize(advantages_flat)

        # Training loop
        for epoch in range(learning_epochs):
            # Create random indices for shuffling
            indices = T.randperm(num_valid, device=self.device)
            num_batches = num_valid // mini_batch_size

            for batch_num in range(num_batches):
                batch_indices = indices[batch_num * mini_batch_size : (batch_num + 1) * mini_batch_size]
                states_batch = tree_index(states_flat, batch_indices)
                goals_batch = goals_flat[batch_indices] if goals is not None else None
                actions_batch = actions_flat[batch_indices]
                raw_actions_batch = raw_actions_flat[batch_indices]
                cur_log_probs_batch = cur_log_probs[batch_indices].detach()
                cur_values_batch = cur_values_flat[batch_indices].detach()
                advantages_batch = advantages_flat[batch_indices].detach()
                returns_batch = returns_flat[batch_indices].detach()

                ## POLICY + VALUE (one shared forward through roots/trunk) ##
                batch_out, _ = self.model(states_batch, goal=goals_batch,
                                          branches=('policy', 'value'))
                new_dist = batch_out['policy']
                if self.policy.distribution == 'categorical':
                    new_log_probs = new_dist.log_prob(actions_batch.view(-1)).unsqueeze(-1)
                else: # Continuous Distributions
                    new_log_probs = new_dist.log_prob_from_z(raw_actions_batch).unsqueeze(-1)

                log_ratio = new_log_probs - cur_log_probs_batch
                prob_ratio = T.exp(log_ratio)

                # Calculate Surrogate Loss
                surr1 = prob_ratio * advantages_batch
                surr2 = T.clamp(prob_ratio, 1 - policy_clip, 1 + policy_clip) * advantages_batch
                surrogate_loss = -T.min(surr1, surr2).mean()

                # Calculate Entropy penalty
                entropies = new_dist.entropy()
                mean_entropy = entropies.mean()
                entropy_penalty = mean_entropy * -entropy_coefficient

                # Calculate the KL penalty
                with T.no_grad():
                    kl = prob_ratio - 1 - log_ratio
                    mean_kl = kl.mean()
                kl_penalty = mean_kl * kl_coefficient

                # Calculate policy loss
                policy_loss = surrogate_loss + entropy_penalty + kl_penalty

                ## VALUE ##
                values = batch_out['value']
                loss = (values - returns_batch).pow(2)
                clipped_values = cur_values_batch + (values - cur_values_batch).clamp(-value_clip, value_clip)
                clipped_value_loss = (clipped_values - returns_batch).pow(2)
                value_loss = self.value_coef * (0.5 * T.max(loss, clipped_value_loss).mean())

                # ONE combined backward; shared modules accumulate gradients
                # from both losses and are stepped exactly once (SB3/RSL-RL
                # on-policy standard).
                self.model.zero_grad()
                (policy_loss + value_loss).backward()
                policy_grad_norm = None
                value_grad_norm = None
                if self.policy_grad_clip:
                    policy_grad_norm = self.model.clip(
                        self.policy_grad_clip, modules=self.model.branch_module_names('policy'))
                if self.value_grad_clip:
                    value_grad_norm = self.model.clip(
                        self.value_grad_clip, modules=self.model.branch_module_names('value'))
                shared_modules = self.model.shared_module_names()
                if shared_modules:
                    shared_clip = self.shared_grad_clip
                    if shared_clip is None:
                        shared_clip = max(self.policy_grad_clip or float('inf'),
                                          self.value_grad_clip or float('inf'))
                    self.model.clip(shared_clip, modules=shared_modules)

                # Log diag data
                if should_log_diag and (epoch == learning_epochs - 1) and (batch_num == num_batches - 1):
                    self.logger.debug(
                        "ac_diag step=%d learn_count=%d %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s",
                        step,
                        self._learn_count,
                        summarize_tensor(states_batch, "states batch"),
                        summarize_tensor(actions_batch, "actions batch"),
                        summarize_tensor(rewards, "rewards"),
                        summarize_tensor(next_states, "next_states"),
                        summarize_tensor(goals_batch, "goals batch"),
                        summarize_tensor(next_ach_goals, "next_ach_goals"),
                        summarize_tensor(terminations, "terminations"),
                        summarize_tensor(truncations, "truncations"),
                        summarize_tensor(cur_values_batch, "values batch"),
                        summarize_tensor(cur_next_values, "next_values"),
                        summarize_tensor(td_errors, "td_errors"),
                        summarize_tensor(advantages_batch, "advantages batch"),
                        summarize_tensor(returns_batch, "returns batch"),
                        summarize_tensor(cur_log_probs_batch, "log_probs batch"),
                        summarize_tensor(new_log_probs, "new_log_probs batch"),
                        summarize_tensor(prob_ratio, "prob_ratio batch"),
                        summarize_tensor(entropies, "entropies batch"),
                        summarize_tensor(kl, "kl batch"),
                        summarize_tensor(surr1, "surr1"),
                        summarize_tensor(surr2, "surr2"),
                        summarize_tensor(surrogate_loss, "surrogate_loss"),
                        summarize_tensor(policy_loss, "policy_loss"),
                        summarize_tensor(value_loss, "value_loss"),
                        f"entropy_coef={float(entropy_coefficient)}",
                    )

                    self.logger.debug(
                        "ac_grads step=%d learn_count=%d value_grad_norm=%.6f policy_grad_norm=%.6f "
                        "value_loss=%.6f policy_loss=%.6f",
                        step,
                        self._learn_count,
                        float(value_grad_norm) if value_grad_norm is not None else -1.0,
                        float(policy_grad_norm) if policy_grad_norm is not None else -1.0,
                        float(value_loss.item()),
                        float(policy_loss.item()),
                    )

                # Update models (every module optimizer steps once)
                self.model.step()

            
            if self.kl_adapter and mean_kl > self.kl_adapter.target_kl * 1.5:
                break  # Stop this learn cycle's epochs early

        # Step schedulers/adapters
        if self.kl_adapter:
            self.kl_adapter.step(mean_kl)

        if self.auto_entropy_tuning:
            self.entropy_optimizer.zero_grad()
            alpha_loss = -(self.log_alpha * (new_log_probs + self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.entropy_optimizer.step()

        policy_learning_rate = self.model.learning_rate('branches.policy')
        value_learning_rate = self.model.learning_rate('branches.value')

        # Get temperature value from policy if categorical
        if self.policy.distribution == 'categorical':
            temperature = self.policy.temperature
            if self.policy.temperature_schedule:
                temperature *= self.policy.temperature_schedule.get_factor()
            learn_metrics.update({'temperature': temperature})

        learn_metrics.update({
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': mean_entropy.item(),
            'kl': mean_kl.item(),
            'prob_ratio': prob_ratio.detach().cpu().flatten().mean().item(),
            'temporal_difference': td_errors.reshape(total_samples, 1)[valid_idx].cpu().flatten().mean().item(),
            'advantages': advantages.reshape(total_samples, 1)[valid_idx].cpu().flatten().mean().item(),
            'returns': returns.reshape(total_samples, 1)[valid_idx].cpu().flatten().mean().item(),
            'policy_clip': policy_clip,
            'value_clip': value_clip,
            'entropy_coefficient': entropy_coefficient,
            'kl_coefficient': kl_coefficient,
            'policy_learning_rate': policy_learning_rate,
            'value_learning_rate': value_learning_rate
        })

        if self.intrinsic_motivation:
            learn_metrics.update({
                "intrinsic_loss": im_loss,
                "learn_intrinsic_reward": im_learn_rewards.mean().item(),
                "intrinsic_reward": im_rewards.mean().item(),
                "reward_weight": self.intrinsic_motivation.reward_weight * self.intrinsic_motivation.reward_scheduler.get_factor() \
                    if self.intrinsic_motivation.reward_scheduler else self.intrinsic_motivation.reward_weight
            })

        return learn_metrics

    def _recurrent_update(
        self,
        *,
        step: int,
        states_flat, next_states_flat, actions_flat, raw_actions_flat, goals_flat,
        rewards, terminations, truncations, first_steps,
        traj_len: int, num_envs: int,
        learning_epochs: int, mini_batch_size: int,
        policy_clip: float, value_clip: float,
        entropy_coefficient, kl_coefficient,
    ) -> dict:
        """Hidden-aware PPO update over full env sequences (RSL-RL style).

        Minibatches are subsets of envs carrying their full rollout sequences;
        ``mini_batch_size`` is interpreted in env units (values exceeding or not
        dividing ``num_envs`` fall back to all envs). Hidden states start from
        the rollout-start snapshot and reset mid-sequence at episode boundaries
        (``first_steps``). Next-state values for GAE are shifted values from the
        same hidden stream; the final step uses the post-sequence hidden.
        Each minibatch combines policy and value losses into one backward and
        ``model.step()``.

        Args:
            step (int): Global training step (diagnostics / logging).
            states_flat: Flattened observations ``(T * E, ...)``.
            next_states_flat: Flattened next observations.
            actions_flat: Flattened actions.
            raw_actions_flat: Flattened pre-squash actions for continuous
                log-probs.
            goals_flat: Flattened goals, or ``None``.
            rewards: Rewards shaped ``(T, E)`` (extrinsic and/or intrinsic).
            terminations: Termination flags ``(T, E)``.
            truncations: Truncation flags ``(T, E)``.
            first_steps: Episode-start / phantom-step mask ``(T, E)``.
            traj_len (int): Rollout length ``T``.
            num_envs (int): Parallel environment count ``E``.
            learning_epochs (int): Number of passes over env subsets.
            mini_batch_size (int): Envs per minibatch (see above).
            policy_clip (float): Effective PPO ratio clip epsilon.
            value_clip (float): Effective value-function clip range.
            entropy_coefficient: Scalar entropy weight (tensor or float).
            kl_coefficient: Scalar KL penalty weight (tensor or float).

        Returns:
            Metrics dict matching the feedforward ``learn`` path, after
            advancing the rollout-start hidden via `_advance_rollout_window`.
        """
        learn_metrics: dict = {}

        # ---- Sequence views: (T, E, *feat) -> (E, T, *feat) ---------------
        def to_seq(flat):
            seq = unflatten_leading(flat, (traj_len, num_envs))
            from phoenx.obs_utils import tree_map as _tree_map
            return _tree_map(lambda x: x.transpose(0, 1).contiguous(), seq)

        states_seq = to_seq(states_flat)
        next_states_seq = to_seq(next_states_flat)
        actions_seq = actions_flat.reshape(traj_len, num_envs, -1).transpose(0, 1).contiguous()
        raw_actions_seq = raw_actions_flat.reshape(traj_len, num_envs, -1).transpose(0, 1).contiguous()
        goals_seq = (goals_flat.reshape(traj_len, num_envs, -1).transpose(0, 1).contiguous()
                     if goals_flat is not None else None)
        start_mask = first_steps.transpose(0, 1).bool().to(self.device)  # (E, T)
        valid_mask = (~start_mask).float()                               # phantom rows excluded
        valid_count = valid_mask.sum().clamp(min=1.0)

        # ---- Initial hidden for this rollout window ------------------------
        hidden0 = self._rollout_start_hidden
        if hidden0:
            hidden_batch = next(iter(hidden0.values()))
            hidden_batch = hidden_batch[0] if isinstance(hidden_batch, tuple) else hidden_batch
            if hidden_batch.shape[1] != num_envs:
                hidden0 = None
        if not hidden0:
            hidden0 = self.model.init_hidden(num_envs)

        # ---- Old log probs / values (sequence forward, no grad) ------------
        with T.no_grad():
            cur_out, final_hidden = self.model(
                states_seq, goal=goals_seq, branches=('policy', 'value'),
                hidden=hidden0, start_mask=start_mask, mode='sequence')
            cur_dist = cur_out['policy']
            if self.policy.distribution == 'categorical':
                cur_log_probs = cur_dist.log_prob(actions_seq.squeeze(-1))      # (E, T)
            else:
                cur_log_probs = cur_dist.log_prob_from_z(raw_actions_seq)       # (E, T)
            cur_values_et = cur_out['value'].squeeze(-1)                        # (E, T)

            # Next values: shifted within the same hidden stream (exact inside
            # episodes); the final step uses the post-sequence hidden.
            last_next = tree_index(next_states_seq, (slice(None), -1))
            goals_last = goals_seq[:, -1] if goals_seq is not None else None
            last_out, _ = self.model(last_next, goal=goals_last, branches=('value',),
                                     hidden=final_hidden, mode='step')
            next_values_et = T.cat(
                [cur_values_et[:, 1:], last_out['value'].reshape(num_envs, 1)], dim=1)

        cur_values_tm = cur_values_et.transpose(0, 1)      # (T, E)
        next_values_tm = next_values_et.transpose(0, 1)    # (T, E)

        advantages_tm, returns_tm, td_errors_tm = compute_advantages_and_returns(
            rewards, cur_values_tm, next_values_tm, terminations, truncations,
            self.discount, self.gae_coefficient, self.bootstrap_truncations, self.device)
        advantages_et = advantages_tm.transpose(0, 1)      # (E, T)
        returns_et = returns_tm.transpose(0, 1)

        # Normalize advantages (running stats fed with valid entries only)
        if self.advantage_normalizer:
            valid_flat = advantages_et[valid_mask.bool()].reshape(-1, 1)
            if getattr(self.advantage_normalizer, 'add', None):
                self.advantage_normalizer.add(valid_flat)
            advantages_et = self.advantage_normalizer.normalize(
                advantages_et.reshape(-1, 1)).reshape(num_envs, traj_len)

        # ---- Env-subset minibatches over full sequences ---------------------
        envs_per_batch = mini_batch_size if 0 < mini_batch_size <= num_envs else num_envs
        if num_envs % envs_per_batch != 0:
            envs_per_batch = num_envs
        num_batches = num_envs // envs_per_batch

        for epoch in range(learning_epochs):
            perm = T.randperm(num_envs, device=self.device)
            for batch_num in range(num_batches):
                idx = perm[batch_num * envs_per_batch:(batch_num + 1) * envs_per_batch]
                mask_b = valid_mask[idx]
                mask_sum = mask_b.sum().clamp(min=1.0)

                batch_out, _ = self.model(
                    tree_index(states_seq, idx),
                    goal=goals_seq[idx] if goals_seq is not None else None,
                    branches=('policy', 'value'),
                    hidden=self.model.index_hidden(hidden0, idx),
                    start_mask=start_mask[idx], mode='sequence')
                new_dist = batch_out['policy']
                if self.policy.distribution == 'categorical':
                    new_log_probs = new_dist.log_prob(actions_seq[idx].squeeze(-1))
                else:
                    new_log_probs = new_dist.log_prob_from_z(raw_actions_seq[idx])

                log_ratio = new_log_probs - cur_log_probs[idx]
                prob_ratio = T.exp(log_ratio)

                adv_b = advantages_et[idx]
                surr1 = prob_ratio * adv_b
                surr2 = T.clamp(prob_ratio, 1 - policy_clip, 1 + policy_clip) * adv_b
                surrogate_loss = -(T.min(surr1, surr2) * mask_b).sum() / mask_sum

                entropies = new_dist.entropy()
                mean_entropy = (entropies * mask_b).sum() / mask_sum
                entropy_penalty = mean_entropy * -entropy_coefficient

                with T.no_grad():
                    kl = prob_ratio - 1 - log_ratio
                    mean_kl = (kl * mask_b).sum() / mask_sum
                kl_penalty = mean_kl * kl_coefficient

                policy_loss = surrogate_loss + entropy_penalty + kl_penalty

                values = batch_out['value'].squeeze(-1)
                cur_v_b = cur_values_et[idx]
                ret_b = returns_et[idx]
                loss = (values - ret_b).pow(2)
                clipped_values = cur_v_b + (values - cur_v_b).clamp(-value_clip, value_clip)
                clipped_value_loss = (clipped_values - ret_b).pow(2)
                value_loss = self.value_coef * (
                    0.5 * (T.max(loss, clipped_value_loss) * mask_b).sum() / mask_sum)

                self.model.zero_grad()
                (policy_loss + value_loss).backward()
                if self.policy_grad_clip:
                    self.model.clip(self.policy_grad_clip,
                                    modules=self.model.branch_module_names('policy'))
                if self.value_grad_clip:
                    self.model.clip(self.value_grad_clip,
                                    modules=self.model.branch_module_names('value'))
                shared_modules = self.model.shared_module_names()
                if shared_modules:
                    shared_clip = self.shared_grad_clip
                    if shared_clip is None:
                        shared_clip = max(self.policy_grad_clip or float('inf'),
                                          self.value_grad_clip or float('inf'))
                    self.model.clip(shared_clip, modules=shared_modules)
                self.model.step()

            if self.kl_adapter and mean_kl > self.kl_adapter.target_kl * 1.5:
                break

        if self.kl_adapter:
            self.kl_adapter.step(mean_kl)

        if self.auto_entropy_tuning:
            self.entropy_optimizer.zero_grad()
            masked_log_probs = (new_log_probs * mask_b).sum() / mask_sum
            alpha_loss = -(self.log_alpha * (masked_log_probs + self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.entropy_optimizer.step()

        if self.policy.distribution == 'categorical':
            temperature = self.policy.temperature
            if self.policy.temperature_schedule:
                temperature *= self.policy.temperature_schedule.get_factor()
            learn_metrics.update({'temperature': temperature})

        learn_metrics.update({
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': mean_entropy.item(),
            'kl': mean_kl.item(),
            'prob_ratio': prob_ratio.detach().cpu().flatten().mean().item(),
            'temporal_difference': (td_errors_tm.transpose(0, 1) * valid_mask).sum().item() / valid_count.item(),
            'advantages': (advantages_et * valid_mask).sum().item() / valid_count.item(),
            'returns': (returns_et * valid_mask).sum().item() / valid_count.item(),
            'policy_clip': policy_clip,
            'value_clip': value_clip,
            'entropy_coefficient': entropy_coefficient,
            'kl_coefficient': kl_coefficient,
            'policy_learning_rate': self.model.learning_rate('branches.policy'),
            'value_learning_rate': self.model.learning_rate('branches.value'),
        })

        # The rollout buffer resets after learn: the live hidden becomes the
        # next window's initial hidden.
        self._advance_rollout_window()
        return learn_metrics

    def get_config(self):
        """Return the architecture description ``{"type", "config"}``.

        Extends the base payload with the composite model config, PPO clip / KL
        settings, normalizers, entropy tuning, and optional intrinsic motivation.

        Returns:
            payload (dict): Mapping with ``type`` (class name) and ``config``
                (constructor kwargs suitable for ``from_config`` after env
                injection).
        """
        config = super().get_config()
        config["config"].update({
            "model": self.model.get_config(),
            "discount": self.discount,
            "state_normalizer": self.state_normalizer.get_config() if self.state_normalizer is not None else None,
            "goal_normalizer": self.goal_normalizer.get_config() if self.goal_normalizer is not None else None,
            "advantage_normalizer": self.advantage_normalizer.get_config() if self.advantage_normalizer is not None else None,
            "reward_normalizer": self.reward_normalizer.get_config() if self.reward_normalizer is not None else None,
            "entropy_coefficient": self.entropy_coefficient,
            "entropy_schedule": self.entropy_schedule.get_config() if self.entropy_schedule is not None else None,
            "auto_entropy_tuning": self.auto_entropy_tuning,
            "entropy_lr": self.entropy_lr,
            "target_entropy_scale": self.target_entropy_scale,
            "gae_coefficient": self.gae_coefficient,
            "policy_clip": self.policy_clip,
            "policy_clip_schedule": self.policy_clip_schedule.get_config() if self.policy_clip_schedule else None,
            "policy_grad_clip": self.policy_grad_clip,
            "value_clip": self.value_clip,
            "value_clip_schedule": self.value_clip_schedule.get_config() if self.value_clip_schedule else None,
            "value_grad_clip": self.value_grad_clip,
            "shared_grad_clip": self.shared_grad_clip,
            "value_coef": self.value_coef,
            "reward_clip": self.reward_clip,
            "kl_coefficient": self.kl_coefficient,
            "kl_adapter": self.kl_adapter.get_config() if self.kl_adapter else None,
            "intrinsic_motivation": self.intrinsic_motivation.get_config() if self.intrinsic_motivation else None,
            "bootstrap_truncations": self.bootstrap_truncations
        })
        return config

class DDPG(Agent):
    """Deep Deterministic Policy Gradient (DDPG) off-policy agent.

    Assembles a composite ``ModularModel`` with a deterministic continuous
    policy head and a continuous Q critic (roots → trunk → branches). A
    Polyak-averaged ``target_model`` clones the policy and critic branches
    (plus shared roots/trunk). Exploration uses optional
    [Noise][phoenx.noise.Noise] plus an epsilon-greedy random-action chance.

    The critic loss owns shared roots/trunk; the policy trains on detached
    shared features (off-policy gradient ownership). Targets use n-step returns
    via [compute_n_step_return][phoenx.agent_utils.compute_n_step_return].
    """

    MODEL_ATTRS = ("model",)
    TARGET_ATTRS = ("target_model",)
    SCHEDULE_ATTRS = ("noise_schedule",)
    DEFAULT_SHARED_UPDATE = 'critic'

    def __init__(
        self,
        roots: Dict[str, SubNetwork] | None = None,
        trunk: SubNetwork | None = None,
        policy: DeterministicActorHead | None = None,
        critic: ContinuousQHead | None = None,
        *,
        optimizer_params: dict | None = None,
        lr_scheduler: ScheduleWrapper | None = None,
        shared_update: str | None = None,
        discount: float=0.99,
        tau: float=0.001,
        action_epsilon: float = 0.2,
        state_normalizer: BaseNormalizer | None = None,
        goal_normalizer: BaseNormalizer | None = None,
        reward_normalizer: RewardNorm | None = None,
        noise: Noise | None = None,
        noise_schedule: ScheduleWrapper | None = None,
        noise_clip: float = 0.5,
        raw_action_l2_coef: float = 0.0,
        policy_grad_clip: float = float('inf'),
        critic_grad_clip: float = float('inf'),
        critic_huber_delta: float = 1.0,
        N: int=1, # N-steps
        recurrent_burn_in: int = 0, # R2D2 burn-in steps (temporal models; < N)
        intrinsic_motivation: IntrinsicMotivation | None = None,
        save_dir: str = "models",
        device: str | T.device | None = None,
        **kwargs: Any,
    ):
        """Initialize the DDPG agent and assemble its composite model.

        Args:
            roots: Optional per-modality encoder SubNetworks (name -> SubNetwork).
            trunk: Optional shared fusion SubNetwork (may hold temporal layers).
            policy: Deterministic continuous actor head (required).
            critic: Continuous Q-head (required).
            optimizer_params: Model-wide default optimizer spec.
            lr_scheduler: Model-wide default LR scheduler.
            shared_update: Gradient-ownership rule for shared modules; defaults
                to ``'critic'`` via ``DEFAULT_SHARED_UPDATE``.
            discount: Discount factor ``gamma`` for n-step returns.
            tau: Polyak interpolation factor for
                [soft_update][phoenx.agent_utils.soft_update] of the target
                model.
            action_epsilon: Probability of sampling a uniform random action in
                ``'train'`` context (after warmup).
            state_normalizer: Optional observation normalizer.
            goal_normalizer: Optional goal normalizer for goal-conditioned envs.
            reward_normalizer: Optional reward normalizer.
            noise: Optional exploration noise added to the policy action in
                ``'train'`` (for example
                [NormalNoise][phoenx.noise.NormalNoise] or
                [UniformNoise][phoenx.noise.UniformNoise]).
            noise_schedule: Optional multiplier schedule applied to exploration
                noise.
            noise_clip: Absolute clamp on exploration noise when ``> 0``.
            raw_action_l2_coef: L2 penalty weight on pre-squash / raw policy
                outputs in the actor loss.
            policy_grad_clip: Max grad norm for the policy branch.
            critic_grad_clip: Max grad norm for critic and shared modules on
                the critic update.
            critic_huber_delta: Huber loss delta for the critic TD objective.
            N: N-step return horizon expected from the replay sample.
            recurrent_burn_in: R2D2 burn-in length for temporal models; must be
                ``< N`` when set. Anchors the critic / actor TD position after
                burn-in.
            intrinsic_motivation: Optional intrinsic motivation module trained
                inside ``learn`` and mixed into the reward used for targets.
            save_dir: Default save directory passed to the base agent.
            device: Torch device; ``None`` resolves via ``get_device``.
            **kwargs (Any): Extra attributes set on the instance via the base
                ``Agent`` constructor.
        """
        super().__init__(save_dir, device, **kwargs)
        self._check_head('policy', policy, DeterministicActorHead)
        self._check_head('critic', critic, ContinuousQHead)
        self.model = self._assemble_model(
            {'policy': policy, 'critic': critic}, roots=roots, trunk=trunk,
            optimizer_params=optimizer_params, lr_scheduler=lr_scheduler,
            shared_update=shared_update,
        )
        self.discount = discount
        self.tau = tau
        self.state_normalizer = state_normalizer
        self.goal_normalizer = goal_normalizer
        self.reward_normalizer = reward_normalizer
        self.policy_grad_clip = policy_grad_clip
        self.critic_grad_clip = critic_grad_clip
        self.critic_huber_delta = critic_huber_delta
        self.critic_loss_fn = T.nn.HuberLoss(reduction='none', delta=critic_huber_delta)
        self.N = N
        self.recurrent_burn_in = recurrent_burn_in
        if recurrent_burn_in and recurrent_burn_in >= N:
            raise ValueError(f"recurrent_burn_in ({recurrent_burn_in}) must be < N ({N})")
        self.intrinsic_motivation = intrinsic_motivation
        # Target: policy + critic branches (+ shared roots/trunk), name-matched
        self.target_model = self.model.clone(branches=['policy', 'critic'])
        self.action_epsilon = action_epsilon
        self.noise = noise
        self.noise_schedule = noise_schedule
        self.noise_clip = noise_clip
        self.raw_action_l2_coef = raw_action_l2_coef

    def act(
        self,
        states: np.ndarray | T.Tensor,
        goals: np.ndarray | T.Tensor | None = None,
        context: str = 'train',
        step: int | None = None,
        warmup: int | None = None,
        **kwargs: Any,
    ) -> Action:
        """Select a deterministic action, with train-time exploration noise.

        Temporal models always run a policy forward (to advance recurrent /
        context state) and may attach the pre-step hidden for R2D2 storage.
        In ``'train'``, warmup and ``action_epsilon`` may replace the policy
        with a uniform env sample; otherwise clipped / scheduled exploration
        noise is added to the policy action. Any non-``'train'`` context uses
        the deterministic policy without noise (no invalid-context raise).

        Args:
            states: Current observation(s).
            goals: Optional goal vector(s) for goal-conditioned envs.
            context: ``'train'`` for exploration, anything else for the
                deterministic policy (typically ``'test'``).
            step: Global step used with ``warmup`` for random-action warmup.
            warmup: When ``step <= warmup``, sample from the env action space.
            **kwargs (Any): May include ``dones`` (previous-step done flags)
                forwarded to `_rollout_forward` for temporal reset.

        Returns:
            action (`Action`): Package with ``actions``, ``raw_actions``, and
                optional ``hidden`` for recurrent buffers.
        """
        # Temporal models always run the forward (to advance the recurrent /
        # context stream) and attach the pre-step hidden for R2D2 storage.
        pol_outputs = None
        pre_hidden_flat = None
        if self.model.is_temporal:
            with T.no_grad():
                pol_outputs = self._rollout_forward(
                    states, goals=goals, branches=('policy',), dones=kwargs.get('dones'))
            if self.model.is_recurrent and self._last_pre_hidden is not None:
                pre_hidden_flat = self.model.hidden_to_tensors(self._last_pre_hidden)

        # If training
        if context == 'train':
            # If warmup, sample random action from action space
            if (step is not None) and (step <= warmup):
                actions = T.as_tensor(self.model.env.action_space.sample(), device=self.device)
                raw_actions = actions
            # if random number is less than epsilon, sample random action
            elif np.random.random() < self.action_epsilon:
                actions = T.as_tensor(self.model.env.action_space.sample(), device=self.device)
                raw_actions = actions
            # otherwise, sample action from policy
            else:
                noise = self.noise(self.model.env.action_space.shape)
                # Apply noise clipping if needed
                if self.noise_clip > 0:
                    noise = noise.clamp(-self.noise_clip, self.noise_clip)
                # Apply noise schedule if needed
                if self.noise_schedule:
                    noise *= self.noise_schedule.get_factor()
                
                with T.no_grad():
                    if pol_outputs is None:
                        pol_outputs, _ = self.model(states, goal=goals, branches=('policy',))
                    raw_actions, actions = pol_outputs['policy']
                
                # Convert the action space bounds to a tensor on the same device
                actions = (actions + noise).clip(self.policy.act_space_low, self.policy.act_space_high)

        else: # context == 'test'
            with T.no_grad():
                if pol_outputs is None:
                    pol_outputs, _ = self.model(states, goal=goals, branches=('policy',))
                raw_actions, actions = pol_outputs['policy']
        
        return Action(actions, raw_actions=raw_actions, hidden=pre_hidden_flat)

    def soft_update_targets(self):
        """Soft-update the target model from the online model via Polyak averaging.

        Implements [HasTargetNetworks][phoenx.rl_agents.HasTargetNetworks] using
        [soft_update][phoenx.agent_utils.soft_update] with ``self.tau``. The
        target holds policy and critic branches (plus shared modules).
        """
        soft_update(self.model, self.target_model, self.tau)

    def learn(self, step: int, sample: dict, **kwargs: Any)->dict:
        """Apply one DDPG critic-then-actor update from a replay sample.

        Builds n-step TD targets from the target policy and target critic,
        updates the critic (owning shared roots/trunk), then updates the
        policy on detached shared features with actor loss ``-Q(s, pi(s))``
        plus optional raw-action L2. Does not call ``soft_update_targets``;
        the trainer invokes that separately.

        Args:
            step: Global training step (IM extrinsic gate, diagnostics).
            sample: Replay batch with ``states``, ``actions``, ``raw_actions``,
                rewards, ``intrinsic_rewards``, next states, terminations /
                truncations, ``trajectory_lengths``, goals, and optional PER
                ``weights`` / ``indices``.
            **kwargs (Any): Unused; accepted for ``Agent.learn`` API
                compatibility.

        Returns:
            metrics (dict): Loss scalars, TD errors, prediction means, learning
                rates, and optional intrinsic-motivation / noise-anneal fields.
        """
        self._learn_count += 1
        if self._diag_freq is not None:
            should_log_diag = (self._learn_count % self._diag_freq == 0)
        else:
            should_log_diag = False

        learn_metrics = {}

        # Unpack trajectory
        states = sample["states"]
        actions = sample["actions"]
        raw_actions = sample["raw_actions"]
        extrinsic_rewards = sample["rewards"]
        im_rollout_rewards = sample["intrinsic_rewards"]
        next_states = sample["next_states"]
        terminations = sample["terminations"]
        truncations = sample["truncations"]
        trajectory_lengths = sample["trajectory_lengths"]
        # ach_goals = sample["state_achieved_goals"]
        # next_ach_goals = sample["next_state_achieved_goals"]
        goals = sample["desired_goals"]

        if 'weights' in sample:
            weights = sample['weights']
            probs = sample['probs']
            indices = sample['indices']
        else:
            weights = None
            probs = None
            indices = None

        # Get batch_size and n-step trajectory length
        batch_size, n_step_length = extrinsic_rewards.shape

        # Reshape arrays to (batch_size * N, *feat) to train on all steps in N
        # (feature shapes preserved; dict obs handled per key)
        states_flat = flatten_leading(states, 2)
        next_states_flat = flatten_leading(next_states, 2)
        actions_flat = actions.reshape(-1, actions.shape[-1])
        extrinsic_rewards_flat = extrinsic_rewards.reshape(-1, extrinsic_rewards.shape[-1])
        if goals is not None:
            goals_flat = goals.reshape(-1, goals.shape[-1])

        # Normalize states/goals/rewards
        if self.state_normalizer:
            states_flat = self.state_normalizer.normalize(states_flat)
            next_states_flat = self.state_normalizer.normalize(next_states_flat)
        if self.goal_normalizer:
            goals_flat = self.goal_normalizer.normalize(goals_flat)
        if self.reward_normalizer:
            extrinsic_rewards_flat = self.reward_normalizer.normalize(extrinsic_rewards_flat)

        # Create normalized tensors reshaped to [batch_size, n_step_length, *feat]
        states_norm = unflatten_leading(states_flat, (batch_size, n_step_length))
        next_states_norm = unflatten_leading(next_states_flat, (batch_size, n_step_length))
        extrinsic_rewards_norm = extrinsic_rewards_flat.reshape(batch_size, n_step_length)
        if goals is not None:
            goals_norm = goals_flat.reshape(batch_size, n_step_length, -1)
        else:
            goals_norm = None

        # First / last n-step positions (dict-aware indexing)
        states0 = tree_index(states_norm, (slice(None), 0))
        next_states_last = tree_index(next_states_norm, (slice(None), -1))

        # Train Intrinsic Motivation and get intrinsic rewards (flat vector view)
        if self.intrinsic_motivation:
            im_states = flatten_obs(states_flat)
            im_next_states = flatten_obs(next_states_flat)
            im_loss = self.intrinsic_motivation.train(im_states, im_next_states, actions_flat)
            # Compute intrinsic reward
            im_learn_rewards = self.intrinsic_motivation.compute_learn_reward(
                im_states,
                im_next_states,
                actions_flat
            )
            # Add intrinsic learn rewards to intrinsic rollout rewards
            im_rewards = im_learn_rewards.reshape(batch_size, n_step_length) + im_rollout_rewards
            # Add extrinsic reward if past step threshold
            if self.intrinsic_motivation.use_extrinsic_reward(step):
                rewards = extrinsic_rewards_flat.reshape(batch_size, n_step_length) + im_rewards
            else:
                rewards = im_rewards
        else:
            rewards = extrinsic_rewards_flat.reshape(batch_size, n_step_length)
            im_learn_rewards = T.zeros_like(rewards)
            im_rewards = T.zeros_like(rewards)

        goals0 = goals_norm[:,0,:] if goals is not None else None
        goals_last = goals_norm[:,-1,:] if goals is not None else None

        # --- Recurrent (R2D2) setup: stored initial hidden + optional burn-in
        temporal = self.model.is_temporal
        hidden0 = None
        h1 = None
        anchor = 0
        if temporal:
            if self.model.is_recurrent:
                stored = sample.get('initial_hidden')
                hidden0 = (self.model.hidden_from_tensors(
                              {k: v.to(self.device) for k, v in stored.items()})
                           if stored else self.model.init_hidden(batch_size))
                with T.no_grad():
                    # hidden after the first step: initial state of the
                    # next-state stream (exact within the window)
                    _, h1 = self.model(
                        tree_index(states_norm, (slice(None), slice(0, 1))),
                        goal=goals_norm[:, :1] if goals is not None else None,
                        branches=('policy',), hidden=hidden0, mode='sequence')
                    h1 = self.model.detach_hidden(h1)
            anchor = min(self.recurrent_burn_in, n_step_length - 1)
            anchor_goals = goals_norm[:, anchor, :] if goals is not None else None

        # Get target values (goals threaded through every target forward)
        with T.no_grad():
            if temporal:
                # Sequence-mode target stream; bootstrap from the LAST position
                tgt_pol_out, _ = self.target_model(
                    next_states_norm, goal=goals_norm, branches=('policy',),
                    hidden=h1, mode='sequence')
                _, target_actions_seq = tgt_pol_out['policy']
                target_actions = target_actions_seq[:, -1]
                target_actions_full = target_actions.unsqueeze(1).expand_as(target_actions_seq).contiguous()
                tgt_q_out, _ = self.target_model(
                    next_states_norm, action=target_actions_full, goal=goals_norm,
                    branches=('critic',), hidden=h1, mode='sequence')
                target_critic_values = tgt_q_out['critic'][:, -1].squeeze(-1)
                rewards_eff = rewards[:, anchor:]
                terminations_eff = terminations[:, anchor:]
                lengths_eff = (trajectory_lengths - anchor).clamp(min=0)
            else:
                target_pol_out, _ = self.target_model(
                    next_states_last, goal=goals_last, branches=('policy',))
                _, target_actions = target_pol_out['policy']

                target_q_out, _ = self.target_model(
                    next_states_last, action=target_actions, goal=goals_last,
                    branches=('critic',))
                target_critic_values = target_q_out['critic'].squeeze()
                rewards_eff = rewards
                terminations_eff = terminations
                lengths_eff = trajectory_lengths

            targets = compute_n_step_return(
                rewards_eff,
                self.discount,
                device=self.target_model.device
            ).squeeze()

            no_dones_mask = (terminations_eff.sum(dim=1) == 0 ).float() # eliminates bootstrapping terminated episodes
            gamma_pow = self.discount ** lengths_eff # correctly discounts bootstrapped values by traj lengths
            targets += no_dones_mask * gamma_pow * target_critic_values

            targets = T.clamp(targets, min=-1/(1-self.discount))

        # Get current critic predictions (gradients flow through roots+trunk:
        # the critic loss owns the shared body — SAC-AE/DrQ-v2 rule)
        if temporal:
            pred_out, _ = self.model(
                states_norm, action=actions, goal=goals_norm, branches=('critic',),
                hidden=hidden0, mode='sequence')
            predictions = pred_out['critic'][:, anchor].reshape(batch_size)
        else:
            pred_out, _ = self.model(
                states0, action=actions[:,0,:], goal=goals0, branches=('critic',))
            predictions = pred_out['critic'].squeeze()

        # Calculate TD errors (kept as raw signed differences for PER priorities and logging)
        error = targets - predictions

        # Per-sample Huber loss; apply IS weights before averaging if using PER
        per_sample_loss = self.critic_loss_fn(predictions, targets)
        if weights is not None:
            critic_loss = (weights.to(self.model.device) * per_sample_loss).mean()
        else:
            critic_loss = per_sample_loss.mean()

        # Update critic branch + shared roots/trunk
        critic_modules = self.model.branch_module_names('critic') + self.model.shared_module_names()
        self.model.zero_grad()
        critic_loss.backward()
        
        # Clip value gradient
        critic_grad_norm = self.model.clip(self.critic_grad_clip, modules=critic_modules)
        self.model.step(critic_modules)

        # Get actor's action predictions — shared features DETACHED so the
        # actor loss never updates roots/trunk (exactly one owner per param)
        self.model.zero_grad()
        if temporal:
            pol_out, _ = self.model(
                states_norm, goal=goals_norm, branches=('policy',),
                hidden=hidden0, detach_shared=True, mode='sequence')
            mu_seq, pi_seq = pol_out['policy']
            pred_raw_actions = mu_seq[:, anchor]
            pred_actions = pi_seq[:, anchor]
            pred_actions_full = pred_actions.unsqueeze(1).expand_as(pi_seq).contiguous()
            q_out, _ = self.model(
                states_norm, action=pred_actions_full, goal=goals_norm,
                branches=('critic',), hidden=hidden0, detach_shared=True, mode='sequence')
            critic_values = q_out['critic'][:, anchor]
        else:
            pol_out, _ = self.model(
                states0, goal=goals0, branches=('policy',), detach_shared=True)
            pred_raw_actions, pred_actions = pol_out['policy']
            
            # Calculate actor loss based on critic (also on detached features; the
            # critic branch's gradients are discarded — only the policy steps)
            q_out, _ = self.model(
                states0, action=pred_actions, goal=goals0,
                branches=('critic',), detach_shared=True)
            critic_values = q_out['critic']

        if weights is not None:
            actor_loss = -(weights.to(self.model.device) * critic_values).mean()
        else:
            actor_loss = -critic_values.mean()

        # Add raw action l2 regularization if coef > 0
        actor_loss += self.raw_action_l2_coef * pred_raw_actions.pow(2).mean()

        # Update actor branch only
        actor_loss.backward()

        # Clip policy gradient
        policy_grad_norm = self.model.clip(
            self.policy_grad_clip, modules=self.model.branch_module_names('policy'))
        self.model.step(self.model.branch_module_names('policy'))

        # Log diag data
        if should_log_diag:
            self.logger.debug(
                "ac_diag step=%d learn_count=%d %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s",
                step,
                self._learn_count,
                summarize_tensor(states, "states"),
                summarize_tensor(actions, "actions"),
                summarize_tensor(extrinsic_rewards, "rewards"),
                summarize_tensor(im_rollout_rewards, "intrinsic rollout rewards"),
                summarize_tensor(im_learn_rewards, "intrinsic learn rewards"),
                summarize_tensor(im_rewards, "intrinsic rewards"),
                summarize_tensor(next_states, "next_states"),
                summarize_tensor(goals, "goals"),
                summarize_tensor(terminations, "terminations"),
                summarize_tensor(truncations, "truncations"),
                summarize_tensor(target_actions, "target actions"),
                summarize_tensor(target_critic_values, "target critic values"),
                summarize_tensor(targets, "targets"),
                summarize_tensor(predictions, "critic predictions"),
                summarize_tensor(error, "critic errors"),
                summarize_tensor(pred_actions, "predicted actions"),
                summarize_tensor(critic_values, "predicted critic values"),
                summarize_tensor(actor_loss, "actor loss"),
                summarize_tensor(critic_loss, "critic loss"),
            )

            self.logger.debug(
                "ac_grads step=%d learn_count=%d critic_grad_norm=%.6f policy_grad_norm=%.6f "
                "critic_loss=%.6f actor_loss=%.6f",
                step,
                self._learn_count,
                float(critic_grad_norm) if critic_grad_norm is not None else -1.0,
                float(policy_grad_norm) if policy_grad_norm is not None else -1.0,
                float(critic_loss.item()),
                float(actor_loss.item()),
            )

        policy_learning_rate = self.model.learning_rate('branches.policy')
        critic_learning_rate = self.model.learning_rate('branches.critic')
        
        learn_metrics.update({
            "extrinsic_rewards": extrinsic_rewards.mean().item(),
            "policy_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "td_errors": error.detach().flatten(),
            "td_error": error.mean().item(),
            "policy_predictions": pred_actions.mean().item(),
            "critic_predictions": critic_values.mean().item(),
            "target_policy_predictions": target_actions.mean().item(),
            "target_critic_predictions": targets.mean().item(),
            'policy_learning_rate': policy_learning_rate,
            'critic_learning_rate': critic_learning_rate
        })

        if self.intrinsic_motivation:
            learn_metrics.update({
                "intrinsic_loss": im_loss,
                "learn_intrinsic_reward": im_learn_rewards.mean().item(),
                "intrinsic_reward": im_rewards.mean().item(),
                "reward_weight": self.intrinsic_motivation.reward_weight * self.intrinsic_motivation.reward_scheduler.get_factor() \
                    if self.intrinsic_motivation.reward_scheduler else self.intrinsic_motivation.reward_weight
            })
        
        if self.noise_schedule:
            learn_metrics.update({'noise_anneal': self.noise_schedule.get_factor()})

        return learn_metrics

    def get_config(self):
        """Return the architecture description ``{"type", "config"}``.

        Extends the base payload with the composite model, DDPG hyperparameters,
        normalizers, exploration noise, and optional intrinsic motivation.

        Returns:
            payload (dict): Mapping with ``type`` (class name) and ``config``
                (constructor kwargs suitable for ``from_config`` after env
                injection).
        """
        config = super().get_config()
        config["type"] = self.__class__.__name__
        config["config"].update({
            "model": self.model.get_config(),
            "discount": self.discount,
            "tau": self.tau,
            "action_epsilon": self.action_epsilon,
            "state_normalizer": self.state_normalizer.get_config() if self.state_normalizer is not None else None,
            "goal_normalizer": self.goal_normalizer.get_config() if self.goal_normalizer is not None else None,
            "reward_normalizer": self.reward_normalizer.get_config() if self.reward_normalizer is not None else None,
            "noise": self.noise.get_config() if self.noise is not None else None,
            "noise_schedule": self.noise_schedule.get_config() if self.noise_schedule is not None else None,
            "noise_clip": self.noise_clip,
            "raw_action_l2_coef": self.raw_action_l2_coef,
            "policy_grad_clip": self.policy_grad_clip,
            "critic_grad_clip": self.critic_grad_clip,
            "critic_huber_delta": self.critic_huber_delta,
            "N": self.N,
            "recurrent_burn_in": self.recurrent_burn_in,
            "intrinsic_motivation": self.intrinsic_motivation.get_config() if self.intrinsic_motivation is not None else None,
        })
        return config


class TD3(Agent):
    """Twin Delayed Deep Deterministic Policy Gradient (TD3) off-policy agent.

    Extends DDPG with clipped double-Q (``critic`` / ``critic_b``), target-
    policy smoothing via ``target_noise``, and delayed policy optimizer steps
    controlled by ``policy_update_delay``. Assembles a composite
    ``ModularModel`` (roots → trunk → branches); the Polyak
    ``target_model`` clones policy and both critics.

    Critic losses own shared roots/trunk; the policy trains on detached shared
    features (off-policy gradient ownership).
    """

    MODEL_ATTRS = ("model",)
    TARGET_ATTRS = ("target_model",)
    SCHEDULE_ATTRS = ("noise_schedule", "target_noise_schedule")
    DEFAULT_SHARED_UPDATE = 'critic'

    def __init__(
        self,
        roots: Dict[str, SubNetwork] | None = None,
        trunk: SubNetwork | None = None,
        policy: DeterministicActorHead | None = None,
        critic: ContinuousQHead | None = None,
        critic_b: ContinuousQHead | None = None,
        *,
        optimizer_params: dict | None = None,
        lr_scheduler: ScheduleWrapper | None = None,
        shared_update: str | None = None,
        discount: float = 0.99,
        tau: float = 0.005,
        action_epsilon: float = 0.0,
        state_normalizer: BaseNormalizer|None = None,
        goal_normalizer: BaseNormalizer|None = None,
        reward_normalizer: RewardNorm|None = None,
        noise: Noise|None = None,
        noise_schedule: ScheduleWrapper|None = None,
        target_noise: Noise|None = None,
        target_noise_schedule: ScheduleWrapper|None = None,
        noise_clip: float = 0.5,
        raw_action_l2_coef: float = 0.0,
        policy_grad_clip: float = float('inf'),
        critic_grad_clip: float = float('inf'),
        critic_huber_delta: float = 1.0,
        policy_update_delay: int = 2,
        N: int=1, # N-steps
        recurrent_burn_in: int = 0, # R2D2 burn-in steps (temporal models; < N)
        intrinsic_motivation: IntrinsicMotivation|None = None,
        save_dir: str = "models",
        device: str|T.device|None = None,
        **kwargs
    ):
        """Initialize the TD3 agent and assemble its composite model.

        When ``critic_b`` is ``None``, a fresh twin critic is cloned from
        ``critic``'s config.

        Args:
            roots: Optional per-modality encoder SubNetworks (name -> SubNetwork).
            trunk: Optional shared fusion SubNetwork (may hold temporal layers).
            policy: Deterministic continuous actor head (required).
            critic: Continuous Q-head A (required).
            critic_b: Continuous Q-head B; cloned from ``critic`` when ``None``.
            optimizer_params: Model-wide default optimizer spec.
            lr_scheduler: Model-wide default LR scheduler.
            shared_update: Gradient-ownership rule for shared modules; defaults
                to ``'critic'`` via ``DEFAULT_SHARED_UPDATE``.
            discount: Discount factor ``gamma`` for n-step returns.
            tau: Polyak interpolation factor for
                [soft_update][phoenx.agent_utils.soft_update] of the target
                model.
            action_epsilon: Probability of sampling a uniform random action in
                ``'train'`` context (after warmup).
            state_normalizer: Optional observation normalizer.
            goal_normalizer: Optional goal normalizer for goal-conditioned envs.
            reward_normalizer: Optional reward normalizer.
            noise: Optional exploration noise added to the policy action in
                ``'train'``.
            noise_schedule: Optional multiplier schedule for exploration noise.
            target_noise: Noise added to target-policy actions for smoothing.
            target_noise_schedule: Optional multiplier schedule for
                ``target_noise``.
            noise_clip: Absolute clamp applied to both exploration and target
                noise when ``> 0``.
            raw_action_l2_coef: L2 penalty weight on raw policy outputs in the
                actor loss.
            policy_grad_clip: Max grad norm for the policy branch.
            critic_grad_clip: Max grad norm for each critic branch and shared
                modules on the critic update.
            critic_huber_delta: Huber loss delta for both critic TD objectives.
            policy_update_delay: Policy optimizer steps only when
                ``_learn_count % policy_update_delay == 0`` (actor loss is still
                computed and backpropagated every learn call).
            N: N-step return horizon expected from the replay sample.
            recurrent_burn_in: R2D2 burn-in length for temporal models; must be
                ``< N`` when set.
            intrinsic_motivation: Optional intrinsic motivation module trained
                inside ``learn`` and mixed into the reward used for targets.
            save_dir: Default save directory passed to the base agent.
            device: Torch device; ``None`` resolves via ``get_device``.
            **kwargs (Any): Extra attributes set on the instance via the base
                ``Agent`` constructor.
        """
        super().__init__(save_dir, device, **kwargs)
        self._check_head('policy', policy, DeterministicActorHead)
        self._check_head('critic', critic, ContinuousQHead)
        self._check_head('critic_b', critic_b, ContinuousQHead, optional=True)
        self.recurrent_burn_in = recurrent_burn_in
        if recurrent_burn_in and recurrent_burn_in >= N:
            raise ValueError(f"recurrent_burn_in ({recurrent_burn_in}) must be < N ({N})")
        # clone second critic head (fresh weights) if critic_b None
        if critic_b is None:
            critic_b = build_head(critic.get_config(), critic.env)
        self.model = self._assemble_model(
            {'policy': policy, 'critic': critic, 'critic_b': critic_b},
            roots=roots, trunk=trunk,
            optimizer_params=optimizer_params, lr_scheduler=lr_scheduler,
            shared_update=shared_update,
        )
        self.discount = discount
        self.tau = tau
        self.state_normalizer = state_normalizer
        self.goal_normalizer = goal_normalizer
        self.reward_normalizer = reward_normalizer
        self.policy_grad_clip = policy_grad_clip
        self.critic_grad_clip = critic_grad_clip
        self.critic_huber_delta = critic_huber_delta
        self.critic_loss_fn = T.nn.HuberLoss(reduction='none', delta=critic_huber_delta)
        self.N = N
        self.intrinsic_motivation = intrinsic_motivation
        # Target: policy + both critics (+ shared roots/trunk)
        self.target_model = self.model.clone(branches=['policy', 'critic', 'critic_b'])
        self.action_epsilon = action_epsilon
        self.noise = noise
        self.noise_schedule = noise_schedule
        self.noise_clip = noise_clip
        self.raw_action_l2_coef = raw_action_l2_coef
        self.target_noise = target_noise
        self.target_noise_schedule = target_noise_schedule
        self.policy_update_delay = policy_update_delay

    def act(
        self,
        states: np.ndarray | T.Tensor,
        goals: np.ndarray | T.Tensor | None = None,
        context: str = 'train',
        step: int | None = None,
        warmup: int | None = None,
        **kwargs: Any,
    ) -> Action:
        """Select a deterministic action, with train-time exploration noise.

        Same exploration pattern as DDPG: temporal forwards advance recurrent /
        context state; warmup and ``action_epsilon`` may use uniform env samples;
        otherwise clipped / scheduled ``noise`` is added. Any non-``'train'``
        context uses the deterministic policy without noise (no invalid-context
        raise).

        Args:
            states: Current observation(s).
            goals: Optional goal vector(s) for goal-conditioned envs.
            context: ``'train'`` for exploration, anything else for the
                deterministic policy (typically ``'test'``).
            step: Global step used with ``warmup`` for random-action warmup.
            warmup: When ``step <= warmup``, sample from the env action space.
            **kwargs (Any): May include ``dones`` (previous-step done flags)
                forwarded to `_rollout_forward` for temporal reset.

        Returns:
            action (`Action`): Package with ``actions``, ``raw_actions``, and
                optional ``hidden`` for recurrent buffers.
        """
        # Temporal models always run the forward (to advance the recurrent /
        # context stream) and attach the pre-step hidden for R2D2 storage.
        pol_outputs = None
        pre_hidden_flat = None
        if self.model.is_temporal:
            with T.no_grad():
                pol_outputs = self._rollout_forward(
                    states, goals=goals, branches=('policy',), dones=kwargs.get('dones'))
            if self.model.is_recurrent and self._last_pre_hidden is not None:
                pre_hidden_flat = self.model.hidden_to_tensors(self._last_pre_hidden)

        # If training
        if context == 'train':
            # If warmup, sample random action from action space
            if (step is not None) and (step <= warmup):
                actions = T.as_tensor(self.model.env.action_space.sample(), device=self.device)
                raw_actions = actions
            # if random number is less than epsilon, sample random action
            elif np.random.random() < self.action_epsilon:
                actions = T.as_tensor(self.model.env.action_space.sample(), device=self.device)
                raw_actions = actions
            # otherwise, sample action from policy
            else:
                noise = self.noise(self.model.env.action_space.shape)
                # Apply noise clipping if needed
                if self.noise_clip > 0:
                    noise = noise.clamp(-self.noise_clip, self.noise_clip)
                # Apply noise schedule if needed
                if self.noise_schedule:
                    noise *= self.noise_schedule.get_factor()
                
                with T.no_grad():
                    if pol_outputs is None:
                        pol_outputs, _ = self.model(states, goal=goals, branches=('policy',))
                    raw_actions, actions = pol_outputs['policy']
                
                # Convert the action space bounds to a tensor on the same device
                actions = (actions + noise).clip(self.policy.act_space_low, self.policy.act_space_high)

        else: # context == 'test'
            with T.no_grad():
                if pol_outputs is None:
                    pol_outputs, _ = self.model(states, goal=goals, branches=('policy',))
                raw_actions, actions = pol_outputs['policy']

        return Action(actions, raw_actions=raw_actions, hidden=pre_hidden_flat)

    def soft_update_targets(self):
        """Soft-update the target model from the online model via Polyak averaging.

        Implements [HasTargetNetworks][phoenx.rl_agents.HasTargetNetworks] using
        [soft_update][phoenx.agent_utils.soft_update] with ``self.tau``. The
        target holds policy and both critic branches (plus shared modules).
        """
        soft_update(self.model, self.target_model, self.tau)
            
    def learn(self, step: int, sample: dict, **kwargs: Any)->dict:
        """Apply one TD3 critic-then-actor update from a replay sample.

        Builds n-step TD targets with target-policy smoothing and
        ``min(Q_a, Q_b)``, updates both critics (owning shared roots/trunk), then
        computes the actor loss on detached shared features. The actor
        ``backward`` runs every call; ``model.step`` for the policy branch runs
        only when ``_learn_count % policy_update_delay == 0``. Does not call
        ``soft_update_targets``.

        Args:
            step: Global training step (IM extrinsic gate, diagnostics).
            sample: Replay batch with ``states``, ``actions``, ``raw_actions``,
                rewards, ``intrinsic_rewards``, next states, terminations /
                truncations, ``trajectory_lengths``, goals, and optional PER
                ``weights`` / ``indices``.
            **kwargs (Any): Unused; accepted for ``Agent.learn`` API
                compatibility.

        Returns:
            metrics (dict): Loss scalars, TD errors, prediction means, learning
                rates, and optional intrinsic-motivation / noise-anneal fields.
        """
        self._learn_count += 1
        if self._diag_freq is not None:
            should_log_diag = (self._learn_count % self._diag_freq == 0)
        else:
            should_log_diag = False

        learn_metrics = {}

        # Unpack trajectory
        states = sample["states"]
        actions = sample["actions"]
        raw_actions = sample["raw_actions"]
        extrinsic_rewards = sample["rewards"]
        im_rollout_rewards = sample["intrinsic_rewards"]
        next_states = sample["next_states"]
        terminations = sample["terminations"]
        truncations = sample["truncations"]
        trajectory_lengths = sample["trajectory_lengths"]
        # ach_goals = sample["state_achieved_goals"]
        # next_ach_goals = sample["next_state_achieved_goals"]
        goals = sample["desired_goals"]

        if 'weights' in sample:
            weights = sample['weights']
            probs = sample['probs']
            indices = sample['indices']
        else:
            weights = None
            probs = None
            indices = None

        # Get batch_size and n-step trajectory length
        batch_size, n_step_length = extrinsic_rewards.shape

        # Reshape arrays to (batch_size * N, *feat) to train on all steps in N
        # (feature shapes preserved; dict obs handled per key)
        states_flat = flatten_leading(states, 2)
        next_states_flat = flatten_leading(next_states, 2)
        actions_flat = actions.reshape(-1, actions.shape[-1])
        extrinsic_rewards_flat = extrinsic_rewards.reshape(-1, extrinsic_rewards.shape[-1])
        if goals is not None:
            goals_flat = goals.reshape(-1, goals.shape[-1])

        # Normalize states/goals/rewards
        if self.state_normalizer:
            states_flat = self.state_normalizer.normalize(states_flat)
            next_states_flat = self.state_normalizer.normalize(next_states_flat)
        if self.goal_normalizer:
            goals_flat = self.goal_normalizer.normalize(goals_flat)
        if self.reward_normalizer:
            extrinsic_rewards_flat = self.reward_normalizer.normalize(extrinsic_rewards_flat)

        # Create normalized tensors reshaped to [batch_size, n_step_length, *feat]
        states_norm = unflatten_leading(states_flat, (batch_size, n_step_length))
        next_states_norm = unflatten_leading(next_states_flat, (batch_size, n_step_length))
        extrinsic_rewards_norm = extrinsic_rewards_flat.reshape(batch_size, n_step_length)
        if goals is not None:
            goals_norm = goals_flat.reshape(batch_size, n_step_length, -1)
        else:
            goals_norm = None

        # First / last n-step positions (dict-aware indexing)
        states0 = tree_index(states_norm, (slice(None), 0))
        next_states_last = tree_index(next_states_norm, (slice(None), -1))

        # Train Intrinsic Motivation and get intrinsic rewards (flat vector view)
        if self.intrinsic_motivation:
            im_states = flatten_obs(states_flat)
            im_next_states = flatten_obs(next_states_flat)
            im_loss = self.intrinsic_motivation.train(im_states, im_next_states, actions_flat)
            # Compute intrinsic reward
            im_learn_rewards = self.intrinsic_motivation.compute_learn_reward(
                im_states,
                im_next_states,
                actions_flat
            )
            # Add intrinsic learn rewards to intrinsic rollout rewards
            im_rewards = im_learn_rewards.reshape(batch_size, n_step_length) + im_rollout_rewards
            # Add extrinsic reward if past step threshold
            if self.intrinsic_motivation.use_extrinsic_reward(step):
                rewards = extrinsic_rewards_flat.reshape(batch_size, n_step_length) + im_rewards
            else:
                rewards = im_rewards
        else:
            rewards = extrinsic_rewards_flat.reshape(batch_size, n_step_length)
            im_learn_rewards = T.zeros_like(rewards)
            im_rewards = T.zeros_like(rewards)

        goals0 = goals_norm[:,0,:] if goals is not None else None
        goals_last = goals_norm[:,-1,:] if goals is not None else None

        # --- Recurrent (R2D2) setup: stored initial hidden + optional burn-in
        temporal = self.model.is_temporal
        hidden0 = None
        h1 = None
        anchor = 0
        if temporal:
            if self.model.is_recurrent:
                stored = sample.get('initial_hidden')
                hidden0 = (self.model.hidden_from_tensors(
                              {k: v.to(self.device) for k, v in stored.items()})
                           if stored else self.model.init_hidden(batch_size))
                with T.no_grad():
                    _, h1 = self.model(
                        tree_index(states_norm, (slice(None), slice(0, 1))),
                        goal=goals_norm[:, :1] if goals is not None else None,
                        branches=('policy',), hidden=hidden0, mode='sequence')
                    h1 = self.model.detach_hidden(h1)
            anchor = min(self.recurrent_burn_in, n_step_length - 1)

        # Get target values (goals threaded through every target forward)
        with T.no_grad():
            if temporal:
                tgt_pol_out, _ = self.target_model(
                    next_states_norm, goal=goals_norm, branches=('policy',),
                    hidden=h1, mode='sequence')
                _, target_actions_seq = tgt_pol_out['policy']
                target_actions = target_actions_seq[:, -1]
            else:
                target_pol_out, _ = self.target_model(
                    next_states_last, goal=goals_last, branches=('policy',))
                _, target_actions = target_pol_out['policy']

            noise = self.target_noise(target_actions.shape)
            # Apply noise clipping if needed
            if self.noise_clip > 0:
                noise = noise.clamp(-self.noise_clip, self.noise_clip)
            # Apply noise schedule if needed
            if self.target_noise_schedule is not None:
                noise *= self.target_noise_schedule.get_factor()
                learn_metrics.update({'target_noise_anneal': self.target_noise_schedule.get_factor()})   
                
            # Add noise to target actions and clamp to env action space
            target_actions = target_actions + noise
            target_actions = target_actions.clamp(self.policy.act_space_low, self.policy.act_space_high)
            
            if temporal:
                target_actions_full = target_actions.unsqueeze(1).expand(
                    batch_size, n_step_length, target_actions.shape[-1]).contiguous()
                tgt_q_out, _ = self.target_model(
                    next_states_norm, action=target_actions_full, goal=goals_norm,
                    branches=('critic', 'critic_b'), hidden=h1, mode='sequence')
                target_critic_values_a = tgt_q_out['critic'][:, -1].reshape(batch_size)
                target_critic_values_b = tgt_q_out['critic_b'][:, -1].reshape(batch_size)
                rewards_eff = rewards[:, anchor:]
                terminations_eff = terminations[:, anchor:]
                lengths_eff = (trajectory_lengths - anchor).clamp(min=0)
            else:
                target_q_out, _ = self.target_model(
                    next_states_last, action=target_actions, goal=goals_last,
                    branches=('critic', 'critic_b'))
                target_critic_values_a = target_q_out['critic'].squeeze()
                target_critic_values_b = target_q_out['critic_b'].squeeze()
                rewards_eff = rewards
                terminations_eff = terminations
                lengths_eff = trajectory_lengths

            targets = compute_n_step_return(
                rewards_eff,
                self.discount,
                device=self.target_model.device
            ).squeeze()

            target_critic_values = T.minimum(target_critic_values_a, target_critic_values_b)
            no_dones_mask = (terminations_eff.sum(dim=1) == 0 ).float() # eliminates bootstrapping terminated episodes
            gamma_pow = self.discount ** lengths_eff # correctly discounts bootstrapped values by traj lengths
            targets += no_dones_mask * gamma_pow * target_critic_values
            
            
            targets = T.clamp(targets, min=-1/(1-self.discount))

        # Get current critic predictions (one shared forward for both critics;
        # the critic loss owns the shared roots/trunk)
        if temporal:
            pred_out, _ = self.model(
                states_norm, action=actions, goal=goals_norm,
                branches=('critic', 'critic_b'), hidden=hidden0, mode='sequence')
            predictions_a = pred_out['critic'][:, anchor].reshape(batch_size)
            predictions_b = pred_out['critic_b'][:, anchor].reshape(batch_size)
        else:
            pred_out, _ = self.model(
                states0, action=actions[:,0,:], goal=goals0,
                branches=('critic', 'critic_b'))
            predictions_a = pred_out['critic'].squeeze()
            predictions_b = pred_out['critic_b'].squeeze()

        # Calculate TD errors (kept as raw signed differences for PER priorities and logging)
        error_a = targets - predictions_a
        error_b = targets - predictions_b
        # error = (error_a.abs() + error_b.abs()) / 2  # Average of absolute errors for priorities
        error = T.minimum(error_a, error_b)

        # Per-sample Huber loss; apply IS weights before averaging if using PER
        per_sample_loss_a = self.critic_loss_fn(predictions_a, targets)
        per_sample_loss_b = self.critic_loss_fn(predictions_b, targets)
        if weights is not None:
            critic_loss_a = (weights.to(self.model.device) * per_sample_loss_a).mean()
            critic_loss_b = (weights.to(self.model.device) * per_sample_loss_b).mean()
            critic_loss = critic_loss_a + critic_loss_b
        else:
            critic_loss_a = per_sample_loss_a.mean()
            critic_loss_b = per_sample_loss_b.mean()
            critic_loss = critic_loss_a + critic_loss_b

        # Update critics + shared roots/trunk (one combined critic backward)
        critic_modules = (self.model.branch_module_names('critic', 'critic_b')
                          + self.model.shared_module_names())
        self.model.zero_grad()
        critic_loss.backward()

        # Clip value gradient (per critic branch)
        critic_a_grad_norm = self.model.clip(
            self.critic_grad_clip, modules=self.model.branch_module_names('critic'))
        critic_b_grad_norm = self.model.clip(
            self.critic_grad_clip, modules=self.model.branch_module_names('critic_b'))
        shared_modules = self.model.shared_module_names()
        if shared_modules:
            self.model.clip(self.critic_grad_clip, modules=shared_modules)
        self.model.step(critic_modules)
        
        # Get actor's action predictions — shared features DETACHED (SAC-AE rule)
        self.model.zero_grad()
        if temporal:
            pol_out, _ = self.model(
                states_norm, goal=goals_norm, branches=('policy',),
                hidden=hidden0, detach_shared=True, mode='sequence')
            mu_seq, pi_seq = pol_out['policy']
            pred_raw_actions = mu_seq[:, anchor]
            pred_actions = pi_seq[:, anchor]
            pred_actions_full = pred_actions.unsqueeze(1).expand_as(pi_seq).contiguous()
            q_out, _ = self.model(
                states_norm, action=pred_actions_full, goal=goals_norm,
                branches=('critic',), hidden=hidden0, detach_shared=True, mode='sequence')
            critic_values = q_out['critic'][:, anchor]
        else:
            pol_out, _ = self.model(
                states0, goal=goals0, branches=('policy',), detach_shared=True)
            pred_raw_actions, pred_actions = pol_out['policy']
            
            # Calculate actor loss based on critic (detached features; critic
            # branch gradients are discarded — only the policy branch steps)
            q_out, _ = self.model(
                states0, action=pred_actions, goal=goals0,
                branches=('critic',), detach_shared=True)
            critic_values = q_out['critic']
        
        # Apply importance sampling weights if using prioritized replay
        if weights is not None:
            actor_loss = -(weights.to(self.model.device) * critic_values).mean()
        else:
            actor_loss = -critic_values.mean()
        
        # Add raw action l2 regularization if coef > 0
        actor_loss += self.raw_action_l2_coef * pred_raw_actions.pow(2).mean()

        
        # Update actor branch only (delayed)
        actor_loss.backward()
        # Clip policy gradient
        policy_grad_norm = self.model.clip(
            self.policy_grad_clip, modules=self.model.branch_module_names('policy'))
        if self._learn_count % self.policy_update_delay == 0:
            self.model.step(self.model.branch_module_names('policy'))

        # Log diag data
        if should_log_diag:
            self.logger.debug(
                "ac_diag step=%d learn_count=%d %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s",
                step,
                self._learn_count,
                summarize_tensor(states, "states"),
                summarize_tensor(actions, "actions"),
                summarize_tensor(extrinsic_rewards, "rewards"),
                summarize_tensor(im_rollout_rewards, "intrinsic rollout rewards"),
                summarize_tensor(im_learn_rewards, "intrinsic learn rewards"),
                summarize_tensor(im_rewards, "intrinsic rewards"),
                summarize_tensor(next_states, "next_states"),
                summarize_tensor(goals, "goals"),
                summarize_tensor(terminations, "terminations"),
                summarize_tensor(truncations, "truncations"),
                summarize_tensor(target_actions, "target actions"),
                summarize_tensor(target_critic_values_a, "target critic values A"),
                summarize_tensor(target_critic_values_b, "target critic values B"),
                summarize_tensor(target_critic_values, "target critic values"),
                summarize_tensor(targets, "targets"),
                summarize_tensor(predictions_a, "critic predictions A"),
                summarize_tensor(predictions_b, "critic predictions B"),
                summarize_tensor(error_a, "critic errors A"),
                summarize_tensor(error_b, "critic errors B"),
                summarize_tensor(error, "critic errors"),
                summarize_tensor(pred_actions, "predicted actions"),
                summarize_tensor(critic_values, "predicted critic values"),
                summarize_tensor(actor_loss, "actor loss"),
                summarize_tensor(critic_loss, "critic loss"),
            )

            self.logger.debug(
                "ac_grads step=%d learn_count=%d critic_a_grad_norm=%.6f critic_b_grad_norm=%.6f policy_grad_norm=%.6f "
                "critic_loss=%.6f actor_loss=%.6f",
                step,
                self._learn_count,
                float(critic_a_grad_norm) if critic_a_grad_norm is not None else -1.0,
                float(critic_b_grad_norm) if critic_b_grad_norm is not None else -1.0,
                float(policy_grad_norm) if policy_grad_norm is not None else -1.0,
                float(critic_loss.item()),
                float(actor_loss.item()),
            )

        policy_learning_rate = self.model.learning_rate('branches.policy')
        critic_learning_rate = self.model.learning_rate('branches.critic')
        critic_b_learning_rate = self.model.learning_rate('branches.critic_b')

        # Add metrics to step_logs
        learn_metrics.update({
            "extrinsic_rewards": extrinsic_rewards.mean().item(),
            "policy_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "td_errors": error.detach().flatten(),
            "td_error": error.mean().item(),
            "policy_predictions": pred_actions.mean().item(),
            "critic_predictions": critic_values.mean().item(),
            "target_policy_predictions": target_actions.mean().item(),
            "target_critic_predictions": targets.mean().item(),
            'policy_learning_rate': policy_learning_rate,
            'critic_learning_rate': critic_learning_rate,
            'critic_b_learning_rate': critic_b_learning_rate,
        })

        if self.intrinsic_motivation:
            learn_metrics.update({
                "intrinsic_loss": im_loss,
                "learn_intrinsic_reward": im_learn_rewards.mean().item(),
                "intrinsic_reward": im_rewards.mean().item(),
                "reward_weight": self.intrinsic_motivation.reward_weight * self.intrinsic_motivation.reward_scheduler.get_factor() \
                    if self.intrinsic_motivation.reward_scheduler else self.intrinsic_motivation.reward_weight
            })

        if self.noise_schedule:
            learn_metrics.update({'noise_anneal': self.noise_schedule.get_factor()})
        
        return learn_metrics

    def get_config(self):
        """Return the architecture description ``{"type", "config"}``.

        Extends the base payload with the composite model, TD3 hyperparameters
        (including twin-critic, target noise, and ``policy_update_delay``),
        normalizers, and optional intrinsic motivation.

        Returns:
            payload (dict): Mapping with ``type`` (class name) and ``config``
                (constructor kwargs suitable for ``from_config`` after env
                injection).
        """
        config = super().get_config()
        config["type"] = self.__class__.__name__
        config["config"].update({
            "model": self.model.get_config(),
            "discount": self.discount,
            "tau": self.tau,
            "action_epsilon": self.action_epsilon,
            "state_normalizer": self.state_normalizer.get_config() if self.state_normalizer is not None else None,
            "goal_normalizer": self.goal_normalizer.get_config() if self.goal_normalizer is not None else None,
            "reward_normalizer": self.reward_normalizer.get_config() if self.reward_normalizer is not None else None,
            "noise": self.noise.get_config() if self.noise is not None else None,
            "noise_schedule": self.noise_schedule.get_config() if self.noise_schedule is not None else None,
            "target_noise": self.target_noise.get_config() if self.target_noise is not None else None,
            "target_noise_schedule": self.target_noise_schedule.get_config() if self.target_noise_schedule is not None else None,
            "noise_clip": self.noise_clip,
            "raw_action_l2_coef": self.raw_action_l2_coef,
            "policy_grad_clip": self.policy_grad_clip,
            "critic_grad_clip": self.critic_grad_clip,
            "critic_huber_delta": self.critic_huber_delta,
            "policy_update_delay": self.policy_update_delay,
            "N": self.N,
            "recurrent_burn_in": self.recurrent_burn_in,
            "intrinsic_motivation": self.intrinsic_motivation.get_config() if self.intrinsic_motivation is not None else None,
        })
        return config

class SAC(Agent):
    """Soft Actor-Critic (SAC) off-policy agent.

    Assembles a composite ``ModularModel`` with a stochastic policy
    (continuous ``normal`` / ``beta`` / ``kumaraswamy``, or discrete) and twin
    Q critics. The Polyak ``target_model`` clones critics only (no target
    policy). Supports automatic entropy tuning via
    [setup_auto_entropy][phoenx.agent_utils.setup_auto_entropy] and Q-retrace
    targets via
    [compute_q_retrace][phoenx.agent_utils.compute_q_retrace].

    Critic losses own shared roots/trunk; the policy trains on detached
    shared features (off-policy gradient ownership).
    """

    MODEL_ATTRS = ("model",)
    TARGET_ATTRS = ("target_model",)
    SCHEDULE_ATTRS = ("entropy_schedule",)
    DEFAULT_SHARED_UPDATE = 'critic'

    def __init__(
        self,
        roots: Dict[str, SubNetwork] | None = None,
        trunk: SubNetwork | None = None,
        policy: StochasticDiscreteHead | StochasticContinuousHead | None = None,
        critic: ContinuousQHead | DiscreteQHead | None = None,
        critic_b: ContinuousQHead | DiscreteQHead | None = None,
        *,
        optimizer_params: dict | None = None,
        lr_scheduler: ScheduleWrapper | None = None,
        shared_update: str | None = None,
        discount: float=0.99,
        tau: float=0.005,
        state_normalizer: BaseNormalizer|None = None,
        goal_normalizer: BaseNormalizer|None = None,
        reward_normalizer: RewardNorm|None = None,
        entropy_coefficient: float=0.2, # Auto set to 1.0 if auto-tuning
        entropy_schedule: ScheduleWrapper|None = None,
        auto_entropy_tuning: bool=True,
        entropy_lr: float=3e-4, # Only used if auto entropy = True
        target_entropy_scale: float=0.98, # Only used if auto entropy = True and discrete action space
        policy_grad_clip: float = float('inf'),
        critic_grad_clip: float = float('inf'),
        critic_huber_delta: float = 1.0,
        N: int=1,
        intrinsic_motivation: IntrinsicMotivation|None = None,
        save_dir: str = "models",
        device: str|T.device|None = None,
        **kwargs
    ):
        """Initialize the SAC agent and assemble its composite model.

        When ``critic_b`` is ``None``, a fresh twin critic is cloned from
        ``critic``'s config. When ``auto_entropy_tuning`` is ``True``, builds
        ``target_entropy``, ``log_alpha``, and ``entropy_optimizer`` via
        [setup_auto_entropy][phoenx.agent_utils.setup_auto_entropy].

        Args:
            roots: Optional per-modality encoder SubNetworks (name -> SubNetwork).
            trunk: Optional shared fusion SubNetwork (may hold temporal layers).
            policy: Stochastic continuous or discrete policy head (required).
            critic: Continuous or discrete Q-head A (required).
            critic_b: Twin Q-head B; cloned from ``critic`` when ``None``.
            optimizer_params: Model-wide default optimizer spec.
            lr_scheduler: Model-wide default LR scheduler.
            shared_update: Gradient-ownership rule for shared modules; defaults
                to ``'critic'`` via ``DEFAULT_SHARED_UPDATE``.
            discount: Discount factor ``gamma`` for Q-retrace.
            tau: Polyak interpolation factor for
                [soft_update][phoenx.agent_utils.soft_update] of the target
                critics.
            state_normalizer: Optional observation normalizer.
            goal_normalizer: Optional goal normalizer for goal-conditioned envs.
            reward_normalizer: Optional reward normalizer.
            entropy_coefficient: Fixed entropy temperature when
                ``auto_entropy_tuning`` is ``False``.
            entropy_schedule: Optional multiplier schedule for
                ``entropy_coefficient`` when auto-tuning is off.
            auto_entropy_tuning: When ``True``, learn ``log_alpha`` toward a
                target entropy.
            entropy_lr: Learning rate for the entropy temperature optimizer.
            target_entropy_scale: Scale used when building the target entropy
                (especially for discrete action spaces).
            policy_grad_clip: Max grad norm for the policy branch.
            critic_grad_clip: Max grad norm for each critic branch and shared
                modules on the critic update.
            critic_huber_delta: Huber loss delta for both critic objectives.
            N: N-step / trajectory window length expected from the replay sample.
            intrinsic_motivation: Optional intrinsic motivation module trained
                inside ``learn`` and mixed into the reward used for targets.
            save_dir: Default save directory passed to the base agent.
            device: Torch device; ``None`` resolves via ``get_device``.
            **kwargs (Any): Extra attributes set on the instance via the base
                ``Agent`` constructor.
        """
        super().__init__(save_dir, device, **kwargs)
        self._check_head('policy', policy, StochasticDiscreteHead, StochasticContinuousHead)
        self._check_head('critic', critic, ContinuousQHead, DiscreteQHead)
        self._check_head('critic_b', critic_b, ContinuousQHead, DiscreteQHead, optional=True)
        # clone second critic head (fresh weights) if critic_b None
        if critic_b is None:
            critic_b = build_head(critic.get_config(), critic.env)
        self.model = self._assemble_model(
            {'policy': policy, 'critic': critic, 'critic_b': critic_b},
            roots=roots, trunk=trunk,
            optimizer_params=optimizer_params, lr_scheduler=lr_scheduler,
            shared_update=shared_update,
        )
        self.discount = discount
        self.tau = tau
        self.state_normalizer = state_normalizer
        self.goal_normalizer = goal_normalizer
        self.reward_normalizer = reward_normalizer
        self.policy_grad_clip = policy_grad_clip
        self.critic_grad_clip = critic_grad_clip
        self.critic_huber_delta = critic_huber_delta
        self.critic_loss_fn = T.nn.HuberLoss(reduction='none', delta=critic_huber_delta)
        self.N = N
        self.intrinsic_motivation = intrinsic_motivation
        # Target: critics only (SAC keeps no target policy) + shared roots/trunk
        self.target_model = self.model.clone(branches=['critic', 'critic_b'])
        self.entropy_coefficient = entropy_coefficient
        self.entropy_schedule = entropy_schedule
        self.auto_entropy_tuning = auto_entropy_tuning
        self.entropy_lr = entropy_lr
        self.target_entropy_scale = target_entropy_scale
        if self.auto_entropy_tuning:
            self.target_entropy, self.log_alpha, self.entropy_optimizer = setup_auto_entropy(
                self.policy,
                target_entropy_scale=target_entropy_scale,
                lr=entropy_lr,
                device=self.device,
            )

    def act(
        self,
        states: np.ndarray | T.Tensor,
        goals: np.ndarray | T.Tensor | None = None,
        context: str = 'train',
        step: int | None = None,
        warmup: int | None = None,
        **kwargs: Any,
    ) -> Action:
        """Select an action from the stochastic policy (or its mean in test).

        Temporal models always run a policy forward to advance recurrent /
        context state and may attach the pre-step hidden. In ``'train'``, warmup
        samples uniform env actions (with synthetic log-probs); otherwise the
        policy is sampled (continuous paths use ``sample_with_z``). In
        ``'test'``, uses mean actions. Raises ``ValueError`` for any other
        context.

        Args:
            states: Current observation(s).
            goals: Optional goal vector(s) for goal-conditioned envs.
            context: ``'train'`` to sample, ``'test'`` for deterministic /
                mean actions.
            step: Global step used with ``warmup`` for random-action warmup.
            warmup: When ``step <= warmup``, sample from the env action space.
            **kwargs (Any): May include ``dones`` (previous-step done flags)
                forwarded to `_rollout_forward` for temporal reset.

        Returns:
            action (`Action`): Package with ``actions``, optional
                ``raw_actions``, ``log_probs``, and optional ``hidden``.
        """
        raw_actions = None

        # Temporal models always run the forward (to advance the recurrent /
        # context stream) and attach the pre-step hidden for R2D2 storage.
        pol_outputs = None
        pre_hidden_flat = None
        if self.model.is_temporal:
            with T.no_grad():
                pol_outputs = self._rollout_forward(
                    states, goals=goals, branches=('policy',), dones=kwargs.get('dones'))
            if self.model.is_recurrent and self._last_pre_hidden is not None:
                pre_hidden_flat = self.model.hidden_to_tensors(self._last_pre_hidden)

        def _policy_dist():
            nonlocal pol_outputs
            if pol_outputs is None:
                pol_outputs, _ = self.model(states, goal=goals, branches=('policy',))
            return pol_outputs['policy']

        if context == 'train':
            # If warmup, sample random action from action space
            if (step is not None) and (step <= warmup):
                actions = T.as_tensor(self.model.env.action_space.sample(), device=self.device)
                if isinstance(self.policy, StochasticContinuousHead): # Continuous
                    with T.no_grad():
                        raw_actions = _policy_dist().z_from_action(actions)
                    delta = T.as_tensor(
                        self.policy.act_space.high - self.policy.act_space.low,
                        device=self.device,
                    )
                    log_probs = (-T.log(delta).sum(-1)) * T.ones(actions.shape[0], device=self.device)
                else: # Discrete
                    num_actions = T.as_tensor(self.policy.act_space.n, device=self.device)
                    log_probs = T.full((actions.shape[0],), -T.log(num_actions), device=self.device)
            
            else: # Sample action from policy
                with T.no_grad():
                    dist = _policy_dist()
                    if isinstance(self.policy, StochasticContinuousHead):
                        actions, raw_actions = dist.sample_with_z()
                        log_probs = dist.log_prob_from_z(raw_actions)
                    else: # Discrete
                        actions = dist.sample()
                        log_probs = dist.log_prob(actions)

        elif context == 'test':
            with T.no_grad():
                dist = _policy_dist()
                if isinstance(self.policy, StochasticContinuousHead):
                    actions, raw_actions = dist.mean_with_z()
                    log_probs = dist.log_prob_from_z(raw_actions)
                else: # Discrete
                    actions = self.policy.get_mean_actions(dist)
                    log_probs = dist.log_prob(actions)

        else:
            raise ValueError(f"Invalid context: {context}")

        return Action(actions, raw_actions=raw_actions, log_probs=log_probs, hidden=pre_hidden_flat)

    def soft_update_targets(self):
        """Soft-update the target critics from the online model via Polyak averaging.

        Implements [HasTargetNetworks][phoenx.rl_agents.HasTargetNetworks] using
        [soft_update][phoenx.agent_utils.soft_update] with ``self.tau``. The
        target holds both critic branches only (no target policy).
        """
        soft_update(self.model, self.target_model, self.tau)

    def learn(self, step: int, sample: dict, **kwargs: Any)->dict:
        """Apply one SAC critic-then-actor update from a replay sample.

        Builds Q-retrace targets with entropy-regularized next-state values from
        the online policy and twin target critics, updates both critics (owning
        shared roots/trunk), then updates the policy on detached shared features
        with entropy regularization. When auto-entropy is enabled, updates
        ``log_alpha`` in a separate step. Does not call ``soft_update_targets``.

        Args:
            step: Global training step (IM extrinsic gate, diagnostics).
            sample: Replay batch with ``states``, ``actions``, ``raw_actions``,
                ``log_probs``, rewards, ``intrinsic_rewards``, next states,
                terminations / truncations, ``trajectory_lengths``, goals, and
                optional PER ``weights`` / ``indices``.
            **kwargs (Any): Unused; accepted for ``Agent.learn`` API
                compatibility.

        Returns:
            metrics (dict): Loss scalars, TD errors, entropy, prediction means,
                learning rates, and optional intrinsic-motivation fields.
        """
        self._learn_count += 1
        if self._diag_freq is not None:
            should_log_diag = (self._learn_count % self._diag_freq == 0)
        else:
            should_log_diag = False

        learn_metrics = {}

        # Unpack trajectory
        states = sample["states"]
        actions = sample["actions"]
        raw_actions = sample["raw_actions"]
        buf_log_probs = sample["log_probs"]
        extrinsic_rewards = sample["rewards"]
        im_rollout_rewards = sample["intrinsic_rewards"]
        next_states = sample["next_states"]
        terminations = sample["terminations"]
        truncations = sample["truncations"]
        trajectory_lengths = sample["trajectory_lengths"]
        ach_goals = sample["state_achieved_goals"]
        next_ach_goals = sample["next_state_achieved_goals"]
        goals = sample["desired_goals"]

        if 'weights' in sample:
            weights = sample['weights']
            probs = sample['probs']
            indices = sample['indices']
        else:
            weights = None
            probs = None
            indices = None

        # Get entropy coefficient
        if self.auto_entropy_tuning:
            entropy_coefficient = self.log_alpha.exp()
        else:
            entropy_coefficient = self.entropy_coefficient
            # Apply scheduling to entropy coefficient
            if self.entropy_schedule:
                entropy_coefficient *= self.entropy_schedule.get_factor()

        # Get batch_size and n-step trajectory length
        batch_size, n_step_length = extrinsic_rewards.shape

        # Reshape arrays to (batch_size * N, *feat) to train on all steps in N
        # (feature shapes preserved; dict obs handled per key)
        states_flat = flatten_leading(states, 2)
        next_states_flat = flatten_leading(next_states, 2)
        actions_flat = actions.reshape(-1, actions.shape[-1])
        raw_actions_flat = raw_actions.reshape(-1, raw_actions.shape[-1])
        extrinsic_rewards_flat = extrinsic_rewards.reshape(-1, extrinsic_rewards.shape[-1])
        if goals is not None:
            goals_flat = goals.reshape(-1, goals.shape[-1])
            ach_goals_flat = ach_goals.reshape(-1, ach_goals.shape[-1])
            next_ach_goals_flat = next_ach_goals.reshape(-1, next_ach_goals.shape[-1])
        
        # Normalize states/goals/rewards
        if self.state_normalizer:
            states_flat = self.state_normalizer.normalize(states_flat)
            next_states_flat = self.state_normalizer.normalize(next_states_flat)
        if self.goal_normalizer:
            # ach_goals_flat = self.goal_normalizer.normalize(ach_goals_flat)
            # next_ach_goals_flat = self.goal_normalizer.normalize(next_ach_goals_flat)
            goals_flat = self.goal_normalizer.normalize(goals_flat)
        if self.reward_normalizer:
            extrinsic_rewards_flat = self.reward_normalizer.normalize(extrinsic_rewards_flat)
        
        # Train Intrinsic Motivation and get intrinsic rewards (flat vector view)
        if self.intrinsic_motivation:
            im_states = flatten_obs(states_flat)
            im_next_states = flatten_obs(next_states_flat)
            im_loss = self.intrinsic_motivation.train(im_states, im_next_states, actions_flat)
            # Compute intrinsic reward
            im_learn_rewards = self.intrinsic_motivation.compute_learn_reward(
                im_states,
                im_next_states,
                actions_flat
            )
            # Add intrinsic learn rewards to intrinsic rollout rewards
            im_rewards = im_learn_rewards.reshape(batch_size, n_step_length) + im_rollout_rewards
            # Add extrinsic reward if past step threshold
            if self.intrinsic_motivation.use_extrinsic_reward(step):
                rewards = extrinsic_rewards_flat.reshape(batch_size, n_step_length) + im_rewards
            else:
                rewards = im_rewards
        else:
            rewards = extrinsic_rewards_flat.reshape(batch_size, n_step_length)
            im_learn_rewards = T.zeros_like(rewards)
            im_rewards = T.zeros_like(rewards)

        goals_arg = goals_flat if goals is not None else None

        # --- Recurrent (R2D2) setup: sequence-mode window encodes with the
        # stored initial hidden (retrace consumes the whole window, so all
        # per-step encodings come from the exact rollout hidden stream).
        temporal = self.model.is_temporal
        hidden0 = None
        h1 = None
        if temporal:
            states_seq = unflatten_leading(states_flat, (batch_size, n_step_length))
            next_states_seq = unflatten_leading(next_states_flat, (batch_size, n_step_length))
            actions_seq = actions_flat.reshape(batch_size, n_step_length, -1)
            raw_actions_seq = raw_actions_flat.reshape(batch_size, n_step_length, -1)
            goals_seq = (goals_flat.reshape(batch_size, n_step_length, -1)
                         if goals is not None else None)
            if self.model.is_recurrent:
                stored = sample.get('initial_hidden')
                hidden0 = (self.model.hidden_from_tensors(
                              {k: v.to(self.device) for k, v in stored.items()})
                           if stored else self.model.init_hidden(batch_size))
                with T.no_grad():
                    _, h1 = self.model(
                        tree_index(states_seq, (slice(None), slice(0, 1))),
                        goal=goals_seq[:, :1] if goals is not None else None,
                        branches=('policy',), hidden=hidden0, mode='sequence')
                    h1 = self.model.detach_hidden(h1)

        with T.no_grad():
            # Get current policy for sampled states
            if temporal:
                cur_out, _ = self.model(states_seq, goal=goals_seq, branches=('policy',),
                                        hidden=hidden0, mode='sequence')
            else:
                cur_out, _ = self.model(states_flat, goal=goals_arg, branches=('policy',))
            cur_dist = cur_out['policy']

            # Get current values of sampled states and log probs of taking the sampled actions
            # Continuous Action Space
            if self.policy.distribution in ['normal', 'beta', 'kumaraswamy']:
                if temporal:
                    cur_log_probs = cur_dist.log_prob_from_z(raw_actions_seq)          # (B, N)
                    tq_out, _ = self.target_model(
                        states_seq, action=actions_seq, goal=goals_seq,
                        branches=('critic', 'critic_b'), hidden=hidden0, mode='sequence')
                    q_cur = T.minimum(tq_out['critic'], tq_out['critic_b']).reshape(batch_size, n_step_length)
                else:
                    cur_log_probs = cur_dist.log_prob_from_z(raw_actions_flat).reshape(batch_size, n_step_length)
                    tq_out, _ = self.target_model(
                        states_flat, action=actions_flat, goal=goals_arg,
                        branches=('critic', 'critic_b'))
                    q_cur = T.minimum(
                        tq_out['critic'], tq_out['critic_b']
                    ).reshape(batch_size, n_step_length)

            else: # Discrete Action Space
                if temporal:
                    cur_log_probs = cur_dist.logits.gather(
                        2, actions_seq.long()).reshape(batch_size, n_step_length)
                    tq_out, _ = self.target_model(
                        states_seq, goal=goals_seq, branches=('critic', 'critic_b'),
                        hidden=hidden0, mode='sequence')
                    q_cur_all = T.minimum(tq_out['critic'], tq_out['critic_b'])
                    q_cur = q_cur_all.gather(2, actions_seq.long()).reshape(batch_size, n_step_length)
                else:
                    cur_log_probs = cur_dist.logits.gather(1, actions_flat.long()).reshape(batch_size, n_step_length)
                    tq_out, _ = self.target_model(
                        states_flat, goal=goals_arg, branches=('critic', 'critic_b'))
                    q_cur_all = T.minimum(tq_out['critic'], tq_out['critic_b'])
                    q_cur = q_cur_all.gather(1, actions_flat.long()).reshape(batch_size, n_step_length)

            ## Critic Update ##
            if temporal:
                next_out, _ = self.model(next_states_seq, goal=goals_seq, branches=('policy',),
                                         hidden=h1, mode='sequence')
            else:
                next_out, _ = self.model(next_states_flat, goal=goals_arg, branches=('policy',))
            next_dist = next_out['policy']

            # Continuous critic target values
            if self.policy.distribution in ['normal', 'beta', 'kumaraswamy']:
                target_actions, target_z = next_dist.sample_with_z()
                next_log_probs = next_dist.log_prob_from_z(target_z)
                if temporal:
                    next_tq_out, _ = self.target_model(
                        next_states_seq, action=target_actions, goal=goals_seq,
                        branches=('critic', 'critic_b'), hidden=h1, mode='sequence')
                    q_next = T.minimum(
                        next_tq_out['critic'], next_tq_out['critic_b']).squeeze(-1)
                    target_q = (q_next - entropy_coefficient * next_log_probs).reshape(batch_size, n_step_length)
                else:
                    next_tq_out, _ = self.target_model(
                        next_states_flat, action=target_actions, goal=goals_arg,
                        branches=('critic', 'critic_b'))
                    q_next = T.minimum(
                        next_tq_out['critic'], next_tq_out['critic_b']
                    ).squeeze(-1)
                    target_q = (q_next - entropy_coefficient * next_log_probs).reshape(batch_size, n_step_length)

            else: # Discrete critic target values
                target_actions = next_dist.sample().float()
                next_log_probs = next_dist.logits
                if temporal:
                    next_tq_out, _ = self.target_model(
                        next_states_seq, goal=goals_seq, branches=('critic', 'critic_b'),
                        hidden=h1, mode='sequence')
                else:
                    next_tq_out, _ = self.target_model(
                        next_states_flat, goal=goals_arg, branches=('critic', 'critic_b'))
                q_next = T.minimum(next_tq_out['critic'], next_tq_out['critic_b'])
                target_q = (next_dist.probs * (q_next - entropy_coefficient * next_log_probs)).sum(-1).reshape(batch_size, n_step_length)

            
            q_retrace, q_metrics = compute_q_retrace(
                rewards,
                terminations,
                truncations,
                trajectory_lengths,
                q_cur,
                target_q,
                cur_log_probs,
                buf_log_probs,
                self.discount,
                device=self.device
            )
            # Collect retrace boundary diagnostics
            if q_metrics.get("done_window_final_cum_c") or q_metrics.get("done_window_max_leakage"):
                self._nstep_retrace_stats.append({
                    "done_window_final_cum_c": q_metrics["done_window_final_cum_c"],
                    "done_window_max_leakage": q_metrics["done_window_max_leakage"],
                })

            # Set low bound of q-retrace to -1/1-self.discount
            q_retrace = T.clamp(q_retrace, min=-1/(1-self.discount))

        # Reshape flat states, goals, actions to [batch_size, n-step, *feat]
        states_reshaped = unflatten_leading(states_flat, (batch_size, n_step_length))
        actions_reshaped = actions_flat.reshape(batch_size, n_step_length, -1)
        goals_reshaped = goals_flat.reshape(batch_size, n_step_length, -1) if goals is not None else None
        goals0 = goals_reshaped[:,0,:] if goals_reshaped is not None else None
        states0 = tree_index(states_reshaped, (slice(None), 0))

        # Critic predictions (one shared forward for both critics; the critic
        # loss owns the shared roots/trunk — SAC-AE/DrQ-v2 rule).
        # Temporal models encode the whole window in sequence mode with the
        # stored hidden and read the anchor (first) position.
        if self.policy.distribution in ['normal', 'beta', 'kumaraswamy']:
            if temporal:
                q_out, _ = self.model(
                    states_reshaped, action=actions_reshaped, goal=goals_reshaped,
                    branches=('critic', 'critic_b'), hidden=hidden0, mode='sequence')
                q1_preds = q_out['critic'][:, 0].reshape(batch_size)
                q2_preds = q_out['critic_b'][:, 0].reshape(batch_size)
            else:
                q_out, _ = self.model(
                    states0, action=actions_reshaped[:,0,:], goal=goals0,
                    branches=('critic', 'critic_b'))
                q1_preds = q_out['critic'].squeeze()
                q2_preds = q_out['critic_b'].squeeze()

        else: # Discrete critic predictions
            if temporal:
                q_out, _ = self.model(
                    states_reshaped, goal=goals_reshaped,
                    branches=('critic', 'critic_b'), hidden=hidden0, mode='sequence')
                q1 = q_out['critic'][:, 0]
                q2 = q_out['critic_b'][:, 0]
            else:
                q_out, _ = self.model(
                    states0, goal=goals0, branches=('critic', 'critic_b'))
                q1 = q_out['critic']
                q2 = q_out['critic_b']
            buffer_actions = actions_reshaped[:,0,:].squeeze(-1).long().unsqueeze(1)
            q1_preds = q1.gather(1, buffer_actions).squeeze(1)
            q2_preds = q2.gather(1, buffer_actions).squeeze(1)

        # Per-sample Huber loss for each critic
        q1_loss = self.critic_loss_fn(q1_preds, q_retrace.detach())
        q2_loss = self.critic_loss_fn(q2_preds, q_retrace.detach())
        # Get min raw TD error (used to update PER priorities — kept as signed difference, not Huber-transformed)
        errors = (T.minimum(q1_preds, q2_preds) - q_retrace).detach().flatten()
        # Apply importance sampling weights if using prioritized replay
        if weights is not None:
            q1_loss = weights.to(self.model.device) * q1_loss
            q2_loss = weights.to(self.model.device) * q2_loss
        critic_loss = 0.5 * (q1_loss.mean() + q2_loss.mean())

        critic_modules = (self.model.branch_module_names('critic', 'critic_b')
                          + self.model.shared_module_names())
        self.model.zero_grad()
        critic_loss.backward()
        critic_a_grad_norm = self.model.clip(
            self.critic_grad_clip, modules=self.model.branch_module_names('critic'))
        critic_b_grad_norm = self.model.clip(
            self.critic_grad_clip, modules=self.model.branch_module_names('critic_b'))
        shared_modules = self.model.shared_module_names()
        if shared_modules:
            self.model.clip(self.critic_grad_clip, modules=shared_modules)
        self.model.step(critic_modules)

        ## Update Policy ## — shared features DETACHED so the actor loss never
        ## reaches roots/trunk (exactly one owner per shared parameter)
        self.model.zero_grad()
        if temporal:
            pol_out, _ = self.model(
                states_reshaped, goal=goals_reshaped, branches=('policy',),
                hidden=hidden0, detach_shared=True, mode='sequence')
            dist_seq = pol_out['policy']
            # Continuous policy update (anchor position 0 of the window)
            if self.policy.distribution in ['normal', 'beta', 'kumaraswamy']:
                new_actions_seq, new_z_seq = dist_seq.rsample_with_z()
                log_probs = dist_seq.log_prob_from_z(new_z_seq)[:, 0]
                new_actions = new_actions_seq[:, 0]
                aq_out, _ = self.model(
                    states_reshaped, action=new_actions_seq, goal=goals_reshaped,
                    branches=('critic', 'critic_b'), hidden=hidden0,
                    detach_shared=True, mode='sequence')
                q1 = aq_out['critic'][:, 0].reshape(batch_size)
                q2 = aq_out['critic_b'][:, 0].reshape(batch_size)
                min_q = T.minimum(q1, q2)
                actor_loss = entropy_coefficient * log_probs - min_q
                dist = dist_seq
            else: # Discrete policy update at the anchor position
                new_actions = dist_seq.sample().float()[:, 0]
                log_probs = dist_seq.logits[:, 0]
                probs0 = dist_seq.probs[:, 0]
                aq_out, _ = self.model(
                    states_reshaped, goal=goals_reshaped,
                    branches=('critic', 'critic_b'), hidden=hidden0,
                    detach_shared=True, mode='sequence')
                q1 = aq_out['critic'][:, 0]
                q2 = aq_out['critic_b'][:, 0]
                min_q = T.minimum(q1, q2)
                actor_loss = (probs0 * (entropy_coefficient * log_probs - min_q)).sum(-1)
                dist = dist_seq
        else:
            pol_out, _ = self.model(
                states0, goal=goals0, branches=('policy',), detach_shared=True)
            dist = pol_out['policy']
            # Continuous policy update
            if self.policy.distribution in ['normal', 'beta', 'kumaraswamy']:
                new_actions, new_z = dist.rsample_with_z()
                log_probs = dist.log_prob_from_z(new_z)
                aq_out, _ = self.model(
                    states0, action=new_actions, goal=goals0,
                    branches=('critic', 'critic_b'), detach_shared=True)
                q1 = aq_out['critic'].squeeze()
                q2 = aq_out['critic_b'].squeeze()
                min_q = T.minimum(q1, q2)
                actor_loss = entropy_coefficient * log_probs - min_q

            else: # Discrete policy update
                new_actions = dist.sample().float()
                log_probs = dist.logits
                aq_out, _ = self.model(
                    states0, goal=goals0,
                    branches=('critic', 'critic_b'), detach_shared=True)
                q1 = aq_out['critic']
                q2 = aq_out['critic_b']
                min_q = T.minimum(q1, q2)
                actor_loss = (dist.probs * (entropy_coefficient * log_probs - min_q)).sum(-1)


        # if weights is not None:
        #     actor_loss = weights.to(self.model.device) * actor_loss

        actor_loss = actor_loss.mean()

        actor_loss.backward()
        policy_grad_norm = self.model.clip(
            self.policy_grad_clip, modules=self.model.branch_module_names('policy'))
        self.model.step(self.model.branch_module_names('policy'))

        if self.policy.distribution in ['normal', 'beta', 'kumaraswamy']:
            entropy = -log_probs
        else:  # Discrete actor
            probs_for_entropy = probs0 if temporal else dist.probs
            entropy = -(probs_for_entropy * log_probs).sum(dim=-1)
        if self.auto_entropy_tuning:
            self.entropy_optimizer.zero_grad()
            alpha_loss = -(self.log_alpha * (-entropy.mean() + self.target_entropy).detach())
            alpha_loss.backward()
            self.entropy_optimizer.step()

        # Log diag data
        if should_log_diag:
            self.logger.debug(
                "ac_diag step=%d learn_count=%d %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s",
                step,
                self._learn_count,
                summarize_tensor(states_reshaped, "states"),
                summarize_tensor(actions_reshaped, "actions"),
                summarize_tensor(rewards, "rewards"),
                summarize_tensor(im_rollout_rewards, "intrinsic rollout rewards"),
                summarize_tensor(im_learn_rewards, "intrinsic learn rewards"),
                summarize_tensor(im_rewards, "intrinsic rewards"),
                summarize_tensor(next_states, "next_states"),
                summarize_tensor(goals_reshaped, "goals"),
                summarize_tensor(next_ach_goals, "next_ach_goals"),
                summarize_tensor(terminations, "terminations"),
                summarize_tensor(truncations, "truncations"),
                summarize_tensor(target_actions, "target actions"),
                summarize_tensor(target_q, "target critic values"),
                summarize_tensor(q_retrace, "q retrace"),
                summarize_tensor(q_metrics["td_errors"], "td errors"),
                summarize_tensor(q_metrics["mask"], "mask"),
                summarize_tensor(q_metrics["is_ratio"], "is ratio"),
                summarize_tensor(q_metrics["cum_c"], "cum c"),
                summarize_tensor(q1_preds, "critic predictions A"),
                summarize_tensor(q2_preds, "critic predictions B"),
                summarize_tensor(q1_loss, "critic errors A"),
                summarize_tensor(q2_loss, "critic errors B"),
                summarize_tensor(critic_loss, "critic errors"),
                summarize_tensor(new_actions, "predicted actions"),
                summarize_tensor(log_probs, "log probs"),
                summarize_tensor(entropy, "entropy"),
                summarize_tensor(min_q, "predicted critic values"),
                summarize_tensor(actor_loss, "actor loss"),
                summarize_tensor(critic_loss, "critic loss"),
            )

            mask = q_metrics["mask"]
            cum_c_final = q_metrics["cum_c"]
            td = q_metrics["td_errors"]
            # Only look at rows that actually had a termination or truncation
            has_done = (terminations | truncations).any(dim=1)
            if has_done.any():
                for i in range(batch_size):
                    if has_done[i]:
                        L = int(trajectory_lengths[i].item())
                        done_pos = (terminations[i, :L] | truncations[i, :L]).nonzero(as_tuple=True)[0]
                        self.logger.debug(
                            "[RETRACE-DIAG] learn_count=%d "
                            "row=%d L=%d done_at=%s "
                            "final_cum_c=%.4f "
                            "mask_tail=%s",
                            step,
                            i,
                            L,
                            done_pos.tolist(),
                            cum_c_final[i].item(),
                            mask[i, done_pos[-1]+1:L].tolist() if done_pos.numel() > 0 else 'N/A',
                        )

            self.logger.debug(
                "ac_grads step=%d learn_count=%d critic_a_grad_norm=%.6f critic_b_grad_norm=%.6f policy_grad_norm=%.6f "
                "critic_loss=%.6f actor_loss=%.6f",
                step,
                self._learn_count,
                float(critic_a_grad_norm) if critic_a_grad_norm is not None else -1.0,
                float(critic_b_grad_norm) if critic_b_grad_norm is not None else -1.0,
                float(policy_grad_norm) if policy_grad_norm is not None else -1.0,
                float(critic_loss.item()),
                float(actor_loss.item()),
            )

        policy_learning_rate = self.model.learning_rate('branches.policy')
        critic_learning_rate = self.model.learning_rate('branches.critic')
        critic_b_learning_rate = self.model.learning_rate('branches.critic_b')

        learn_metrics.update({
            "extrinsic_rewards": extrinsic_rewards.mean().item(),
            "policy_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "td_errors": errors,
            "td_error": q_metrics["td_errors"].mean().item(),
            "policy_predictions": new_actions.mean().item(),
            "critic_predictions": min_q.mean().item(),
            "target_critic_predictions": target_q.mean().item(),
            "entropy_coefficient": entropy_coefficient,
            "entropy": float(entropy.mean().item()),
            'policy_learning_rate': policy_learning_rate,
            'critic_learning_rate': critic_learning_rate,
            'critic_b_learning_rate': critic_b_learning_rate,
        })
        
        if self.intrinsic_motivation:
            learn_metrics.update({
                "intrinsic_loss": im_loss,
                "learn_intrinsic_reward": im_learn_rewards.mean().item(),
                "intrinsic_reward": im_rewards.mean().item(),
                "reward_weight": self.intrinsic_motivation.reward_weight * self.intrinsic_motivation.reward_scheduler.get_factor() \
                    if self.intrinsic_motivation.reward_scheduler else self.intrinsic_motivation.reward_weight
            })

        return learn_metrics

    def get_config(self):
        """Return the architecture description ``{"type", "config"}``.

        Extends the base payload with the composite model, SAC hyperparameters
        (entropy tuning, twin critics), normalizers, and optional intrinsic
        motivation.

        Returns:
            payload (dict): Mapping with ``type`` (class name) and ``config``
                (constructor kwargs suitable for ``from_config`` after env
                injection).
        """
        config = super().get_config()
        config["type"] = self.__class__.__name__
        config["config"].update({
            "model": self.model.get_config(),
            "discount": self.discount,
            "tau": self.tau,
            "state_normalizer": self.state_normalizer.get_config() if self.state_normalizer is not None else None,
            "goal_normalizer": self.goal_normalizer.get_config() if self.goal_normalizer is not None else None,
            "reward_normalizer": self.reward_normalizer.get_config() if self.reward_normalizer is not None else None,
            "entropy_coefficient": self.entropy_coefficient,
            "entropy_schedule": self.entropy_schedule.get_config() if self.entropy_schedule is not None else None,
            "auto_entropy_tuning": self.auto_entropy_tuning,
            "entropy_lr": self.entropy_lr,
            "target_entropy_scale": self.target_entropy_scale,
            "policy_grad_clip": self.policy_grad_clip,
            "critic_grad_clip": self.critic_grad_clip,
            "critic_huber_delta": self.critic_huber_delta,
            "N": self.N,
            "intrinsic_motivation": self.intrinsic_motivation.get_config() if self.intrinsic_motivation is not None else None,
        })
        return config


@runtime_checkable
class HasTargetNetworks(Protocol):
    """Protocol for agents that maintain Polyak-averaged target networks."""

    def soft_update_targets(self) -> None:
        """Soft-update every target network from its online counterpart."""
        ...


# Registry of every concrete agent class, keyed by class name (the "type" tag
# emitted by Agent.get_config). Used by build_agent to reconstruct from a config.
AGENT_REGISTRY: Dict[str, type] = {
    "Reinforce": Reinforce,
    "ActorCritic": ActorCritic,
    "PPO": PPO,
    "DDPG": DDPG,
    "TD3": TD3,
    "SAC": SAC,
}


def build_agent(config: dict, env: EnvWrapper) -> "Agent":
    """Rebuild an agent from a ``{"type", "config"}`` dict, injecting ``env``.

    Single entry point for turning a saved agent config into a live
    (fresh-tensor) agent. Tensor state is restored afterwards via
    [Agent.load_state][phoenx.rl_agents.Agent.load_state].

    Args:
        config: Mapping with ``type`` (registry key) and ``config`` (inner
            constructor kwargs passed to ``from_config``).
        env: Live environment injected into rebuilt models.

    Returns:
        Fresh agent instance of the requested type.

    Raises:
        KeyError: If ``config`` has no ``"type"`` key.
        ValueError: If ``config["type"]`` is not a key in ``AGENT_REGISTRY``.
    """
    agent_type = config["type"]
    if agent_type not in AGENT_REGISTRY:
        raise ValueError(
            f"Unknown agent type: {agent_type!r}. Available: {list(AGENT_REGISTRY)}"
        )
    return AGENT_REGISTRY[agent_type].from_config(config["config"], env)