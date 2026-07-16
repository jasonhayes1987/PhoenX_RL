from scripts.agent import create_intrinsic_motivation, create_normalizer, create_policy
from app.rl_agents import PPO
from app.env_wrapper import EnvWrapper
from app.models import ValueModel
from app.normalizer import create_normalizer as normalizer_factory
from app.schedulers import ScheduleWrapper
from app.adaptive_kl import AdaptiveKL


def create_value(config: dict, env: EnvWrapper) -> ValueModel:
    """Create a value model from a config."""
    config = dict(config)
    config["env"] = env
    config["lr_scheduler"] = (
        ScheduleWrapper(**config["lr_scheduler"]) if config.get("lr_scheduler", None) else None
    )
    return ValueModel(**config)


def build(config: dict, env: EnvWrapper):
    agent_cfg = config["agent"]["config"]

    # build policy / value
    agent_cfg["policy"] = create_policy(agent_cfg["policy"], env)
    agent_cfg["value"] = create_value(agent_cfg["value"], env)

    # normalizers
    if agent_cfg.get("state_normalizer", None):
        agent_cfg["state_normalizer"] = create_normalizer(
            agent_cfg["state_normalizer"], env, config["env"]["config"]["obs_key"]
        )
    else:
        agent_cfg["state_normalizer"] = None

    if agent_cfg.get("goal_normalizer", None):
        agent_cfg["goal_normalizer"] = create_normalizer(
            agent_cfg["goal_normalizer"], env, config["env"]["config"]["goal_key"]
        )
    else:
        agent_cfg["goal_normalizer"] = None

    if agent_cfg.get("advantage_normalizer", None):
        # Advantage is scalar; do not infer dims from env obs (Isaac Dict spaces).
        adv_cfg = dict(agent_cfg["advantage_normalizer"])
        adv_cfg["config"] = dict(adv_cfg.get("config", {}))
        adv_cfg["config"]["num_features"] = 1
        agent_cfg["advantage_normalizer"] = normalizer_factory(adv_cfg)
    else:
        agent_cfg["advantage_normalizer"] = None

    if agent_cfg.get("reward_normalizer", None):
        agent_cfg["reward_normalizer"] = create_normalizer(agent_cfg["reward_normalizer"], env)
    else:
        agent_cfg["reward_normalizer"] = None

    # schedules / adapters
    if agent_cfg.get("entropy_schedule", None):
        agent_cfg["entropy_schedule"] = ScheduleWrapper(**agent_cfg["entropy_schedule"])
    else:
        agent_cfg["entropy_schedule"] = None

    if agent_cfg.get("policy_clip_schedule", None):
        agent_cfg["policy_clip_schedule"] = ScheduleWrapper(**agent_cfg["policy_clip_schedule"])
    else:
        agent_cfg["policy_clip_schedule"] = None

    if agent_cfg.get("value_clip_schedule", None):
        agent_cfg["value_clip_schedule"] = ScheduleWrapper(**agent_cfg["value_clip_schedule"])
    else:
        agent_cfg["value_clip_schedule"] = None

    if agent_cfg.get("kl_adapter", None):
        agent_cfg["kl_adapter"] = AdaptiveKL(**agent_cfg["kl_adapter"])
    else:
        agent_cfg["kl_adapter"] = None

    # intrinsic motivation
    if agent_cfg.get("intrinsic_motivation", None):
        agent_cfg["intrinsic_motivation"] = create_intrinsic_motivation(
            agent_cfg["intrinsic_motivation"], env, config["env"]["config"]["obs_key"]
        )
    else:
        agent_cfg["intrinsic_motivation"] = None

    return PPO(**agent_cfg)
