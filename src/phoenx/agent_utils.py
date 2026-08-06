"""Return estimators and shared update helpers used by the agents.

These are the stateless numeric pieces the agents in ``phoenx.rl_agents`` call
during an update: N-step and Monte Carlo returns, TD errors, GAE, and the
Retrace-corrected Q target. Alongside them sit a few helpers that act on
modules and optimizers rather than raw tensors — auto-entropy setup, Polyak
averaging, and gradient-norm reporting.

Conventions:
    - per-timestep tensors are laid out ``[timesteps, num_envs]``, and N-step
      windows are ``[batch_size, N]``;
    - ``device`` arguments are resolved through ``torch_utils.get_device``, so
      ``None`` selects the framework default;
    - the ``compute_*`` functions return new tensors and leave their inputs
      untouched, while ``soft_update`` writes into the target module in place.
"""

import torch as T
from torch.optim import Optimizer
from .torch_utils import get_device

def compute_n_step_return(
    rewards: T.Tensor,           # [batch_size, N]
    gamma: float,
    device:
    T.device|str|None = None
) -> T.Tensor:
    """Compute N-step returns for a batch of sequences.

    Args:
        rewards: Tensor of rewards [batch_size, N].
        gamma: Discount factor.
        device: Device for tensor operations.

    Returns:
        Tensor of N-step returns [batch_size].
    """
    device = get_device(device)
    batch_size, N = rewards.shape
    discount_factors = T.pow(gamma, T.arange(N, device=device).float()).unsqueeze(0).expand(batch_size, N)

    return (rewards * discount_factors).sum(dim=1)

def compute_td_error(
    rewards: T.Tensor,
    values: T.Tensor,
    next_values: T.Tensor,
    terminations: T.Tensor,
    truncations: T.Tensor,
    gamma: float,
    bootstrap_truncations: bool
    ) -> T.Tensor:
    """Compute TD errors for a batch of trajectories.

    Args:
        rewards: Tensor of rewards [batch_size, num_envs].
        values: Tensor of values [batch_size, num_envs].
        next_values: Tensor of next values [batch_size, num_envs].
        terminations: Tensor of termination flags [batch_size, num_envs].
        truncations: Tensor of truncation flags [batch_size, num_envs].
        gamma: Discount factor.
        bootstrap_truncations: Whether to bootstrap the returns on truncated episodes.

    Returns:
        Tensor of TD errors [batch_size, num_envs].
    """
    if bootstrap_truncations:
        dones = terminations
    else:
        dones = T.logical_or(terminations, truncations)
    return rewards + gamma * next_values * T.logical_not(dones) - values

def compute_monte_carlo_returns(
    rewards:T.Tensor,
    gamma:float,
    device:T.device|str|None = None
) -> T.Tensor:
    """Compute discounted returns for each step in a trajectory.

    Args:
        rewards: Tensor of rewards [batch_size, num_envs].
        gamma: Discount factor.
        device: Device for tensor operations.

    Returns:
        Tensor of discounted returns [batch_size, num_envs].
    """
    device = get_device(device)
    returns = []
    discounted_return = 0.0
    for reward in reversed(rewards):
        discounted_return = reward + gamma * discounted_return
        returns.append(discounted_return)
    returns.reverse()
    return T.tensor(returns, device=device)

def compute_gae(
    td_errors:T.Tensor,
    terminations:T.Tensor,
    truncations:T.Tensor,
    gamma:float,
    gae_lambda:float,
    device:T.device|str|None = None
    ) -> T.Tensor:
    """Compute Generalized Advantage Estimation (GAE) for a batch of TD errors.

    Args:
        td_errors: Tensor of TD errors [timesteps, num_envs].
        terminations: Tensor of termination flags [timesteps, num_envs].
        truncations: Tensor of truncation flags [timesteps, num_envs].
        gamma: Discount factor.
        gae_lambda: GAE lambda parameter.
        device: Device for tensor operations.

    Returns:
        Tensor of advantages [timesteps, num_envs].
    """
    device = get_device(device)
    timesteps, num_envs = td_errors.shape
    advantages = T.zeros(timesteps, num_envs, device=device)
    advantage = T.zeros(num_envs, device=device)

    # if bootstrap_truncations:
    #     dones = T.logical_or(terminations, first_steps)
    # else:
    dones = T.logical_or(terminations, truncations)

    for t in reversed(range(timesteps)):
        advantage = td_errors[t] + gamma * gae_lambda * advantage * T.logical_not(dones[t])
        advantages[t] = advantage
    return advantages

def compute_advantages_and_returns(
    rewards:T.Tensor,
    values:T.Tensor,
    next_values:T.Tensor,
    terminations:T.Tensor,
    truncations:T.Tensor,
    gamma:float,
    gae_lambda:float,
    bootstrap_truncations: bool,
    device:T.device|str|None = None
) -> tuple[T.Tensor, T.Tensor, T.Tensor]:
    """Compute advantages and returns for a batch of trajectories.

    Args:
        rewards: Tensor of rewards [batch_size, num_envs].
        values: Tensor of values [batch_size, num_envs].
        next_values: Tensor of next values [batch_size, num_envs].
        terminations: Tensor of termination flags [batch_size, num_envs].
        truncations: Tensor of truncation flags [batch_size, num_envs].
        gamma: Discount factor.
        gae_lambda: GAE lambda parameter.
        bootstrap_truncations: Whether to bootstrap the returns on truncated episodes.
        device: Device for tensor operations.

    Returns:
        Tensor of advantages [batch_size, num_envs].
        Tensor of returns [batch_size, num_envs].
        Tensor of TD errors [batch_size, num_envs].
    """
    device = get_device(device)
    td_errors = compute_td_error(rewards, values, next_values, terminations, truncations, gamma, bootstrap_truncations)
    advantages = compute_gae(td_errors, terminations, truncations, gamma, gae_lambda, device)
    returns = advantages + values
    return advantages, returns, td_errors

def compute_q_retrace(
    rewards: T.Tensor,
    terminations: T.Tensor,
    truncations: T.Tensor,
    trajectory_lengths: T.Tensor,
    q_cur: T.Tensor,
    target_q: T.Tensor,
    cur_log_probs: T.Tensor,
    buf_log_probs: T.Tensor,
    discount: float,
    *,
    device: T.device | str | None = None
) -> tuple[T.Tensor, dict[str, T.Tensor]]:
    """Computes target Q values as sum of weighted TD errors using importance sampling ratios across the n-step window.

    Args:
        rewards: Tensor of rewards [batch_size, n_step_length].
        terminations: Tensor of termination flags [batch_size, n_step_length].
        truncations: Tensor of truncation flags [batch_size, n_step_length].
        trajectory_lengths: Tensor of trajectory lengths [batch_size].
        q_cur: Tensor of current Q values [batch_size, n_step_length].
        target_q: Tensor of target Q values [batch_size, n_step_length].
        cur_log_probs: Tensor of current log probabilities [batch_size, n_step_length].
        buf_log_probs: Tensor of log probabilities of the buffer [batch_size, n_step_length].
        discount: Discount factor.
        device: Device for tensor operations.

    Returns:
        Tensor of Q values [batch_size].
        Dictionary of metrics [td_errors, mask, is_ratio, cum_c].
    """
    device = get_device(device)
    batch_size, n_step_length = rewards.shape
    # Compute TD errors across n-step window
    td_errors = rewards + discount * (1 - terminations.float()) * target_q.detach() - q_cur.detach()

    # Compute IS ratios
    is_ratio = T.clamp(T.exp(cur_log_probs - buf_log_probs), min=0.5, max=1.0)
    # Mask invalid steps and IS ratios from terminated_state +1 : N
    valid = (T.arange(n_step_length, device=device)[None, :] < trajectory_lengths[:, None]).float()
    mask = T.ones(batch_size, n_step_length, device=device)
    dones = T.logical_or(terminations, truncations)
    for k in range(1, n_step_length):
        mask[:, k] = mask[:, k-1] * (1 - dones[:, k-1].float()) * valid[:, k]
    is_ratio = is_ratio * mask

    # Compute q retrace
    cum_c = T.ones(batch_size, device=device)
    retrace_sum = T.zeros(batch_size, device=device)

    for k in range(n_step_length):
        gamma = discount ** k
        retrace_sum += gamma * cum_c * td_errors[:, k]
        # Update cumulative weight IS ratio
        if k < n_step_length - 1:
            cum_c = cum_c * is_ratio[:, k+1]

    q_retrace = q_cur[:, 0] + retrace_sum

    # Compute boundary leakage diagnostics
    # done_window_final_cum_c = []
    # done_window_max_leakage = []
    # has_done = (terminations | truncations).any(dim=1)
    # if has_done.any():
    #     for i in range(batch_size):
    #         if has_done[i]:
    #             L = int(trajectory_lengths[i].item())
    #             if L > 0:
    #                 done_mask = (terminations[i, :L] | truncations[i, :L])
    #                 if done_mask.any():
    #                     first_done = int(done_mask.nonzero(as_tuple=True)[0][0])
    #                     done_window_final_cum_c.append(float(cum_c[i].item()))
    #                     if first_done + 1 < L:
    #                         leakage = mask[i, first_done + 1 : L].max().item()
    #                         done_window_max_leakage.append(leakage)
    metrics = {
        "td_errors": td_errors,
        "mask": mask,
        "is_ratio": is_ratio,
        "cum_c": cum_c,
        # "done_window_final_cum_c": done_window_final_cum_c,
        # "done_window_max_leakage": done_window_max_leakage,
    }

    return q_retrace, metrics

def setup_auto_entropy(policy, *, target_entropy_scale=0.98,
                      lr=3e-4, device=None):
    """Build the target entropy, log-alpha, and optimizer for entropy tuning.

    The target depends on the policy's action distribution: continuous
    distributions target ``-num_actions`` and discrete ones ``log(num_actions)``,
    each scaled by ``target_entropy_scale``.

    Args:
        policy (torch.nn.Module): Policy exposing ``distribution`` and
            ``num_actions``, from which the target entropy is derived.
        target_entropy_scale (float): Fraction of the reference entropy to aim
            for.
        lr (float): Learning rate for the log-alpha optimizer.
        device (torch.device | str | None): Device the log-alpha parameter is
            allocated on.

    Returns:
        target_entropy (float): Entropy the policy is tuned toward.
        log_alpha (torch.Tensor): Learnable log-alpha, requiring grad.
        optimizer (torch.optim.Adam): Optimizer that updates ``log_alpha``.
    """
    if policy.distribution in ("normal", "beta", "kumaraswamy"):
        target_entropy = target_entropy_scale * -float(policy.num_actions)
    else:  # discrete
        target_entropy = target_entropy_scale * T.log(
            T.tensor(policy.num_actions, dtype=T.float32, device=device)
        ).item()
    log_alpha = T.zeros(1, requires_grad=True, device=device)
    optimizer = T.optim.Adam([log_alpha], lr=lr)
    return target_entropy, log_alpha, optimizer

@T.no_grad()
def soft_update(current_module, target_module, tau: float) -> None:
    """Soft update a module's parameters and buffers to target_module.

    Parameters and buffers are matched BY NAME (not by position), so the
    target may be a subset of the current module (e.g. a branch-subset clone
    of a ModularModel used as a target network).

    Args:
        current_module (torch.nn.Module): Module to read parameters from.
        target_module (torch.nn.Module): Module updated in place; may hold a
            subset of ``current_module``'s named parameters.
        tau: Interpolation factor, where ``1.0`` copies outright.
    """
    cur_params = dict(current_module.named_parameters())
    for name, tp in target_module.named_parameters():
        cp = cur_params.get(name)
        if cp is not None:
            tp.data.lerp_(cp.data, tau)
    cur_buf = dict(current_module.named_buffers())
    for name, tbuf in target_module.named_buffers():
        if name in cur_buf:
            tbuf.copy_(cur_buf[name])

def grad_norm_from_optimizer(optimizer: Optimizer) -> float:
    """Compute the global L2 norm of the gradients an optimizer owns.

    Only parameters carrying a populated ``.grad`` contribute, so calling this
    before ``backward`` — or after ``zero_grad(set_to_none=True)`` — reports
    zero rather than raising.

    Args:
        optimizer: Optimizer whose ``param_groups`` are scanned.

    Returns:
        Global gradient L2 norm, or ``0.0`` when no parameter carries one.
    """
    total_sq = None
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is None:
                continue
            grad_sq = p.grad.detach().pow(2).sum()
            total_sq = grad_sq if total_sq is None else total_sq + grad_sq
    return float(T.sqrt(total_sq)) if total_sq is not None else 0.0
