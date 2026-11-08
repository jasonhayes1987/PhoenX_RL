"""Self-tuning coefficient for a KL-divergence penalty.

PPO can discourage large policy updates by adding ``beta * KL`` to its policy
loss. A fixed ``beta`` is awkward to pick: one that meaningfully constrains
early updates keeps throttling the policy long after it has settled. This
module adjusts the coefficient from the KL the updates actually produce.

``AdaptiveKL`` owns that single number and nothing else. A PPO agent reads the
current coefficient through ``get_beta`` before an update and reports the
measured divergence to ``step`` afterwards, which multiplies ``beta`` up or
down whenever the observation strays outside the tolerance band around
``target_kl``. Attaching one is optional; without it ``kl_coefficient`` stays
at whatever the agent was configured with.

Configuration and mutable state are deliberately separate so a run can resume
mid-schedule: ``get_config`` returns the constructor arguments, while
``get_state`` / ``set_state`` carry only the live ``beta``.
"""

class AdaptiveKL():
    """Keeps track of a KL penalty coefficient `beta`.

    The KL penalty coefficient `beta` is adjusted after each update so the observed
    KL divergence hovers near `target_kl`.
    """
    def __init__(
        self,
        initial_beta=1.0,
        beta_max=100.0,
        target_kl=0.01,
        scale_up=2.0,
        scale_down=0.5,
        kl_tolerance_high=1.5,
        kl_tolerance_low=0.5):
        """Initializes the AdaptiveKL object.

        Args:
            initial_beta (float): initial KL penalty
            beta_max (float): maximum value of beta
            target_kl (float): desired KL divergence
            scale_up (float): factor by which to increase beta if KL is too high
            scale_down (float): factor by which to reduce beta if KL is too low
            kl_tolerance_high (float): if observed KL > target_kl * kl_tolerance_high,
                                   we consider that "too high"
            kl_tolerance_low (float): if observed KL < target_kl * kl_tolerance_low,
                                  we consider that "too low"
        """
        self.initial_beta = initial_beta
        self.beta = initial_beta
        self.beta_max = beta_max
        self.target_kl = target_kl
        self.scale_up = scale_up
        self.scale_down = scale_down
        self.kl_tolerance_high = kl_tolerance_high
        self.kl_tolerance_low = kl_tolerance_low

    def step(self, observed_kl: float) -> None:
        """Update beta based on how the observed KL compares to target_kl.

        Typically called after each PPO update (once you can measure KL).

        Args:
            observed_kl (float): the observed KL divergence
        """
        # If KL is way above target, raise beta
        if observed_kl > self.target_kl * self.kl_tolerance_high:
            self.beta = min(self.beta * self.scale_up, self.beta_max)
        # If KL is much below target, lower beta
        elif observed_kl < self.target_kl * self.kl_tolerance_low:
            self.beta *= self.scale_down

    def get_beta(self) -> float:
        """Return the current value of beta."""
        return self.beta

    def get_config(self) -> dict:
        """Return the configuration of the AdaptiveKL object."""
        return {
            "initial_beta": self.initial_beta,
            "beta_max": self.beta_max,
            "target_kl": self.target_kl,
            "scale_up": self.scale_up,
            "scale_down": self.scale_down,
            "kl_tolerance_high": self.kl_tolerance_high,
            "kl_tolerance_low": self.kl_tolerance_low
        }

    def get_state(self) -> dict:
        """Return the mutable state (current beta) for resuming training."""
        return {"beta": self.beta}

    def set_state(self, state: dict) -> None:
        """Restore the mutable state produced by [get_state][phoenx.adaptive_kl.AdaptiveKL.get_state].

        Args:
            state (dict): the state to restore
        """
        if state:
            self.beta = state.get("beta", self.beta)