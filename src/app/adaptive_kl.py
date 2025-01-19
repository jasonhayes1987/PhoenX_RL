
class AdaptiveKL():
    """
    Keeps track of a KL penalty coefficient `beta` that is adjusted
    after each update so the observed KL divergence hovers near `target_kl`.
    """
    def __init__(self, initial_beta=1.0, target_kl=0.01,
                 scale_up=2.0, scale_down=0.5,
                 kl_tolerance_high=1.5, kl_tolerance_low=0.5):
        """
        Args:
            initial_beta (float): initial KL penalty
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
        self.target_kl = target_kl
        self.scale_up = scale_up
        self.scale_down = scale_down
        self.kl_tolerance_high = kl_tolerance_high
        self.kl_tolerance_low = kl_tolerance_low

    def step(self, observed_kl):
        """
        Update beta based on how the observed KL compares to target_kl.
        Typically called after each PPO update (once you can measure KL).
        """
        # If KL is way above target, raise beta
        if observed_kl > self.target_kl * self.kl_tolerance_high:
            self.beta *= self.scale_up
        # If KL is much below target, lower beta
        elif observed_kl < self.target_kl * self.kl_tolerance_low:
            self.beta *= self.scale_down

    def get_beta(self):
        return self.beta
    
    def get_config(self):
        return {
            "initial_beta": self.initial_beta,
            "target_kl": self.target_kl,
            "scale_up": self.scale_up,
            "scale_down": self.scale_down,
            "kl_tolerance_high": self.kl_tolerance_high,
            "kl_tolerance_low": self.kl_tolerance_low
        }