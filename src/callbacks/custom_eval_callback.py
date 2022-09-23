import numpy as np
from stable_baselines3.common.callbacks import EvalCallback


class CustomEvalCallback(EvalCallback):
    """
    Callback used for evaluating and reporting a trial.
    """
    def __init__(self, eval_env, n_eval_episodes=10,
                 eval_freq=10000, deterministic=True, verbose=0):
        super(CustomEvalCallback, self).__init__(eval_env=eval_env, n_eval_episodes=n_eval_episodes,
                                                 eval_freq=eval_freq,
                                                 deterministic=deterministic,
                                                 verbose=verbose)

        self._used_std = []

    def _on_step(self):
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if self.model.actor._last_used_log_std is not None:
                std = np.power(10, self.model.actor._last_used_log_std)
                self._used_std.append(std)
                self.logger.record('last_used_std', std)

        return super(CustomEvalCallback, self)._on_step()
