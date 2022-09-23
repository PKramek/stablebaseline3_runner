from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm


class TqdmCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.progress_bar = None

    def _on_training_start(self):
        self.progress_bar = tqdm(total=self.locals['total_timesteps'])

    def _on_step(self):
        update_value = self.update_value = self.model.n_envs
        self.progress_bar.update(update_value)
        return True

    def _on_training_end(self):
        self.progress_bar.close()
        self.progress_bar = None
