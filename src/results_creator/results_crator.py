from stable_baselines3.common.callbacks import EvalCallback


class ResultsCreator:
    def __init__(self, evaluation_callback_object: EvalCallback):
        self._evaluation_callback_object = evaluation_callback_object
        self._results_dict = None

    @property
    def results_dict(self) -> dict:
        if self._results_dict is None:
            self._results_dict = self._generate_results_dict()

        return self._results_dict

    def _generate_results_dict(self) -> dict:
        results_dir = {
            "eval_return_mean": self._evaluation_callback_object.best_mean_reward,
            "number_of_evaluation_episodes": self._evaluation_callback_object.n_eval_episodes,
            "num_timesteps": self._evaluation_callback_object.num_timesteps
        }

        return results_dir
