from openforge.hyperparameter_optimization.bo_optimizer import get_bo_optimizer


class TuningEngine:
    def __init__(self, config, mrf_wrapper, mrf_hp_space):
        self.config = config
        self.mrf_wrapper = mrf_wrapper
        self.mrf_config_space = mrf_hp_space

        self.optimizer = get_bo_optimizer(
            self.config, self.mrf_config_space, self.bo_target_function
        )

    def bo_target_function(self, mrf_hp_space):
        """
        Target function for Bayesian Optimization.
        """

        f1_score = self.mrf_wrapper.run(mrf_hp_space)

        return -f1_score

    def run(self):
        best_hp = self.optimizer.optimize()

        return best_hp
