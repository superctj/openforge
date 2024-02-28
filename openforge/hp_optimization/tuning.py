from ConfigSpace import ConfigurationSpace

from openforge.hp_optimization.bo_optimizer import get_bo_optimizer
from openforge.utils.custom_logging import get_logger
from openforge.utils.exp_tracker import ExperimentState
from openforge.utils.mrf_common import evaluate_inference_results


class TuningEngine:
    def __init__(self, exp_config, mrf_wrapper, mrf_hp_space):
        self.exp_config = exp_config
        self.mrf_wrapper = mrf_wrapper
        self.mrf_hp_space = mrf_hp_space

        self.optimizer = get_bo_optimizer(
            self.exp_config, self.mrf_hp_space, self.bo_target_function
        )
        self.exp_state = ExperimentState()
        self.logger = get_logger()

    def bo_target_function(self, mrf_hp_config: ConfigurationSpace, seed: int):
        """
        Target function for Bayesian Optimization.
        """

        mrf = self.mrf_wrapper.create_mrf(dict(mrf_hp_config))
        results = self.mrf_wrapper.run_mplp_inference(mrf)
        f1_score, accuracy = evaluate_inference_results(
            self.mrf_wrapper.prior_data, results
        )

        if self.exp_state.best_f1_score < f1_score:
            self.exp_state.best_f1_score = f1_score
            self.exp_state.best_associated_accuracy = accuracy
            self.exp_state.best_hp_config = mrf_hp_config

        if self.exp_state.worst_f1_score > f1_score:
            self.exp_state.worst_f1_score = f1_score
            self.exp_state.worst_associated_accuracy = accuracy
            self.exp_state.worst_hp_config = mrf_hp_config

        return -f1_score

    def run(self):
        best_hp_config = self.optimizer.optimize()

        self.logger.info("\nCompleted hyperparamter optimization.")

        self.logger.info(f"\nBest MRF hyperparameters:\n{best_hp_config}")
        self.logger.info(f"Best F1 score: {self.exp_state.best_f1_score:.2f}")
        self.logger.info(
            f"Accuracy: {self.exp_state.best_associated_accuracy:.2f}"
        )

        self.logger.info(
            f"\nWorst MRF hyperparameters:\n{self.exp_state.worst_hp_config}"
        )
        self.logger.info(f"Worst F1 score: {self.exp_state.worst_f1_score:.2f}")
        self.logger.info(
            f"Accuracy: {self.exp_state.worst_associated_accuracy:.2f}"
        )

        assert dict(best_hp_config) == dict(self.exp_state.best_hp_config)
        return best_hp_config
