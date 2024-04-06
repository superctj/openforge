from configparser import ConfigParser

from ConfigSpace import ConfigurationSpace

from openforge.hp_optimization.bo_optimizer import get_bo_optimizer
from openforge.utils.custom_logging import get_logger
from openforge.utils.exp_tracker import ExperimentState
from openforge.utils.mrf_common import (
    evaluate_inference_results,
    evaluate_multi_class_inference_results,
)
from openforge.utils.prior_model_common import (
    evaluate_prior_model_predictions,
)


class TuningEngine:
    def __init__(
        self, exp_config, mrf_wrapper, mrf_hp_space, multi_class=False
    ):
        self.exp_config = exp_config
        self.mrf_wrapper = mrf_wrapper
        self.mrf_hp_space = mrf_hp_space
        self.multi_class = multi_class

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
        results = self.mrf_wrapper.run_inference(mrf, dict(mrf_hp_config))

        if not self.multi_class:
            f1_score, accuracy, _, _ = evaluate_inference_results(
                self.mrf_wrapper.prior_data, results
            )
        else:
            f1_score, accuracy, _, _ = evaluate_multi_class_inference_results(
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

        self.exp_state.log_best_hp_config(best_hp_config, self.logger)

        return best_hp_config


class PriorModelTuningEngine:
    def __init__(
        self,
        exp_config: ConfigParser,
        prior_model_wrapper: object,
        hp_space: ConfigurationSpace,
        multi_class: bool = False,
    ):
        self.exp_config = exp_config
        self.prior_model_wrapper = prior_model_wrapper
        self.hp_space = hp_space
        self.multi_class = multi_class

        self.optimizer = get_bo_optimizer(
            self.exp_config, self.hp_space, self.bo_target_function
        )
        self.exp_state = ExperimentState()
        self.logger = get_logger()

    def bo_target_function(self, hp_config: ConfigurationSpace, seed: int):
        """Target function for Bayesian Optimization."""

        self.prior_model_wrapper.create_model(dict(hp_config))
        self.prior_model_wrapper.fit()

        y_valid_pred = self.prior_model_wrapper.predict(
            self.prior_model_wrapper.X_valid
        )

        f1_score, accuracy, _, _ = evaluate_prior_model_predictions(
            self.prior_model_wrapper.y_valid,
            y_valid_pred,
            multi_class=self.multi_class,
        )

        if self.exp_state.best_f1_score < f1_score:
            self.exp_state.best_f1_score = f1_score
            self.exp_state.best_associated_accuracy = accuracy
            self.exp_state.best_hp_config = hp_config

        if self.exp_state.worst_f1_score > f1_score:
            self.exp_state.worst_f1_score = f1_score
            self.exp_state.worst_associated_accuracy = accuracy
            self.exp_state.worst_hp_config = hp_config

        return -f1_score

    def run(self):
        best_hp_config = self.optimizer.optimize()

        self.exp_state.log_best_hp_config(best_hp_config, self.logger)

        return best_hp_config
