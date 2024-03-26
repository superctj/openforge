import logging

from ConfigSpace import ConfigurationSpace


class ExperimentState:
    def __init__(self):
        self.default_f1_score = -1
        self.default_associated_accuracy = -1

        self.best_f1_score = -1
        self.best_associated_accuracy = -1
        self.best_hp_config = None

        self.worst_f1_score = 1
        self.worst_associated_accuracy = 1
        self.worst_hp_config = None

    def log_best_hp_config(
        self, best_hp_config: ConfigurationSpace, logger: logging.Logger
    ):
        logger.info("\nCompleted hyperparamter optimization.\n")

        logger.info(f"Best hyperparameter configuration:\n{best_hp_config}")
        logger.info(f"Best F1 score: {self.best_f1_score:.2f}")
        logger.info(
            f"Associated accuracy: {self.best_associated_accuracy:.2f}\n"
        )

        logger.info(
            f"Worst hyperparameter configuration:\n{self.worst_hp_config}"
        )
        logger.info(f"Worst F1 score: {self.worst_f1_score:.2f}")
        logger.info(
            f"Associated accuracy: {self.worst_associated_accuracy:.2f}\n"
        )

        assert dict(best_hp_config) == dict(self.best_hp_config)
