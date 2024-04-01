import argparse
import logging
import os
import pickle

import numpy as np

from sklearn.linear_model import RidgeClassifier
from sklearn.utils import extmath

from openforge.hp_optimization.hp_space import HyperparameterSpace
from openforge.hp_optimization.tuning import PriorModelTuningEngine
from openforge.utils.custom_logging import create_custom_logger
from openforge.utils.prior_model_common import (
    load_openforge_sotab_small,
    log_exp_metrics,
    log_exp_records,
)
from openforge.utils.util import fix_global_random_state, parse_config


class RidgeClassifierTuningWrapper:
    def __init__(self, data_dir: str, random_seed: int, logger: logging.Logger):
        (
            self.X_train,
            self.y_train,
            self.X_valid,
            self.y_valid,
            self.X_test,
            self.y_test,
            self.train_df,
            self.valid_df,
            self.test_df,
        ) = load_openforge_sotab_small(data_dir, logger)
        self.clf = None
        self.random_seed = random_seed

    def create_default_model(self):
        self.clf = RidgeClassifier(
            class_weight="balanced",
            random_state=self.random_seed,
        )

    def create_model(self, hp_config: dict):
        self.clf = RidgeClassifier(
            alpha=hp_config["alpha"],
            tol=hp_config["tol"],
            class_weight="balanced",
            random_state=self.random_seed,
        )

    def fit(self):
        self.clf.fit(self.X_train, self.y_train)

    def predict(self, X: np.ndarray, threshold: float = 0.5):
        y_pred = (self.predict_proba(X)[:, 1] >= threshold).astype(int)

        return y_pred

    def predict_proba(self, X: np.ndarray):
        d = self.clf.decision_function(X)
        d_2d = np.c_[-d, d]

        return extmath.softmax(d_2d)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the experiment configuration file.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="train_w_hp_tuning",
        help="Mode: train_w_default_hp, train_w_hp_tuning or test.",
    )

    args = parser.parse_args()

    # Parse experiment configuration
    config = parse_config(args.config_path)

    # Set global random state
    fix_global_random_state(config.getint("hp_optimization", "random_seed"))

    # Create logger
    output_dir = config.get("results", "output_dir")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = create_custom_logger(output_dir)
    logger.info(f"Running program: {__file__}\n")
    logger.info(f"{args}\n")

    printable_config = {section: dict(config[section]) for section in config}
    logger.info(f"Experiment configuration:\n{printable_config}\n")

    # Create prior model wrapper
    prior_model_wrapper = RidgeClassifierTuningWrapper(
        config.get("benchmark", "data_dir"),
        config.getint("hp_optimization", "random_seed"),
        logger,
    )

    if args.mode == "train_w_hp_tuning":
        # Create MRF hyperparameter space
        hp_space = HyperparameterSpace(
            config.get("hp_optimization", "hp_spec_filepath"),
            config.getint("hp_optimization", "random_seed"),
        ).create_hp_space()

        # Hyperparameter tuning
        tuning_engine = PriorModelTuningEngine(
            config, prior_model_wrapper, hp_space
        )
        best_hp_config = tuning_engine.run()

        # Train a model with the best hyperparameter configuration
        prior_model_wrapper.create_model(dict(best_hp_config))
        prior_model_wrapper.fit()

        # Save model
        model_save_filepath = os.path.join(output_dir, "ridge.pkl")
        with open(model_save_filepath, "wb") as f:
            pickle.dump(prior_model_wrapper.clf, f)
    elif args.mode == "train_w_default_hp":
        prior_model_wrapper.create_default_model()
        prior_model_wrapper.fit()
    else:
        assert args.mode == "test", (
            f"Invalid mode: {args.mode}. Mode can only be train_w_default_hp, "
            "train_w_hp_tuning or test."
        )

        model_save_filepath = os.path.join(output_dir, "ridge.pkl")
        with open(model_save_filepath, "rb") as f:
            prior_model_wrapper.clf = pickle.load(f)

    X_train = prior_model_wrapper.X_train
    y_train = prior_model_wrapper.y_train
    X_valid = prior_model_wrapper.X_valid
    y_valid = prior_model_wrapper.y_valid
    X_test = prior_model_wrapper.X_test
    y_test = prior_model_wrapper.y_test

    y_train_pred = prior_model_wrapper.predict(X_train)
    log_exp_metrics("training", y_train, y_train_pred, logger)

    y_valid_pred = prior_model_wrapper.predict(X_valid)
    log_exp_metrics("validation", y_valid, y_valid_pred, logger)

    y_test_pred = prior_model_wrapper.predict(X_test)
    log_exp_metrics("test", y_test, y_test_pred, logger)

    # Add prediction confidence scores
    valid_df = prior_model_wrapper.valid_df
    test_df = prior_model_wrapper.test_df

    y_valid_proba = prior_model_wrapper.predict_proba(X_valid)
    valid_df["positive_label_prediction_probability"] = y_valid_proba[:, 1]

    y_test_proba = prior_model_wrapper.predict_proba(X_test)
    test_df["positive_label_prediction_probability"] = y_test_proba[:, 1]

    log_exp_records(y_valid, y_valid_pred, y_valid_proba, "validation", logger)
    log_exp_records(y_test, y_test_pred, y_test_proba, "test", logger)

    save_dir = os.path.join(config.get("benchmark", "data_dir"), "ridge")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if args.mode == "train_w_default_hp":
        valid_output_filepath = os.path.join(save_dir, "validation_default.csv")
        test_output_filepath = os.path.join(save_dir, "test_default.csv")
    else:
        valid_output_filepath = os.path.join(save_dir, "validation.csv")
        test_output_filepath = os.path.join(save_dir, "test.csv")

    valid_df.to_csv(valid_output_filepath, index=False)
    test_df.to_csv(test_output_filepath, index=False)
