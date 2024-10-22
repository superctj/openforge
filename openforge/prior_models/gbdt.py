import argparse
import logging
import os
import pickle

import numpy as np

from sklearn.ensemble import GradientBoostingClassifier

from openforge.hp_optimization.hp_space import HyperparameterSpace
from openforge.hp_optimization.tuning import PriorModelTuningEngine
from openforge.utils.custom_logging import create_custom_logger
from openforge.utils.prior_model_common import (
    load_entity_matching_features_and_labels,
    load_sotab_features_and_labels,
    log_exp_metrics,
    log_exp_records,
)
from openforge.utils.util import fix_global_random_state, parse_config


MODEL_NAME = "gbdt"


class GBDTTuningWrapper:
    def __init__(
        self,
        task: str,
        raw_data_dir: str,
        feature_vectors_dir: str,
        random_seed: int,
        logger: logging.Logger,
    ):
        if task == "sotab_v2":
            (
                self.X_train,
                self.y_train,
                self.X_test,
                self.y_test,
                self.train_df,
                self.test_df,
            ) = load_sotab_features_and_labels(
                raw_data_dir, feature_vectors_dir, logger
            )
        elif task == "entity-matching_walmart-amazon":
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
            ) = load_entity_matching_features_and_labels(
                raw_data_dir, feature_vectors_dir, logger
            )
        else:
            raise ValueError(f"Unsupported task: {task}.")

        self.clf = None
        self.random_seed = random_seed

    def create_default_model(self):
        self.clf = GradientBoostingClassifier(
            random_state=self.random_seed,
        )

    def create_model(self, hp_config: dict):
        self.clf = GradientBoostingClassifier(
            learning_rate=hp_config["learning_rate"],
            n_estimators=hp_config["n_estimators"],
            subsample=hp_config["subsample"],
            min_weight_fraction_leaf=hp_config["min_weight_fraction_leaf"],
            max_depth=hp_config["max_depth"],
            random_state=self.random_seed,
        )

    def fit(self):
        self.clf.fit(self.X_train, self.y_train)

    def predict(self, X: np.ndarray, threshold: float = 0.5):
        y_pred = (self.clf.predict_proba(X)[:, 1] >= threshold).astype(int)

        return y_pred

    def predict_proba(self, X: np.ndarray):
        return self.clf.predict_proba(X)


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
    output_dir = config.get("io", "output_dir")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = create_custom_logger(output_dir)
    logger.info(f"Running program: {__file__}\n")
    logger.info(f"{args}\n")

    printable_config = {section: dict(config[section]) for section in config}
    logger.info(f"Experiment configuration:\n{printable_config}\n")

    # Create prior model wrapper
    task = config.get("io", "task")
    prior_model_wrapper = GBDTTuningWrapper(
        task,
        config.get("io", "raw_data_dir"),
        config.get("io", "feature_vectors_dir"),
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
        model_save_filepath = os.path.join(output_dir, f"{MODEL_NAME}.pkl")
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

        model_save_filepath = os.path.join(output_dir, f"{MODEL_NAME}.pkl")
        with open(model_save_filepath, "rb") as f:
            prior_model_wrapper.clf = pickle.load(f)

    X_train = prior_model_wrapper.X_train
    y_train = prior_model_wrapper.y_train
    if task != "sotab_v2":
        X_valid = prior_model_wrapper.X_valid
        y_valid = prior_model_wrapper.y_valid
    X_test = prior_model_wrapper.X_test
    y_test = prior_model_wrapper.y_test

    y_train_pred = prior_model_wrapper.predict(X_train)
    log_exp_metrics("training", y_train, y_train_pred, logger)

    if task != "sotab_v2":
        y_valid_pred = prior_model_wrapper.predict(X_valid)
        log_exp_metrics("validation", y_valid, y_valid_pred, logger)

    y_test_pred = prior_model_wrapper.predict(X_test)
    log_exp_metrics("test", y_test, y_test_pred, logger)

    # Add prediction confidence scores
    train_df = prior_model_wrapper.train_df
    test_df = prior_model_wrapper.test_df

    y_train_proba = prior_model_wrapper.predict_proba(X_train)
    train_df["confidence_score"] = np.max(y_train_proba, axis=1)
    train_df["prediction"] = y_train_pred

    if task != "sotab_v2":
        valid_df = prior_model_wrapper.valid_df
        y_valid_proba = prior_model_wrapper.predict_proba(X_valid)
        valid_df["confidence_score"] = np.max(y_valid_proba, axis=1)
        valid_df["prediction"] = y_valid_pred

    y_test_proba = prior_model_wrapper.predict_proba(X_test)
    test_df["confidence_score"] = np.max(y_test_proba, axis=1)
    test_df["prediction"] = y_test_pred

    log_exp_records(y_train, y_train_pred, y_train_proba, "training", logger)
    if task != "sotab_v2":
        log_exp_records(
            y_valid, y_valid_pred, y_valid_proba, "validation", logger
        )
    log_exp_records(y_test, y_test_pred, y_test_proba, "test", logger)

    save_dir = os.path.join(config.get("io", "output_dir"), f"{MODEL_NAME}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if args.mode == "train_w_default_hp":
        train_output_filepath = os.path.join(save_dir, "training_default.csv")
        if task != "sotab_v2":
            valid_output_filepath = os.path.join(
                save_dir, "validation_default.csv"
            )
        test_output_filepath = os.path.join(save_dir, "test_default.csv")
    else:
        train_output_filepath = os.path.join(save_dir, "training.csv")
        if task != "sotab_v2":
            valid_output_filepath = os.path.join(save_dir, "validation.csv")
        test_output_filepath = os.path.join(save_dir, "test.csv")

    if task == "sotab_v2":
        train_df.rename(
            columns={"relation_variable_name": "random_variable_name"},
            inplace=True,
        )
        test_df.rename(
            columns={"relation_variable_name": "random_variable_name"},
            inplace=True,
        )

    train_df.to_csv(train_output_filepath, index=False)
    if task != "sotab_v2":
        valid_df.to_csv(valid_output_filepath, index=False)
    test_df.to_csv(test_output_filepath, index=False)
