import argparse
import logging
import os
import pickle
import random

from sklearn.svm import SVC

from openforge.hp_optimization.hp_space import HyperparameterSpace
from openforge.hp_optimization.tuning import PriorModelTuningEngine
from openforge.utils.custom_logging import create_custom_logger
from openforge.utils.prior_model_common import (
    load_openforge_sotab_small,
    log_exp_metrics,
)
from openforge.utils.util import fix_global_random_state, parse_config


class LinearSVMTuningWrapper:
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
        self.random_seed = random_seed

    def create_model(self, hp_config: dict):
        clf = SVC(
            kernel="linear",
            C=hp_config["C"],
            probability=True,
            random_state=self.random_seed,
        )
        clf.fit(self.X_train, self.y_train)

        return clf

    def run_inference(self, clf):
        y_valid_pred = clf.predict(self.X_valid)

        return y_valid_pred


def main_without_tuning():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode", type=str, default="train", help="Mode: train or test."
    )

    parser.add_argument(
        "--openforge_sotab_benchmark",
        type=str,
        default="/ssd/congtj/openforge/sotab_v2/artifact/sotab_v2_test_openforge_small",  # noqa: E501
        help="Directory containing the OpenForge-SOTAB benchmark.",
    )

    parser.add_argument(
        "--exp_dir",
        type=str,
        default="/home/congtj/openforge/exps/sotab_v2_test_openforge_small/linear_svm",  # noqa: E501
        help="Directory to save experiment artifacts including model weights and results.",  # noqa: E501
    )

    parser.add_argument(
        "--random_seed", type=int, default=12345, help="Random seed."
    )

    args = parser.parse_args()

    # Fix random seed
    random.seed(args.random_seed)

    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)

    model_save_filepath = os.path.join(args.exp_dir, "linear_svm.pkl")
    output_dir = os.path.join(args.openforge_sotab_benchmark, "linear_svm")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = create_custom_logger(args.exp_dir)
    logger.info(f"Running program: {__file__}\n")
    logger.info(f"{args}\n")

    (
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test,
        train_df,
        valid_df,
        test_df,
    ) = load_openforge_sotab_small(args.openforge_sotab_benchmark, logger)

    if args.mode == "train":
        clf = SVC(
            kernel="linear",
            C=1,
            probability=True,
            random_state=args.random_seed,
        )
        clf.fit(X_train, y_train)

        # Save model
        with open(model_save_filepath, "wb") as f:
            pickle.dump(clf, f)
    else:
        assert args.mode == "test"

        with open(model_save_filepath, "rb") as f:
            clf = pickle.load(f)

    y_train_pred = clf.predict(X_train)
    log_exp_metrics("training", y_train, y_train_pred, logger)

    y_valid_pred = clf.predict(X_valid)
    log_exp_metrics("validation", y_valid, y_valid_pred, logger)

    y_test_pred = clf.predict(X_test)
    log_exp_metrics("test", y_test, y_test_pred, logger)

    # Add prediction confidence scores
    valid_confdc_scores = clf.predict_proba(X_valid)
    valid_df["positive_label_confidence_score"] = valid_confdc_scores[:, 1]

    test_confdc_scores = clf.predict_proba(X_test)
    test_df["positive_label_confidence_score"] = test_confdc_scores[:, 1]

    valid_output_filepath = os.path.join(output_dir, "validation.csv")
    valid_df.to_csv(valid_output_filepath, index=False)

    test_output_filepath = os.path.join(output_dir, "test.csv")
    test_df.to_csv(test_output_filepath, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the experiment configuration file.",
    )

    args = parser.parse_args()

    # Parse experiment configuration
    config = parse_config(args.config_path)

    # Set global random state
    fix_global_random_state(config.getint("hp_optimization", "random_seed"))

    # Create MRF hyperparameter space
    hp_space = HyperparameterSpace(
        config.get("hp_optimization", "hp_spec_filepath"),
        config.getint("hp_optimization", "random_seed"),
    ).create_hp_space()

    # Create logger
    output_dir = config.get("results", "output_dir")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = create_custom_logger(output_dir)
    printable_config = {section: dict(config[section]) for section in config}
    logger.info(f"Experiment configuration:\n{printable_config}\n")

    # Create prior model wrapper
    prior_model_wrapper = LinearSVMTuningWrapper(
        config.get("benchmark", "data_dir"),
        config.getint("hp_optimization", "random_seed"),
        logger,
    )

    # Hyperparameter tuning
    tuning_engine = PriorModelTuningEngine(
        config, prior_model_wrapper, hp_space
    )
    best_hp_config = tuning_engine.run()

    # Train a model with the best hyperparameter configuration
    clf = prior_model_wrapper.create_model(dict(best_hp_config))

    # Save model
    model_save_filepath = os.path.join(output_dir, "linear_svm.pkl")
    with open(model_save_filepath, "wb") as f:
        pickle.dump(clf, f)

    X_train = prior_model_wrapper.X_train
    y_train = prior_model_wrapper.y_train
    X_valid = prior_model_wrapper.X_valid
    y_valid = prior_model_wrapper.y_valid
    X_test = prior_model_wrapper.X_test
    y_test = prior_model_wrapper.y_test
    valid_df = prior_model_wrapper.valid_df
    test_df = prior_model_wrapper.test_df

    y_train_pred = clf.predict(X_train)
    log_exp_metrics("training", y_train, y_train_pred, logger)

    y_valid_pred = clf.predict(X_valid)
    log_exp_metrics("validation", y_valid, y_valid_pred, logger)

    y_test_pred = clf.predict(X_test)
    log_exp_metrics("test", y_test, y_test_pred, logger)

    # Add prediction confidence scores
    valid_confdc_scores = clf.predict_proba(X_valid)
    valid_df["positive_label_confidence_score"] = valid_confdc_scores[:, 1]

    test_confdc_scores = clf.predict_proba(X_test)
    test_df["positive_label_confidence_score"] = test_confdc_scores[:, 1]

    save_dir = os.path.join(config.get("benchmark", "data_dir"), "linear_svm")

    valid_output_filepath = os.path.join(save_dir, "validation.csv")
    valid_df.to_csv(valid_output_filepath, index=False)

    test_output_filepath = os.path.join(save_dir, "test.csv")
    test_df.to_csv(test_output_filepath, index=False)
