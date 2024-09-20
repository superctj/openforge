import argparse
import json
import os

import numpy as np

from snorkel.labeling.model import LabelModel, MajorityLabelVoter

from openforge.utils.custom_logging import create_custom_logger
from openforge.utils.util import parse_config


def load_predictions_and_ground_truth(
    root_dir: str, model_ids: list[str], input_filename: str, logger=None
) -> tuple[np.array, list]:
    all_predictions = []
    ground_truth = []

    for i, mid in enumerate(model_ids):
        preds = []
        input_filepath = os.path.join(root_dir, f"{mid}/{input_filename}")

        with open(input_filepath, "r") as f:
            inputs = json.load(f)

            for item in inputs:
                preds.append(item["prediction"])

                if i == 0:
                    ground_truth.append(int(item["label"]))

        all_predictions.append(preds)

    all_predictions = np.array(all_predictions).T
    ground_truth = np.array(ground_truth)
    assert all_predictions.shape[0] == ground_truth.shape[0]
    assert all_predictions.shape[1] == len(model_ids)

    return all_predictions, ground_truth


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

    # Create logger
    output_dir = config.get("exp", "output_dir")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = create_custom_logger(output_dir)
    logger.info(f"Running program: {__file__}\n")
    logger.info(f"{args}\n")

    printable_config = {section: dict(config[section]) for section in config}
    logger.info(f"Experiment configuration:\n{printable_config}\n")

    # Load predictions from individual models
    root_dir = config.get("exp", "root_dir")
    input_filename = config.get("exp", "input_filename")
    model_ids = [
        x.strip() for x in config.get("ensemble", "model_ids").split(",")
    ]
    all_predictions, ground_truth = load_predictions_and_ground_truth(
        root_dir, model_ids, input_filename, logger
    )

    majority_model = MajorityLabelVoter()
    majority_scores = majority_model.score(
        L=all_predictions, Y=ground_truth, metrics=["f1", "precision", "recall"]
    )

    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(
        L_train=all_predictions,
        n_epochs=config.getint("ensemble", "num_epochs"),
        log_freq=100,
        seed=config.getint("exp", "random_seed"),
    )
    label_model_scores = label_model.score(
        L=all_predictions, Y=ground_truth, metrics=["f1", "precision", "recall"]
    )

    logger.info(f"Majority voting scores:\n{majority_scores}")
    logger.info("-" * 80)
    logger.info(f"Label model scores:\n{label_model_scores}")
