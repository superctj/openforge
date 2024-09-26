import argparse
import json
import os

from itertools import combinations

import numpy as np
import pandas as pd

from openforge.utils.custom_logging import create_custom_logger
from openforge.utils.prior_model_common import log_exp_metrics
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


def prepare_mrf_inputs_for_sotab(
    source_filepath: str,
    output_dir: str,
    preds: list[int],
    pred_confdc_scores: list[float],
):
    df = pd.read_json(source_filepath)
    df["prediction"] = preds
    df["confidence_score"] = pred_confdc_scores

    # Assign random variable names (it is recorded as "relation_variable_name"
    # in the JSON for legacy reasons)
    rv_names = []

    for i, j in combinations(range(1, 47), 2):
        rv_names.append(f"R_{i}-{j}")

    df["relation_variable_name"] = rv_names

    filename = source_filepath.split("/")[-1]
    output_filepath = os.path.join(output_dir, filename)
    df.to_json(output_filepath, orient="records", indent=4)


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

    majority_preds = []
    majority_confdc_scores = []
    n = all_predictions.shape[0]

    for i in range(n):
        # Get the majority vote prediction and confidence score
        majority_vote = np.argmax(np.bincount(all_predictions[i]))
        confdc_score = np.mean(all_predictions[i] == majority_vote)

        majority_preds.append(majority_vote)
        majority_confdc_scores.append(confdc_score)

        logger.info(f"{i+1} / {n}")
        logger.info(f"All predictions: {all_predictions[i]}")
        logger.info(f"Ground truth: {ground_truth[i]}")

        logger.info(f"Majority vote prediction: {majority_vote}")
        logger.info(f"Majority vote confidence score: {confdc_score}")
        logger.info("-" * 80)

    prepare_mrf_inputs_for_sotab(
        source_filepath=os.path.join(root_dir, input_filename),
        output_dir=output_dir,
        preds=majority_preds,
        pred_confdc_scores=majority_confdc_scores,
    )

    # Log experiment metrics
    log_exp_metrics(
        input_filename,
        y=ground_truth,
        y_pred=np.array(majority_preds),
        logger=logger,
    )
