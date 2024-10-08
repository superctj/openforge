import argparse
import os

import numpy as np
import pandas as pd
import torch

from transformers import pipeline

from openforge.utils.custom_logging import create_custom_logger
from openforge.utils.prior_model_common import log_exp_metrics
from openforge.utils.util import parse_config


HYPOTHESIS_TEMPLATE = "The two semantic types are semantically {}"
CLASS_LABELS = [
    "non-equivalent",
    "equivalent",
]


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
        default="inference",
        help="Mode of operation. 'inference' will incur model API to obtain predictions; 'test' will incur model API up to 3 times for testing purpose; 'evaluation' will evaluate existing predictions.",  # noqa: E501
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

    data_filepath = config.get("exp", "data_filepath")
    df = pd.read_json(data_filepath)
    logger.info(f"\n{df.head()}\n")

    model_id = config.get("llm", "model_id")
    # temperature = config.getfloat("llm", "temperature")

    if args.mode == "inference" or args.mode == "test":
        device = "cuda" if torch.cuda.is_available() else "cpu"

        clf = pipeline(
            "zero-shot-classification", model=model_id, device=device
        )

        all_predictions = []
        all_confdc_scores = []
        all_labels = []

        for i, row in df.iterrows():
            logger.info(f"{i+1}/{df.shape[0]}:")

            prompt = row["prompt"]
            outputs = clf(
                prompt,
                CLASS_LABELS,
                hypothesis_template=HYPOTHESIS_TEMPLATE,
                multi_label=False,
            )
            pred = np.argmax(outputs["scores"])
            confdc_score = outputs["scores"][pred]

            all_predictions.append(pred)
            all_confdc_scores.append(confdc_score)
            all_labels.append(int(row["label"]))

            logger.info(
                f"prediction={pred} with confidence score={confdc_score}"
            )
            logger.info(f"label={row['label']}")
            logger.info("-" * 80)

            if args.mode == "test" and i >= 2:
                break

        df["prediction"] = all_predictions
        output_filename = data_filepath.split("/")[-1].split(".")[0]
        output_filepath = os.path.join(output_dir, f"{output_filename}.json")
        df.to_json(output_filepath, orient="records", indent=4)
    else:
        assert args.mode == "evaluation"

        all_predictions = df["prediction"].tolist()
        all_labels = df["label"].tolist()

    log_exp_metrics(
        output_filename, all_labels, all_predictions, logger, multi_class=False
    )
