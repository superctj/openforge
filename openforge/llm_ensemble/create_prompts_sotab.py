import argparse
import os

import pandas as pd

from openforge.utils.custom_logging import create_custom_logger
from openforge.utils.llm_common import (
    craft_sotab_user_prompt,
    load_openforge_sotab_benchmark,
    sample_few_shot_examples,
)
from openforge.utils.util import parse_config


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
    output_dir = config.get("results", "output_dir")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = create_custom_logger(output_dir)
    logger.info(f"Running program: {__file__}\n")
    logger.info(f"{args}\n")

    printable_config = {section: dict(config[section]) for section in config}
    logger.info(f"Experiment configuration:\n{printable_config}\n")

    train_df, valid_df, test_df = load_openforge_sotab_benchmark(
        config.get("benchmark", "data_dir"), logger
    )

    random_seed = config.getint("benchmark", "random_seed")
    num_shots = config.getint("benchmark", "num_shots")

    all_prompts = []
    all_labels = []

    for i, row in test_df.iterrows():
        few_shot_df = sample_few_shot_examples(
            train_df, n=num_shots, balanced=True, random_seed=random_seed
        )
        prompt = craft_sotab_user_prompt(row, few_shot_df)

        all_prompts.append(prompt)
        all_labels.append(row["relation_variable_label"])

        if i == 0:
            logger.info(f"1st prompt:\n{prompt}")

    df = pd.DataFrame({"prompt": all_prompts, "label": all_labels})
    df.to_json(
        os.path.join(
            output_dir, f"sotab_test-w_sample_values-{num_shots}_shots.json"
        ),
        orient="records",
        lines=True,
    )
