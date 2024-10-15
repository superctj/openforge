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


def create_prompts_per_split(
    input_df: pd.DataFrame,
    few_shot_df: pd.DataFrame,
    data_dir: str,
    split: str,
    num_shots: int,
):
    all_prompts = []
    all_labels = []
    all_rv_names = []

    for i, row in input_df.iterrows():
        prompt = craft_sotab_user_prompt(row, few_shot_df, root_dir=data_dir)

        all_prompts.append(prompt)
        all_labels.append(row["relation_variable_label"])
        all_rv_names.append(row["relation_variable_name"])

        if i == 0:
            logger.info(f"1st prompt:\n{prompt}")

    output_df = pd.DataFrame(
        {
            "prompt": all_prompts,
            "label": all_labels,
            "random_variable_name": all_rv_names,
        }
    )

    if num_shots == 0:
        output_df.to_json(
            os.path.join(output_dir, f"{split}_{num_shots}-shot.json"),
            orient="records",
            indent=4,
        )
    else:
        output_df.to_json(
            os.path.join(output_dir, f"{split}_{num_shots}-shots.json"),
            orient="records",
            indent=4,
        )


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

    data_dir = config.get("benchmark", "data_dir")
    train_df, valid_df, test_df = load_openforge_sotab_benchmark(
        os.path.join(data_dir, "artifact"), logger
    )

    random_seed = config.getint("benchmark", "random_seed")
    num_shots = config.getint("benchmark", "num_shots")

    few_shot_df = sample_few_shot_examples(
        train_df, n=num_shots, balanced=True, random_seed=random_seed
    )

    create_prompts_per_split(
        valid_df, few_shot_df, data_dir, split="valid", num_shots=num_shots
    )

    create_prompts_per_split(
        test_df, few_shot_df, data_dir, split="test", num_shots=num_shots
    )
