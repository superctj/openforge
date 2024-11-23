import argparse
import os

import pandas as pd

from openforge.utils.custom_logging import create_custom_logger
from openforge.utils.llm_common import (
    craft_icpsr_user_prompt,
    load_icpsr_dataset,
)
from openforge.utils.util import parse_config


def sample_few_shot_examples(
    df: pd.DataFrame, n: int = 10, balanced: bool = True, random_seed: int = 42
) -> pd.DataFrame:
    if n == 0:
        return pd.DataFrame()

    if not balanced:
        return df.sample(n, random_state=random_seed)

    sample_df = pd.concat(
        [
            df[df["relation_variable_label"] == 1].sample(
                n // 2, random_state=random_seed
            ),
            df[df["relation_variable_label"] == 0].sample(
                n - n // 2, random_state=random_seed
            ),
        ]
    )
    sample_df = sample_df.sample(frac=1)  # random shuffle

    return sample_df[
        [
            "concept_1",
            "concept_2",
            "relation_variable_label",
        ]
    ]


def create_prompts_per_split(
    input_df: pd.DataFrame,
    few_shot_df: pd.DataFrame,
    split: str,
    output_dir: str,
):
    all_prompts = []

    for i, row in input_df.iterrows():
        prompt = craft_icpsr_user_prompt(row, few_shot_df)

        all_prompts.append(prompt)

        if i == 0:
            logger.info(f"1st prompt:\n{prompt}")

    input_df["prompt"] = all_prompts

    input_df.to_json(
        os.path.join(output_dir, f"{split}.json"),
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
    output_dir = config.get("io", "output_dir")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = create_custom_logger(output_dir)
    logger.info(f"Running program: {__file__}\n")
    logger.info(f"{args}\n")

    printable_config = {section: dict(config[section]) for section in config}
    logger.info(f"Experiment configuration:\n{printable_config}\n")

    input_dir = config.get("io", "input_dir")
    train_df, valid_df, test_df = load_icpsr_dataset(
        input_dir, rename_columns=False
    )

    random_seed = config.getint("prompts", "random_seed")
    num_shots = config.getint("prompts", "num_shots")

    few_shot_df = sample_few_shot_examples(
        train_df, n=num_shots, balanced=True, random_seed=random_seed
    )

    create_prompts_per_split(
        valid_df,
        few_shot_df,
        split="validation",
        output_dir=output_dir,
    )

    create_prompts_per_split(
        test_df,
        few_shot_df,
        split="test",
        output_dir=output_dir,
    )
