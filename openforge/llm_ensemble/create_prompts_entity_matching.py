import argparse
import os

import pandas as pd

from openforge.utils.custom_logging import create_custom_logger
from openforge.utils.llm_common import load_unicorn_entity_matching_benchmark
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
            df[df["label"] == 1].sample(n // 2, random_state=random_seed),
            df[df["label"] == 0].sample(n - n // 2, random_state=random_seed),
        ]
    )
    sample_df = sample_df.sample(frac=1)  # random shuffle

    return sample_df


def craft_entity_matching_user_prompt(
    test_record: dict, few_shot_df: pd.DataFrame
) -> str:
    prompt = """Entity matching is the task of determining whether two entity descriptions refer to the same real-world entity."""  # noqa: E501

    if few_shot_df is not None and not few_shot_df.empty:
        prompt += """\n\nFor example,\n\n"""

        fewshot_prompt = "\n\n".join(
            [
                "Input:\nEntity description 1: {}\nEntity description 2: {}\n\nOutput:\n{}".format(  # noqa: E501
                    row[0],
                    row[1],
                    '{"match": true}' if row[2] else '{"match": false}',
                )
                for row in few_shot_df.values.tolist()
            ]
        )

        prompt += fewshot_prompt

    prompt += """

Now, for the following pair of entity descriptions, please determine if they refer to the same real-world entity. Return your prediction in the following JSON format: '{{"match": true}}' or '{{"match": false}}'.

Input:
Entity description 1: {}
Entity description 2: {}

Output:
""".format(  # noqa: E501
        test_record["entity_1"],
        test_record["entity_2"],
    )

    return prompt


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

    random_seed = config.getint("benchmark", "random_seed")
    num_shots = config.getint("benchmark", "num_shots")

    train_df, valid_df, test_df = load_unicorn_entity_matching_benchmark(
        config.get("benchmark", "data_dir")
    )

    few_shot_df = sample_few_shot_examples(
        train_df, n=num_shots, balanced=True, random_seed=random_seed
    )

    all_prompts = []
    all_labels = []

    for i, row in test_df.iterrows():
        prompt = craft_entity_matching_user_prompt(row, few_shot_df)

        all_prompts.append(prompt)
        all_labels.append(row["label"])

        if i == 0:
            logger.info(f"1st prompt:\n{prompt}")

    df = pd.DataFrame({"prompt": all_prompts, "label": all_labels})

    if num_shots == 0:
        df.to_json(
            os.path.join(output_dir, f"{num_shots}_shot.json"),
            orient="records",
            indent=4,
        )
    else:
        df.to_json(
            os.path.join(output_dir, f"{num_shots}_shots.json"),
            orient="records",
            indent=4,
        )
