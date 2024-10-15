import argparse
import os

import pandas as pd

from openforge.utils.custom_logging import create_custom_logger
from openforge.utils.util import parse_config


def load_walmart_amazon_datasets(data_dir: str) -> pd.DataFrame:
    l_table = pd.read_csv(os.path.join(data_dir, "tableA.csv"))
    r_table = pd.read_csv(os.path.join(data_dir, "tableB.csv"))

    train_label_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    valid_label_df = pd.read_csv(os.path.join(data_dir, "valid.csv"))
    test_label_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

    return l_table, r_table, train_label_df, valid_label_df, test_label_df


def preprocess_walmart_amazon_dataset(
    l_df: pd.DataFrame,
    r_df: pd.DataFrame,
    label_df,
    output_dir: str,
    split: str,
) -> pd.DataFrame:
    entity_1_descriptions = []
    entity_2_descriptions = []
    all_labels = []

    for _, row in label_df.iterrows():
        l_id = row["ltable_id"]
        r_id = row["rtable_id"]
        label = row["label"]

        l_row = l_df[l_df["id"] == l_id].iloc[0]
        r_row = r_df[r_df["id"] == r_id].iloc[0]

        l_description = f"The product title is {l_row['title']}, the product category is {l_row['category']}, the product brand is {l_row['brand']}, the product model number is {l_row['modelno']}, and the product price is {l_row['price']}"  # noqa: E501
        r_description = f"The product title is {r_row['title']}, the product category is {r_row['category']}, the product brand is {r_row['brand']}, the product model number is {r_row['modelno']}, and the product price is {r_row['price']}"  # noqa: E501

        entity_1_descriptions.append(l_description)
        entity_2_descriptions.append(r_description)
        all_labels.append(label)

    df = pd.DataFrame(
        {
            "entity_1": entity_1_descriptions,
            "entity_2": entity_2_descriptions,
            "label": all_labels,
        }
    )

    df.to_json(
        os.path.join(output_dir, f"preprocessed_{split}.json"),
        orient="records",
        indent=4,
    )

    return df


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
    row: pd.Series, few_shot_df: pd.DataFrame
) -> str:
    prompt = """Entity matching is the task of determining whether two data instances refer to the same real-world entity."""  # noqa: E501

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

Now, for the following pair of instances, please determine if they refer to the same real-world entity. Return your prediction and confidence score in the following JSON format: '{{"equivalent": true, "confidence score":}}' or '{{"equivalent": false, "confidence score":}}'. Confidence score needs to be greater than 0.5 and smaller than 1.

Input:
Instance 1: {}
Instance 2: {}

Output:
""".format(  # noqa: E501
        row["entity_1"],
        row["entity_2"],
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

    l_table, r_table, train_label_df, valid_label_df, test_label_df = (
        load_walmart_amazon_datasets(config.get("benchmark", "data_dir"))
    )

    train_df = preprocess_walmart_amazon_dataset(
        l_table, r_table, train_label_df, output_dir, "train"
    )
    valid_df = preprocess_walmart_amazon_dataset(
        l_table, r_table, valid_label_df, output_dir, "valid"
    )
    test_df = preprocess_walmart_amazon_dataset(
        l_table, r_table, test_label_df, output_dir, "test"
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
            os.path.join(output_dir, f"test_{num_shots}-shot.json"),
            orient="records",
            indent=4,
        )
    else:
        df.to_json(
            os.path.join(output_dir, f"test_{num_shots}-shots.json"),
            orient="records",
            indent=4,
        )
