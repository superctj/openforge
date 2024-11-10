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
    all_l_ids = []
    all_r_ids = []
    all_l_entities = []
    all_r_entities = []
    all_labels = []

    for _, row in label_df.iterrows():
        l_id = row["ltable_id"]
        r_id = row["rtable_id"]
        label = row["label"]

        l_row = l_df[l_df["id"] == l_id].iloc[0]
        r_row = r_df[r_df["id"] == r_id].iloc[0]

        l_entity = f"The product title is {l_row['title']}, the product category is {l_row['category']}, the product brand is {l_row['brand']}, the product model number is {l_row['modelno']}, and the product price is {l_row['price']}"  # noqa: E501
        r_entity = f"The product title is {r_row['title']}, the product category is {r_row['category']}, the product brand is {r_row['brand']}, the product model number is {r_row['modelno']}, and the product price is {r_row['price']}"  # noqa: E501

        all_l_ids.append(l_id)
        all_r_ids.append(r_id)
        all_l_entities.append(l_entity)
        all_r_entities.append(r_entity)
        all_labels.append(label)

    df = pd.DataFrame(
        {
            "l_id": all_l_ids,
            "r_id": all_r_ids,
            "l_entity": all_l_entities,
            "r_entity": all_r_entities,
            "label": all_labels,
        }
    )

    df.to_json(
        os.path.join(output_dir, f"preprocessed_{split}.json"),
        orient="records",
        indent=4,
    )

    return df


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

    l_table, r_table, train_label_df, valid_label_df, test_label_df = (
        load_walmart_amazon_datasets(config.get("io", "input_dir"))
    )

    preprocess_walmart_amazon_dataset(
        l_table, r_table, train_label_df, output_dir, "training"
    )
    preprocess_walmart_amazon_dataset(
        l_table, r_table, valid_label_df, output_dir, "validation"
    )
    preprocess_walmart_amazon_dataset(
        l_table, r_table, test_label_df, output_dir, "test"
    )
