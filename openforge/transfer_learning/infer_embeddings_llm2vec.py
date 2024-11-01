import argparse
import os

import numpy as np
import pandas as pd
import torch

from llm2vec import LLM2Vec

from openforge.utils.custom_logging import create_custom_logger
from openforge.utils.llm_common import (
    load_openforge_sotab_benchmark,
    sample_column_values,
)
from openforge.utils.util import parse_config


SOTAB_INSTRUCTION = "Classify a given pair of semantic column types (associated with sample values) as either equivalent or non-equivalent: "  # noqa: E501

ENTITY_MATCHING_INSTRUCTION = "Classify a given pair of entities as either equivalent or non-equivalent: "  # noqa: E501


def generate_embeddings_per_split_for_sotab_v2(
    model,
    data_dir: str,
    input_df: pd.DataFrame,
    batch_size: int,
    split: str,
    output_dir: str,
    logger=None,
):
    batch_inputs = []
    all_embeddings = []

    for i, row in input_df.iterrows():
        if data_dir:
            label_1_table_path = row["label_1_table_path"].replace(
                "/ssd/congtj/openforge/sotab_v2", data_dir
            )
            label_2_table_path = row["label_2_table_path"].replace(
                "/ssd/congtj/openforge/sotab_v2", data_dir
            )

        if i != 0 and i % batch_size == 0:
            with torch.no_grad():
                batch_embeddings = (
                    model.encode(batch_inputs, batch_size=batch_size)
                    .detach()
                    .cpu()
                    .numpy()
                )
                all_embeddings.append(batch_embeddings)

                if i == batch_size:
                    logger.info(batch_inputs)
                    logger.info(batch_embeddings)

                batch_inputs = []

        type_1_sample_values = sample_column_values(
            label_1_table_path, row["label_1_col_idx"]
        )
        type_2_sample_values = sample_column_values(
            label_2_table_path, row["label_2_col_idx"]
        )
        batch_inputs.append(
            [
                SOTAB_INSTRUCTION,
                f"Type 1 '{row['label_1_processed']}' has sample values: {type_1_sample_values}; Type 2 '{row['label_2_processed']}' has sample values: {type_2_sample_values}.",  # noqa: E501
            ]
        )

    if len(batch_inputs) > 0:
        with torch.no_grad():
            batch_embeddings = model.encode(batch_inputs).detach().cpu().numpy()
        all_embeddings.append(batch_embeddings)

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    assert all_embeddings.shape[0] == input_df.shape[0]

    output_filepath = os.path.join(output_dir, f"{split}_embeddings.npy")
    np.save(output_filepath, all_embeddings)


def generate_embeddings_per_split_for_entity_matching(
    model,
    input_df: pd.DataFrame,
    batch_size: int,
    split: str,
    output_dir: str,
    logger=None,
):
    """
    Generate embeddings per split for entity matching task.
    """

    batch_inputs = []
    all_embeddings = []

    for i, row in input_df.iterrows():
        if i != 0 and i % batch_size == 0:
            with torch.no_grad():
                batch_embeddings = (
                    model.encode(
                        batch_inputs,
                        batch_size=batch_size,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
                all_embeddings.append(batch_embeddings)

                if i == batch_size:
                    logger.info(batch_inputs)
                    logger.info(batch_embeddings)

                batch_inputs = []

        batch_inputs.append(
            [
                ENTITY_MATCHING_INSTRUCTION,
                f"Entity 1: {row['entity_1']}; Entity 2: {row['entity_2']}.",
            ]
        )

    if len(batch_inputs) > 0:
        with torch.no_grad():
            batch_embeddings = (
                model.encode(batch_inputs, batch_size=batch_size)
                .detach()
                .cpu()
                .numpy()
            )
        all_embeddings.append(batch_embeddings)

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    assert all_embeddings.shape[0] == input_df.shape[0]

    output_filepath = os.path.join(output_dir, f"{split}_embeddings.npy")
    np.save(output_filepath, all_embeddings)


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

    task = config.get("encoding", "task")
    base_model_id = config.get("encoding", "base_model_id")
    lora_model_id = config.get("encoding", "lora_model_id")
    pooling_mode = config.get("encoding", "pooling_mode")
    max_length = config.getint("encoding", "max_length")
    batch_size = config.getint("encoding", "batch_size")
    data_dir = config.get("io", "data_dir")

    model = LLM2Vec.from_pretrained(
        base_model_id,
        peft_model_name_or_path=lora_model_id,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.bfloat16,
        pooling_mode=pooling_mode,
        max_length=max_length,
    )

    if task == "sotab-v2":
        train_df, _, test_df = load_openforge_sotab_benchmark(
            os.path.join(data_dir, "artifact"), logger
        )

        generate_embeddings_per_split_for_sotab_v2(
            model,
            data_dir,
            train_df,
            batch_size,
            split="training",
            output_dir=output_dir,
            logger=logger,
        )

        generate_embeddings_per_split_for_sotab_v2(
            model,
            data_dir,
            test_df,
            batch_size,
            split="test",
            output_dir=output_dir,
            logger=logger,
        )
    elif task == "entity-matching_walmart-amazon":
        train_df = pd.read_json(
            os.path.join(data_dir, "preprocessed_train.json")
        )
        valid_df = pd.read_json(
            os.path.join(data_dir, "preprocessed_valid.json")
        )
        test_df = pd.read_json(os.path.join(data_dir, "preprocessed_test.json"))

        generate_embeddings_per_split_for_entity_matching(
            model,
            train_df,
            batch_size,
            split="training",
            output_dir=output_dir,
            logger=logger,
        )

        generate_embeddings_per_split_for_entity_matching(
            model,
            valid_df,
            batch_size,
            split="validation",
            output_dir=output_dir,
            logger=logger,
        )

        generate_embeddings_per_split_for_entity_matching(
            model,
            test_df,
            batch_size,
            split="test",
            output_dir=output_dir,
            logger=logger,
        )
    else:
        raise ValueError(f"Unsupported task: {task}")
