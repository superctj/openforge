import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# from transformers import AutoModel
from sentence_transformers import SentenceTransformer

from openforge.utils.custom_logging import create_custom_logger
from openforge.utils.llm_common import (
    load_openforge_sotab_benchmark,
    sample_column_values,
)
from openforge.utils.util import parse_config


SOTAB_INSTRUCTION = "Instruct: Classify a given pair of semantic column types (associated with sample values) as either equivalent or non-equivalent\nQuery: "  # noqa: E501


def generate_embeddings_per_split(
    model,
    data_dir: str,
    input_df: pd.DataFrame,
    batch_size: int,
    max_length: int,
    device: torch.device,
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
                batch_embeddings = model.encode(
                    batch_inputs,
                    batch_size=batch_size,
                    prompt=SOTAB_INSTRUCTION,
                    device=device,
                )
                all_embeddings.append(batch_embeddings)

                if i == 32:
                    logger.info(batch_inputs)
                    logger.info(batch_embeddings)
                
                batch_inputs = []

        type_1_sample_values = ", ".join(
            sample_column_values(label_1_table_path, row["label_1_col_idx"])
        )
        type_2_sample_values = ", ".join(
            sample_column_values(label_2_table_path, row["label_2_col_idx"])
        )
        batch_inputs.append(
            f"Type 1 '{row['label_1_processed']}' has sample values: {type_1_sample_values}; Type 2 '{row['label_2_processed']}' has sample values: {type_2_sample_values}.{model.tokenizer.eos_token}"  # noqa: E501
        )

    if len(batch_inputs) > 0:
        with torch.no_grad():
            batch_embeddings = model.encode(
                batch_inputs,
                batch_size=batch_size,
                prompt=SOTAB_INSTRUCTION,
                device=device,
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

    model_id = config.get("encoding", "model_id")
    max_length = config.getint("encoding", "max_length")
    batch_size = config.getint("encoding", "batch_size")

    # model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    model = SentenceTransformer(
        model_id,
        trust_remote_code=True,
        model_kwargs={"torch_dtype": "bfloat16"},
    )
    model.max_seq_length = max_length
    model.tokenizer.padding_side = "right"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    data_dir = config.get("io", "data_dir")
    # Training and validation set are the same for SOTAB-v2 dataset
    train_df, _, test_df = load_openforge_sotab_benchmark(
        os.path.join(data_dir, "artifact"), logger
    )

    generate_embeddings_per_split(
        model,
        data_dir,
        train_df,
        batch_size,
        max_length,
        device,
        split="training",
        output_dir=output_dir,
        logger=logger,
    )

    generate_embeddings_per_split(
        model,
        data_dir,
        test_df,
        batch_size,
        max_length,
        device,
        split="test",
        output_dir=output_dir,
        logger=logger,
    )
