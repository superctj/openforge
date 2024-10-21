import argparse
import os

import numpy as np
import pandas as pd
import torch

from llm2vec import LLM2Vec

# from peft import PeftModel
# from transformers import AutoTokenizer, AutoModel, AutoConfig

from openforge.utils.custom_logging import create_custom_logger
from openforge.utils.llm_common import (
    load_openforge_sotab_benchmark,
    sample_column_values,
)
from openforge.utils.util import parse_config

# CONDA_ENV_PREFIX = os.environ.get("CONDA_PREFIX", "")
# os.environ["LD_LIBRARY_PATH"] = os.path.join(
#     CONDA_ENV_PREFIX, "lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib"
# )
# print(os.environ["LD_LIBRARY_PATH"])

SOTAB_INSTRUCTION = "Determine if two semantic column types are equivalent based on whether their sample values are from the same domain: "  # noqa: E501


def generate_embeddings_per_split(
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
            batch_embeddings = (
                model.encode(batch_inputs, batch_size=batch_size)
                .detach()
                .cpu()
                .numpy()
            )
            all_embeddings.append(batch_embeddings)
            logger.info(batch_inputs)
            logger.info(batch_embeddings)
            batch_inputs = []

            if i % 32 == 0:
                exit(0)

        type_1_sample_values = sample_column_values(
            label_1_table_path, row["label_1_col_idx"]
        )
        type_2_sample_values = sample_column_values(
            label_2_table_path, row["label_2_col_idx"]
        )
        # batch_inputs.append(
        #     [
        #         SOTAB_INSTRUCTION,
        #         f"Type 1 {row['label_1_processed']} has sample values {', '.join(type_1_sample_values)}. Type 2 {row['label_2_processed']} has sample values {', '.join(type_2_sample_values)}.",  # noqa: E501
        #     ]
        # )
        batch_inputs.append(
            SOTAB_INSTRUCTION
            + f"Type 1 '{row['label_1_processed']}' has sample values {type_1_sample_values}. Type 2 '{row['label_2_processed']}' has sample values {type_2_sample_values}."  # noqa: E501
        )

    if len(batch_inputs) > 0:
        batch_embeddings = model.encode(batch_inputs).detach().cpu().numpy()
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

    base_model_id = config.get("encoding", "base_model_id")
    lora_model_id = config.get("encoding", "lora_model_id")
    pooling_mode = config.get("encoding", "pooling_mode")
    max_length = config.getint("encoding", "max_length")
    batch_size = config.getint("encoding", "batch_size")

    # tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    # config = AutoConfig.from_pretrained(base_model_id, trust_remote_code=True)
    # model = AutoModel.from_pretrained(
    #     base_model_id,
    #     trust_remote_code=True,
    #     config=config,
    #     torch_dtype=torch.bfloat16,
    #     device_map="cuda" if torch.cuda.is_available() else "cpu",
    # )

    # model = PeftModel.from_pretrained(
    #     model,
    #     base_model_id,
    # )
    # model = model.merge_and_unload()  # This can take several minutes on cpu

    # # Loading supervised model. This loads the trained LoRA weights on top of MNTP model. Hence the final weights are -- Base model + MNTP (LoRA) + supervised (LoRA). # noqa: E501
    # model = PeftModel.from_pretrained(model, lora_model_id)

    # # Wrapper for encoding and pooling operations
    # l2v = LLM2Vec(
    #     model, tokenizer, pooling_mode=pooling_mode, max_length=max_length
    # )
    model = LLM2Vec.from_pretrained(
        base_model_id,
        peft_model_name_or_path=lora_model_id,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.bfloat16,
        pooling_mode=pooling_mode,
        max_length=max_length,
    )

    data_dir = config.get("io", "data_dir")
    _, valid_df, test_df = load_openforge_sotab_benchmark(
        os.path.join(data_dir, "artifact"), logger
    )

    generate_embeddings_per_split(
        model,
        data_dir,
        valid_df,
        batch_size,
        split="valid",
        output_dir=output_dir,
        logger=logger,
    )

    generate_embeddings_per_split(
        model,
        data_dir,
        test_df,
        batch_size,
        split="test",
        output_dir=output_dir,
        logger=logger,
    )
