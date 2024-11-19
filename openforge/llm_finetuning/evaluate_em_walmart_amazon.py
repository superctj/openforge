import argparse
import os

import pandas as pd
import torch

from datasets import Dataset
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from openforge.utils.custom_logging import create_custom_logger
from openforge.utils.llm_common import (
    ID2LABEL,
    LABEL2ID,
    encode_em_walmart_amazon_input,
)
from openforge.utils.prior_model_common import log_exp_metrics
from openforge.utils.util import parse_config


def get_predictions(
    model,
    tokenizer,
    data_collator,
    input_df,
    batch_size,
    temperature,
    output_dir,
    split,
    device,
    logger,
):
    input_dataset = Dataset.from_pandas(input_df)
    tokenized_input_dataset = input_dataset.map(
        encode_em_walmart_amazon_input,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=[
            "l_id",
            "r_id",
            "l_entity",
            "r_entity",
        ],  # A list of columns to remove after applying the function
    )

    # Rename the label column to labels because the model expects the argument
    # to be named labels
    tokenized_input_dataset = tokenized_input_dataset.rename_column(
        "label", "labels"
    )

    # Set the format of the dataset to return PyTorch tensors instead
    # of lists
    tokenized_input_dataset.set_format(
        "torch", columns=["input_ids", "attention_mask", "labels"]
    )

    input_dataloader = DataLoader(
        tokenized_input_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    preds = []
    confdc_scores = []

    for batch in input_dataloader:
        inputs = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits / temperature
        batch_preds = logits.argmax(dim=-1)
        batch_confdc_scores = (
            torch.nn.functional.softmax(logits, dim=-1).max(dim=-1).values
        )
        logger.info(f"batch_preds: {batch_preds}")
        logger.info(f"batch_confdc_scores: {batch_confdc_scores}")

        preds.extend(batch_preds.tolist())
        confdc_scores.extend(batch_confdc_scores.tolist())

    input_df["prediction"] = preds
    input_df["confidence_score"] = confdc_scores

    labels = input_df["label"].tolist()
    log_exp_metrics(f"{split}", labels, preds, logger, multi_class=False)

    # output_filepath = os.path.join(output_dir, f"{split}.json")
    # input_df.to_json(output_filepath, orient="records", indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the config file",
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

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["llm"]["checkpoint_dir"])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(
        config["llm"]["checkpoint_dir"],
        num_labels=len(ID2LABEL),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    if "qwen2.5-7b-instruct_lora" in config["llm"]["checkpoint_dir"]:
        model.config.pad_token_id = tokenizer.eos_token_id

    model = PeftModel.from_pretrained(model, config["llm"]["checkpoint_dir"])
    model = model.to(device)
    model.eval()

    # for name, param in model.named_parameters():
    #     logger.info(f"Module name: {name}")

    input_dir = config.get("io", "input_dir")
    batch_size = config.getint("llm", "batch_size")
    temperature = config.getfloat("llm", "temperature")

    valid_filepath = os.path.join(input_dir, "preprocessed_validation.json")
    valid_df = pd.read_json(valid_filepath, orient="records")
    get_predictions(
        model,
        tokenizer,
        data_collator,
        valid_df,
        batch_size,
        temperature,
        output_dir,
        split="validation",
        device=device,
        logger=logger,
    )

    test_filepath = os.path.join(input_dir, "preprocessed_test.json")
    test_df = pd.read_json(test_filepath, orient="records")
    get_predictions(
        model,
        tokenizer,
        data_collator,
        test_df,
        batch_size,
        temperature,
        output_dir,
        split="test",
        device=device,
        logger=logger,
    )