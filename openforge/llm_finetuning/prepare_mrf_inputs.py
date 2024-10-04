import argparse
import os

import torch

from datasets import Dataset
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
    encode_entity_matching_input,
    load_unicorn_entity_matching_benchmark,
)
from openforge.utils.prior_model_common import log_exp_metrics
from openforge.utils.util import parse_config


def prepare_mrf_inputs_for_entity_matching(
    source_df: str,
    preds: list[int],
    pred_confdc_scores: list[float],
    output_dir: str,
    split: str,
):
    eid = 0
    entity_id_map = {}
    rv_names = []

    for _, row in source_df.iterrows():
        e1 = row["entity_1"]

        if e1 not in entity_id_map:
            entity_id_map[e1] = eid
            eid += 1

    for _, row in source_df.iterrows():
        e2 = row["entity_2"]

        if e2 not in entity_id_map:
            entity_id_map[e2] = eid
            eid += 1

        e1 = row["entity_1"]
        rv_names.append(f"R_{entity_id_map[e1]}-{entity_id_map[e2]}")

    source_df["random_variable_name"] = rv_names
    source_df["prediction"] = preds
    source_df["confidence_score"] = pred_confdc_scores

    # Save the MRF inputs
    output_filepath = os.path.join(output_dir, f"mrf_{split}_split_inputs.json")
    source_df.to_json(output_filepath, orient="records", indent=4)


def get_predictions(model, dataloader, device):
    preds = []
    confdc_scores = []
    labels = []

    for batch in dataloader:
        inputs = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        batch_preds = logits.argmax(dim=-1)

        preds.extend(batch_preds.tolist())
        confdc_scores.extend(
            torch.nn.functional.softmax(logits, dim=-1)
            .max(dim=-1)
            .values.tolist()
        )
        labels.extend(inputs["labels"].tolist())

    return preds, confdc_scores, labels


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
    output_dir = config.get("exp", "output_dir")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = create_custom_logger(output_dir)
    logger.info(f"Running program: {__file__}\n")
    logger.info(f"{args}\n")

    printable_config = {section: dict(config[section]) for section in config}
    logger.info(f"Experiment configuration:\n{printable_config}\n")

    # Load the dataset
    _, valid_df, test_df = load_unicorn_entity_matching_benchmark(
        config.get("exp", "data_dir")
    )
    valid_dataset = Dataset.from_pandas(valid_df)
    test_dataset = Dataset.from_pandas(test_df)

    tokenizer = AutoTokenizer.from_pretrained(config["llm"]["checkpoint_dir"])

    tokenized_valid_dataset = valid_dataset.map(
        encode_entity_matching_input,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=[
            "entity_1",
            "entity_2",
        ],  # A list of columns to remove after applying the function
    )
    tokenized_test_dataset = test_dataset.map(
        encode_entity_matching_input,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=[
            "entity_1",
            "entity_2",
        ],  # A list of columns to remove after applying the function
    )

    # Rename the label column to labels because the model expects the argument
    # to be named labels
    tokenized_valid_dataset = tokenized_valid_dataset.rename_column(
        "label", "labels"
    )
    tokenized_test_dataset = tokenized_test_dataset.rename_column(
        "label", "labels"
    )

    # Set the format of the dataset to return PyTorch tensors instead of lists
    tokenized_valid_dataset.set_format(
        "torch", columns=["input_ids", "attention_mask", "labels"]
    )
    tokenized_test_dataset.set_format(
        "torch", columns=["input_ids", "attention_mask", "labels"]
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    valid_dataloader = DataLoader(
        tokenized_valid_dataset,
        batch_size=config.getint("llm", "batch_size"),
        shuffle=False,
        collate_fn=data_collator,
    )
    test_dataloader = DataLoader(
        tokenized_test_dataset,
        batch_size=config.getint("llm", "batch_size"),
        shuffle=False,
        collate_fn=data_collator,
    )

    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(
        config["llm"]["checkpoint_dir"],
        num_labels=len(ID2LABEL),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    # Get predictions
    valid_preds, valid_confdc_scores, valid_labels = get_predictions(
        model, valid_dataloader, device
    )
    log_exp_metrics(
        "Validation split", valid_labels, valid_preds, logger, multi_class=False
    )

    test_preds, test_confdc_scores, test_labels = get_predictions(
        model, test_dataloader, device
    )
    log_exp_metrics(
        "Test split", test_labels, test_preds, logger, multi_class=False
    )

    prepare_mrf_inputs_for_entity_matching(
        valid_df, valid_preds, valid_confdc_scores, output_dir, split="valid"
    )

    prepare_mrf_inputs_for_entity_matching(
        test_df, test_preds, test_confdc_scores, output_dir, split="test"
    )
