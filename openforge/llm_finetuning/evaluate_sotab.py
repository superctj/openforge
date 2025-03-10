import argparse
import os

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
    encode_data_matching_input,
    load_openforge_sotab_benchmark,
    prepare_sotab_for_sequence_classification,
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
    oid = 0
    object_id_map = {}
    rv_names = []

    for _, row in source_df.iterrows():
        o1 = row["object_1"]

        if o1 not in object_id_map:
            object_id_map[o1] = oid
            oid += 1

    for _, row in source_df.iterrows():
        o2 = row["object_2"]

        if o2 not in object_id_map:
            object_id_map[o2] = oid
            oid += 1

        o1 = row["object_1"]
        rv_names.append(f"R_{object_id_map[o1]}-{object_id_map[o2]}")

    source_df["random_variable_name"] = rv_names
    source_df["prediction"] = preds
    source_df["confidence_score"] = pred_confdc_scores

    # Save the MRF inputs
    output_filepath = os.path.join(output_dir, f"{split}.json")
    source_df.to_json(output_filepath, orient="records", indent=4)


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
):
    input_dataset = Dataset.from_pandas(input_df)
    tokenized_input_dataset = input_dataset.map(
        encode_data_matching_input,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=[
            "object_1",
            "object_2",
        ],  # A list of columns to remove after applying the function
    )

    # Rename the label column to labels because the model expects the argument
    # to be named labels
    tokenized_test_dataset = tokenized_input_dataset.rename_column(
        "label", "labels"
    )

    # Set the format of the dataset to return PyTorch tensors instead of lists
    tokenized_test_dataset.set_format(
        "torch", columns=["input_ids", "attention_mask", "labels"]
    )

    input_dataloader = DataLoader(
        tokenized_test_dataset,
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

        preds.extend(batch_preds.tolist())
        confdc_scores.extend(batch_confdc_scores.tolist())

    input_df["prediction"] = preds
    input_df["confidence_score"] = confdc_scores
    input_df.rename(
        columns={"relation_variable_name": "random_variable_name"}, inplace=True
    )

    labels = input_df["label"].tolist()
    log_exp_metrics(split, labels, preds, logger, multi_class=False)

    output_filepath = os.path.join(output_dir, f"{split}.json")
    input_df.to_json(output_filepath, orient="records", indent=4)


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

    # For this dataset, training and validation splits are the same
    _, valid_df, test_df = load_openforge_sotab_benchmark(
        os.path.join(config.get("io", "input_dir"), "artifact")
    )
    valid_df = prepare_sotab_for_sequence_classification(
        valid_df, root_dir=config.get("io", "input_dir")
    )
    test_df = prepare_sotab_for_sequence_classification(
        test_df, root_dir=config.get("io", "input_dir")
    )

    tokenizer = AutoTokenizer.from_pretrained(config["llm"]["checkpoint_dir"])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

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
    if not model.config.pad_token_id:
        model.config.pad_token_id = model.config.eos_token_id

    model = PeftModel.from_pretrained(model, config["llm"]["checkpoint_dir"])
    model = model.to(device)
    model.eval()

    # Get predictions
    batch_size = config.getint("llm", "batch_size")
    temperature = config.getfloat("llm", "temperature")

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
    )

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
    )
