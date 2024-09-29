import argparse
import os

import evaluate
import numpy as np
import torch

from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from openforge.utils.custom_logging import create_custom_logger
from openforge.utils.llm_common import (
    ID2LABEL,
    LABEL2ID,
    load_unicorn_entity_matching_benchmark,
)
from openforge.utils.util import parse_config

f1_metric = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return f1_metric.compute(predictions=predictions, references=labels)


def encode_entity_matching_input(
    examples,
    tokenizer,
):
    return tokenizer(
        examples["entity_1"],
        examples["entity_2"],
        truncation=True,
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
    output_dir = config.get("exp", "output_dir")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = create_custom_logger(output_dir)
    logger.info(f"Running program: {__file__}\n")
    logger.info(f"{args}\n")

    printable_config = {section: dict(config[section]) for section in config}
    logger.info(f"Experiment configuration:\n{printable_config}\n")

    train_df, valid_df, test_df = load_unicorn_entity_matching_benchmark(
        config.get("exp", "data_dir")
    )
    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)
    test_dataset = Dataset.from_pandas(test_df)

    model_id = config.get("llm", "model_id")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    tokenized_train_dataset = train_dataset.map(
        encode_entity_matching_input,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=[
            "entity_1",
            "entity_2",
        ],  # A list of columns to remove after applying the function
    )
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=len(ID2LABEL),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.to(device)

    # Freeze all layers except the classification head
    for param in model.base_model.parameters():
        param.requires_grad = False

    # Pad the sentences to the longest length in a batch during collation
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=config.getfloat("llm", "learning_rate"),
        per_device_train_batch_size=config.getint("llm", "train_batch_size"),
        per_device_eval_batch_size=config.getint("llm", "eval_batch_size"),
        num_train_epochs=config.getint("llm", "num_train_epochs"),
        weight_decay=config.getfloat("llm", "weight_decay"),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,  # Only to save the best model
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_valid_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    test_results = trainer.evaluate(tokenized_test_dataset)
    logger.info(f"Test results:\n{test_results}")
