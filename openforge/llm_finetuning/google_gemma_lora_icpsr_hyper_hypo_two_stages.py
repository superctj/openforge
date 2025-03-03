import argparse
import os

import evaluate
import numpy as np
import torch

from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from openforge.utils.custom_logging import create_custom_logger
from openforge.utils.llm_common import load_icpsr_hyper_hypo_two_stages_dataset
from openforge.utils.util import parse_config


f1_metric = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
metric_average = "macro"

FIRST_STAGE_ID2LABEL = {0: "null", 1: "hyper-hypo"}
FIRST_STAGE_LABEL2ID = {v: k for k, v in FIRST_STAGE_ID2LABEL.items()}
SECOND_STAGE_ID2LABEL = {0: "hypernymy", 1: "hyponymy"}
SECOND_STAGE_LABEL2ID = {v: k for k, v in SECOND_STAGE_ID2LABEL.items()}


def encode_icpsr_input(examples, tokenizer):
    return tokenizer(
        examples["concept_1"],
        examples["concept_2"],
        truncation=True,
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    f1 = f1_metric.compute(
        predictions=predictions, references=labels, average=metric_average
    )["f1"]
    precision = precision_metric.compute(
        predictions=predictions, references=labels, average=metric_average
    )["precision"]
    recall = recall_metric.compute(
        predictions=predictions, references=labels, average=metric_average
    )["recall"]

    return {"f1": f1, "precision": precision, "recall": recall}


class CustomTrainer(Trainer):
    def __init__(self, class_weights, label_smoothing, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(
                self.class_weights,
                device=model.device,
                dtype=logits.dtype,
            ),
            label_smoothing=self.label_smoothing,
        )
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


def train_and_evaluate(
    train_df,
    valid_df,
    test_df,
    config,
    output_dir,
    stage
):
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_df["label"]),
        y=train_df["label"],
    )
    logger.info(f"Class weights: {class_weights}")

    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)
    test_dataset = Dataset.from_pandas(test_df)

    model_id = config.get("llm", "model_id")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    if model_id.startswith("Qwen"):
        tokenizer.pad_token = tokenizer.eos_token

    tokenized_train_dataset = train_dataset.map(
        encode_icpsr_input,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=[
            "concept_1",
            "concept_2",
        ],  # A list of columns to remove after applying the function
    )
    tokenized_valid_dataset = valid_dataset.map(
        encode_icpsr_input,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=[
            "concept_1",
            "concept_2",
        ],  # A list of columns to remove after applying the function
    )
    tokenized_test_dataset = test_dataset.map(
        encode_icpsr_input,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=[
            "concept_1",
            "concept_2",
        ],  # A list of columns to remove after applying the function
    )

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        target_modules=["q_proj", "v_proj"],
        r=config.getint("llm", "r"),
        lora_alpha=config.getint("llm", "lora_alpha"),
        lora_dropout=config.getfloat("llm", "lora_dropout"),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if stage == "first":
        ID2LABEL = FIRST_STAGE_ID2LABEL
        LABEL2ID = FIRST_STAGE_LABEL2ID
    else:
        ID2LABEL = SECOND_STAGE_ID2LABEL
        LABEL2ID = SECOND_STAGE_LABEL2ID

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    if model_id.startswith("Qwen"):
        model.config.pad_token_id = tokenizer.eos_token_id

    model = get_peft_model(model, peft_config)
    # Check if the score layer is trainable
    for name, param in model.named_parameters():
        if name == "score.weight":
            assert param.requires_grad

    model.print_trainable_parameters()

    # Pad the sentences to the longest length in a batch during collation
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        bf16=True,
        learning_rate=config.getfloat("llm", "learning_rate"),
        per_device_train_batch_size=config.getint("llm", "train_batch_size"),
        per_device_eval_batch_size=config.getint("llm", "eval_batch_size"),
        num_train_epochs=config.getint("llm", "num_train_epochs"),
        weight_decay=config.getfloat("llm", "weight_decay"),
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,  # Only to save the best model
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_valid_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        label_smoothing=config.getfloat("llm", "label_smoothing"),
    )

    trainer.train()

    test_results = trainer.evaluate(tokenized_test_dataset)
    logger.info(f"Test results:\n{test_results}")


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

    (
        (first_stage_train_df, first_stage_valid_df, first_stage_test_df), (second_stage_train_df, second_stage_valid_df, second_stage_test_df)
    ) = load_icpsr_hyper_hypo_two_stages_dataset(
        config.get("io", "input_dir"), rename_columns=True, augment_data=True
    )

    first_stage_output_dir = os.path.join(
        output_dir,
        "first_stage",
    )
    if not os.path.exists(first_stage_output_dir):
        os.makedirs(first_stage_output_dir)

    logger.info("First stage training and evaluation:")
    train_and_evaluate(
        first_stage_train_df,
        first_stage_valid_df,
        first_stage_test_df,
        config,
        first_stage_output_dir,
        stage="first",
    )

    logger.info("-" * 50)
    second_stage_output_dir = os.path.join(
        output_dir,
        "second_stage",
    )
    if not os.path.exists(second_stage_output_dir):
        os.makedirs(second_stage_output_dir)

    logger.info("Second stage training and evaluation:")
    train_and_evaluate(
        second_stage_train_df,
        second_stage_valid_df,
        second_stage_test_df,
        config,
        second_stage_output_dir,
        stage="second",
    )
    
