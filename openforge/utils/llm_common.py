import json
import logging
import os
import re

import numpy as np
import pandas as pd
import torch.nn as nn

from sklearn.utils.class_weight import compute_class_weight
from transformers import Trainer


ID2LABEL = {0: "NONMATCH", 1: "MATCH"}
LABEL2ID = {"NONMATCH": 0, "MATCH": 1}


def load_openforge_sotab_split(
    split_filepath: str, logger: logging.Logger = None
) -> pd.DataFrame:
    """Load a split of an OpenForge-SOTAB dataset.

    Args:
        split_filepath: The file path to the split.
         logger: The (optional) logging instance.

    Returns:
        The split as a DataFrame.
    """

    df = pd.read_csv(split_filepath, delimiter=",", header=0)

    if logger:
        logger.info(f"Number of instances: {df.shape[0]}")
        logger.info(f"Top 5 rows:\n{df.head()}\n")

    return df


def load_openforge_sotab_benchmark(
    data_dir: str, logger: logging.Logger = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load training, validation, and test splits of the OpenForge-SOTAB-X
    benchmark.

    Args:
        data_dir: The directory containing the benchmark.
        logger: The (optional) logging instance.

    Returns:
        The training, validation, and test data.
    """

    train_filepath = os.path.join(data_dir, "training.csv")
    valid_filepath = os.path.join(data_dir, "validation.csv")
    test_filepath = os.path.join(data_dir, "test.csv")

    if logger:
        logger.info("Loading training split...")

    train_df = load_openforge_sotab_split(train_filepath, logger)

    if logger:
        logger.info("Loading validation split...")

    valid_df = load_openforge_sotab_split(valid_filepath, logger)

    if logger:
        logger.info("Loading test split...")

    test_df = load_openforge_sotab_split(test_filepath, logger)

    return train_df, valid_df, test_df


def preprocess_imbalanced_dataset(
    df: pd.DataFrame, factor=10, random_seed: int = 42
) -> tuple[pd.DataFrame, np.array]:
    """Preprocess an imbalanced dataset by undersampling the majority class if
    the majority class is `factor` times bigger than the minority class.

    Args:
        df: The input DataFrame.
        random_seed: The random seed for reproducibility.

    Returns:
        The preprocessed DataFrame and class weights for the preprocessed
        dataset.
    """

    minority_class = df[df["label"] == 1]
    majority_class = df[df["label"] == 0]

    n_minority = len(minority_class)
    n_majority = len(majority_class)

    preprocessed_df = df

    if n_minority * factor < n_majority:
        majority_class = majority_class.sample(
            n_minority * factor, random_state=random_seed
        )

        preprocessed_df = pd.concat([minority_class, majority_class])
        preprocessed_df = preprocessed_df.sample(
            frac=1, random_state=random_seed
        )

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=preprocessed_df["label"],
    )

    return preprocessed_df, class_weights


def load_unicorn_benchmark(data_dir: str):
    train_df = pd.read_json(os.path.join(data_dir, "train.json"))
    valid_df = pd.read_json(os.path.join(data_dir, "valid.json"))
    test_df = pd.read_json(os.path.join(data_dir, "test.json"))

    header = ["object_1", "object_2", "label"]
    train_df.columns = header
    valid_df.columns = header
    test_df.columns = header

    return train_df, valid_df, test_df


def sample_column_values(
    table_filepath: str, col_id: int, n: int = 10, max_len: int = 800
):
    table = pd.read_json(table_filepath, compression="gzip", lines=True)
    table = table.astype(str)
    uniq_vals = table.iloc[:, col_id].dropna().unique().tolist()

    k = n
    sample = uniq_vals[:k]

    # Ballpark constraint on the length of the inputs for few-shot learning
    while len(" ".join(sample)) > max_len:
        k -= 1
        sample = uniq_vals[:k]

    if k:
        return uniq_vals[:k]
    else:
        return uniq_vals[:1]


def sample_few_shot_examples(
    df: pd.DataFrame, n: int = 10, balanced: bool = True, random_seed: int = 42
) -> pd.DataFrame:
    if n == 0:
        return pd.DataFrame()

    if not balanced:
        return df.sample(n, random_state=random_seed)

    sample_df = pd.concat(
        [
            df[df["relation_variable_label"] == 1].sample(
                n // 2, random_state=random_seed
            ),
            df[df["relation_variable_label"] == 0].sample(
                n - n // 2, random_state=random_seed
            ),
        ]
    )
    sample_df = sample_df.sample(frac=1)  # random shuffle

    return sample_df[
        [
            "label_1_processed",
            "label_2_processed",
            "relation_variable_label",
            "label_1_table_path",
            "label_2_table_path",
            "label_1_col_idx",
            "label_2_col_idx",
        ]
    ]


def craft_sotab_user_prompt(
    test_record: dict, few_shot_df: pd.DataFrame
) -> str:
    prompt = """Column semantic types are used to describe semantics of values contained in a table column. Column semantic types from different vocabularies or ontologies can have the same meaning. Determine whether two semantic types are equivelant. For each input semantic type, you will also be provided with sample column values from columns labeled with the input semantic type."""  # noqa: E501

    if not few_shot_df.empty:
        prompt += """\n\nFor example,\n\n"""

        fewshot_prompt = "\n\n".join(
            [
                "Input:\nSemantic type 1: {}\nSample column values for type 1: {}\n\nSemantic type 2: {}\nSample column values for type 2: {}\n\nOutput:\n{}".format(  # noqa: E501
                    row[0],
                    sample_column_values(row[3], row[5]),
                    row[1],
                    sample_column_values(row[4], row[6]),
                    '{"match": true}' if row[2] else '{"match": false}',
                )
                for row in few_shot_df.values.tolist()
            ]
        )

        prompt += fewshot_prompt

    prompt += """

Now, for the following semantic type pairs, please determine if they are equivalent. Return your prediction in the following JSON format: '{{"match": true}}' or '{{"match": false}}'.

Input:
Semantic type 1: {}
Sample column values for type 1: {}

Semantic type 2: {}
Sample column values for type 2: {}

Output:
""".format(  # noqa: E501
        test_record["label_1_processed"],
        sample_column_values(
            test_record["label_1_table_path"], test_record["label_1_col_idx"]
        ),
        test_record["label_2_processed"],
        sample_column_values(
            test_record["label_2_table_path"], test_record["label_2_col_idx"]
        ),
    )

    return prompt


def craft_sotab_entailment_prompt(
    data_record: pd.Series,
    few_shot_df: pd.DataFrame = None,
    num_samples: int = 10,
    root_dir: str = None,
) -> tuple[str, str, list[str]]:
    if root_dir:
        label_1_table_path = data_record["label_1_table_path"].replace(
            "/ssd/congtj/openforge/sotab_v2", root_dir
        )
        label_2_table_path = data_record["label_2_table_path"].replace(
            "/ssd/congtj/openforge/sotab_v2", root_dir
        )
    else:
        label_1_table_path = data_record["label_1_table_path"]
        label_2_table_path = data_record["label_2_table_path"]

    label_1_sample_values = sample_column_values(
        label_1_table_path,
        data_record["label_1_col_idx"],
        n=num_samples,
        max_len=300,
    )
    label_2_sample_values = sample_column_values(
        label_2_table_path,
        data_record["label_2_col_idx"],
        n=num_samples,
        max_len=300,
    )

    prompt = f"Semantic type 1 has name '{data_record['label_1_processed']}' and sample values '{label_1_sample_values}'; Semantic type 2 has name '{data_record['label_2_processed']}' and sample values '{label_2_sample_values}'."  # noqa: E501

    return prompt


def parse_llm_response(response: str) -> int:
    logger = logging.getLogger()

    pattern = r"{[^}]*}"
    matches = re.findall(pattern, response)

    if not matches:
        logger.info(f"No match found in response: {response}")
        return 0

    json_str = matches[0].strip().replace("'", '"')

    try:
        pred = int(json.loads(json_str)["match"])
    except json.JSONDecodeError as e:
        logger.info(f"Invalid response: {json_str}. Original error: {e}")
        pred = 0
    except KeyError as e:
        logger.info(f"Invalid response: {json_str}. Original error: {e}")
        pred = 0

    return pred


def encode_data_matching_input(examples, tokenizer):
    return tokenizer(
        examples["object_1"],
        examples["object_2"],
        truncation=True,
    )


class CustomTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = nn.CrossEntropyLoss(weight=self.class_weights)(logits, labels)

        return (loss, outputs) if return_outputs else loss
