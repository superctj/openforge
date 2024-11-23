import json
import logging
import os
import re

import numpy as np
import pandas as pd

# import torch.nn as nn

from sklearn.utils.class_weight import compute_class_weight

# from transformers import Trainer


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


def load_em_walmart_amazon_dataset(data_dir: str):
    train_df = pd.read_json(
        os.path.join(data_dir, "preprocessed_training.json")
    )
    valid_df = pd.read_json(
        os.path.join(data_dir, "preprocessed_validation.json")
    )
    test_df = pd.read_json(os.path.join(data_dir, "preprocessed_test.json"))

    return train_df, valid_df, test_df


def load_icpsr_dataset(data_dir: str, rename_columns: bool = True):
    columns_needed = [
        "concept_1",
        "concept_2",
        "relation_variable_name",
        "relation_variable_label",
    ]

    train_df = pd.read_csv(
        os.path.join(data_dir, "openforge_icpsr_hyper_training.csv")
    )
    train_df = train_df[columns_needed]

    valid_df = pd.read_csv(
        os.path.join(data_dir, "openforge_icpsr_hyper_validation.csv")
    )
    valid_df = valid_df[columns_needed]

    test_df = pd.read_csv(
        os.path.join(data_dir, "openforge_icpsr_hyper_test.csv")
    )
    test_df = test_df[columns_needed]

    if rename_columns:
        train_df.rename(
            columns={"relation_variable_label": "label"}, inplace=True
        )
        valid_df.rename(
            columns={"relation_variable_label": "label"}, inplace=True
        )
        test_df.rename(
            columns={"relation_variable_label": "label"}, inplace=True
        )

    return train_df, valid_df, test_df


def flatten_list(lst):
    for item in lst:
        if isinstance(item, list):
            yield from flatten_list(item)
        else:
            if item:
                yield item


def sample_column_values(
    table_filepath: str, col_id: int, n: int = 10, max_len: int = 800
):
    table = pd.read_json(table_filepath, compression="gzip", lines=True)

    try:
        uniq_vals = (
            table.iloc[:, col_id]
            .dropna()
            .apply(
                lambda x: (
                    ", ".join(list(flatten_list(x)))
                    if isinstance(x, list)
                    else x
                )
            )
            .unique()
            .tolist()
        )
        uniq_vals = [str(val).split(", ") for val in uniq_vals if val]
        uniq_vals = list(flatten_list(uniq_vals))
    except TypeError as e:
        print(table.iloc[:, col_id].tolist())
        raise e

    k = n
    sample = uniq_vals[:k]

    # Ballpark constraint on the length of the inputs for few-shot learning
    while len(", ".join(sample)) > max_len:
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
    row: pd.Series,
    few_shot_df: pd.DataFrame,
    root_dir: str = None,
) -> str:
    prompt = """Semantic column types are used to describe semantics of values contained in a table column, i.e., what domains these column values belong to. Determine whether two semantic column types are equivelant. For each input type, you will also be provided with sample column values from columns labeled with the type."""  # noqa: E501

    if root_dir:
        label_1_table_path = row["label_1_table_path"].replace(
            "/ssd/congtj/openforge/sotab_v2", root_dir
        )
        label_2_table_path = row["label_2_table_path"].replace(
            "/ssd/congtj/openforge/sotab_v2", root_dir
        )
    else:
        label_1_table_path = row["label_1_table_path"]
        label_2_table_path = row["label_2_table_path"]

    if few_shot_df is not None and not few_shot_df.empty:
        prompt += """\n\nFor example,\n\n"""

        fewshot_prompt = "\n\n".join(
            [
                "Input:\nSemantic type 1: {}\nSample column values for type 1: {}\n\nSemantic type 2: {}\nSample column values for type 2: {}\n\nOutput:\n{}".format(  # noqa: E501
                    row[0],
                    sample_column_values(label_1_table_path, row[5]),
                    row[1],
                    sample_column_values(label_2_table_path, row[6]),
                    (
                        '{"equivalent": true}'
                        if row[2]
                        else '{"equivalent": false}'
                    ),
                )
                for row in few_shot_df.values.tolist()
            ]
        )

        prompt += fewshot_prompt

    prompt += """

Now, for the following pair of semantic column types, please determine if they are equivalent.

Input:
Semantic column type 1: {}
Sample column values for type 1: {}

Semantic column type 2: {}
Sample column values for type 2: {}

Return your prediction and confidence score in the following JSON format: '{{"equivalent": true, "confidence score": 0.75}}'. The prediction can only be true of false, and the confidence score needs to be greater than 0.5 and smaller than 1.

Output:
""".format(  # noqa: E501
        row["label_1_processed"],
        sample_column_values(label_1_table_path, row["label_1_col_idx"]),
        row["label_2_processed"],
        sample_column_values(label_2_table_path, row["label_2_col_idx"]),
    )

    return prompt


# Return your prediction and confidence score in the following JSON format: '{{"equivalent": true, "confidence score":}}' or '{{"equivalent": false, "confidence score":}}'. Confidence score needs to be greater than 0.5 and smaller than 1. # noqa: E501


def craft_sotab_user_prompt_w_single_token_response(
    data_record: pd.Series,
    few_shot_df: pd.DataFrame,
    num_samples: int = 10,
    root_dir: str = None,
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
        max_len=800,
    )
    label_2_sample_values = sample_column_values(
        label_2_table_path,
        data_record["label_2_col_idx"],
        n=num_samples,
        max_len=800,
    )

    prompt += """

Now, for the following semantic type pairs, please determine if they are equivalent. Return your prediction with a single token 'n' or 'e' where 'n' represents non-equivalent and 'e' represents equivalent.

Input:
Semantic type 1: {}
Sample column values for type 1: {}

Semantic type 2: {}
Sample column values for type 2: {}

Output:
""".format(  # noqa: E501
        data_record["label_1_processed"],
        label_1_sample_values,
        data_record["label_2_processed"],
        label_2_sample_values,
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


def prepare_sotab_for_sequence_classification(
    df: pd.DataFrame,
    root_dir: str = None,
    num_samples: int = 5,
    max_len: int = 900,
) -> pd.DataFrame:
    selected_df = df.loc[
        :,
        [
            "label_1_processed",
            "label_2_processed",
            "label_1_table_path",
            "label_2_table_path",
            "label_1_col_idx",
            "label_2_col_idx",
            "relation_variable_label",
            "relation_variable_name",
        ],
    ]

    if root_dir:
        selected_df.loc[:, "label_1_table_path"] = selected_df[
            "label_1_table_path"
        ].apply(lambda x: x.replace("/ssd/congtj/openforge/sotab_v2", root_dir))

        selected_df.loc[:, "label_2_table_path"] = selected_df[
            "label_2_table_path"
        ].apply(lambda x: x.replace("/ssd/congtj/openforge/sotab_v2", root_dir))

    selected_df.loc[:, "object_1"] = selected_df.apply(
        lambda x: x["label_1_processed"]
        + ": "
        + ", ".join(
            sample_column_values(
                x["label_1_table_path"],
                x["label_1_col_idx"],
                n=num_samples,
                max_len=max_len,
            )
        ),
        axis=1,
    )

    selected_df.loc[:, "object_2"] = selected_df.apply(
        lambda x: x["label_2_processed"]
        + ": "
        + ", ".join(
            sample_column_values(
                x["label_2_table_path"],
                x["label_2_col_idx"],
                n=num_samples,
                max_len=max_len,
            )
        ),
        axis=1,
    )

    selected_df = selected_df.loc[
        :,
        [
            "object_1",
            "object_2",
            "relation_variable_label",
            "relation_variable_name",
        ],
    ]

    # Renmame column relation_variable_label to label
    selected_df = selected_df.rename(
        columns={"relation_variable_label": "label"}
    )

    return selected_df


def craft_data_matching_entailment_prompt(
    data_record: pd.Series, few_shot_df: pd.DataFrame = None
) -> str:
    prompt = f"""Data object 1 has description: {data_record["object_1"]}; Data object 2 has description: {data_record["object_2"]}."""  # noqa: E501

    return prompt


def craft_icpsr_user_prompt(
    row: pd.Series,
    few_shot_df: pd.DataFrame,
) -> str:
    prompt = """The task is to determine whether there exists a parent-child relationship between a given pair of concepts, i.e., whether one concept has a broader meaning that the other concept falls under."""  # noqa: E501

    if few_shot_df is not None and not few_shot_df.empty:
        prompt += """\n\nFor example,\n\n"""

        fewshot_prompt = "\n\n".join(
            [
                "Input: ({}, {})\nOutput:{}".format(  # noqa: E501
                    row.concept_1,
                    row.concept_2,
                    (
                        '{"parent-child": true}'
                        if row.relation_variable_label
                        else '{"parent-child": false}'
                    ),
                )
                for row in few_shot_df.itertuples()
            ]
        )

        prompt += fewshot_prompt

    prompt += """

Now, for the following pair of concepts, please determine if there is a parent-child relationship between them. Return your prediction and confidence score in the following JSON format: '{{"parent-child": true, "confidence score": 0.75}}'. The prediction can only be true of false, and the confidence score needs to be greater than 0.5 and smaller than 1.

Input: ({}, {})
Output:
""".format(  # noqa: E501
        row["concept_1"],
        row["concept_2"],
    )

    return prompt


def parse_llm_response(response: str, keyword: str = "equivalent") -> int:
    logger = logging.getLogger()

    pattern = r"{[^}]*}"
    matches = re.findall(pattern, response)

    if not matches:
        logger.info(f"No match found in response: {response}")
        return 0, -1

    json_str = matches[0].strip().replace("'", '"')

    try:
        pred = int(json.loads(json_str)[keyword])
        confdc_score = float(json.loads(json_str)["confidence score"])
    except json.JSONDecodeError as e:
        logger.info(f"Invalid response: {json_str}. Original error: {e}")
        pred = 0
        confdc_score = -1
    except KeyError as e:
        logger.info(f"Invalid response: {json_str}. Original error: {e}")
        pred = 0
        confdc_score = -1
    except ValueError as e:
        logger.info(f"Invalid response: {json_str}. Original error: {e}")
        pred = 0
        confdc_score = -1

    return pred, confdc_score


def encode_data_matching_input(examples, tokenizer):
    return tokenizer(
        examples["object_1"],
        examples["object_2"],
        truncation=True,
    )


def encode_em_walmart_amazon_input(examples, tokenizer):
    return tokenizer(
        examples["l_entity"],
        examples["r_entity"],
        truncation=True,
    )


def encode_icpsr_input(examples, tokenizer):
    return tokenizer(
        examples["concept_1"],
        examples["concept_2"],
        truncation=True,
    )
