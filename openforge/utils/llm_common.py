import json
import logging
import os

import pandas as pd


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


def sample_column_values(table_filepath: str, col_id: int, n: int = 10):
    table = pd.read_json(table_filepath, compression="gzip", lines=True)
    table = table.astype(str)
    uniq_vals = table.iloc[:, col_id].dropna().unique().tolist()

    k = n
    sample = uniq_vals[:k]

    # Ballpark constraint on the length of the inputs for few-shot learning
    while len(" ".join(sample)) > 800:
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

    # For example:

    # """  # noqa: E501
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


def parse_openai_response(response: str) -> int:
    logger = logging.getLogger()

    json_str = response.choices[0].message.content
    json_str = json_str.replace("'", '"')

    try:
        pred = int(json.loads(json_str)["match"])
    except json.decoder.JSONDecodeError as e:
        logger.warning(f"Invalid response: {json_str}. Original error: {e}")
        pred = 0

    return pred

    # try:
    #     decision = json.loads(json_str)["match"]
    # except json.JSONDecodeError as e:
    #     raise ValueError(f"Invalid response: {json_str}. Original error: {e}")
    # except KeyError as e:
    #     raise ValueError(f"Invalid response: {json_str}. Original error: {e}")

    # return int(decision)
