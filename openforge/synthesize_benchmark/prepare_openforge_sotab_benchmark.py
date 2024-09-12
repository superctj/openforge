import argparse
import os
import random

from itertools import combinations

import pandas as pd

from openforge.utils.custom_logging import create_custom_logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source_data_filepath",
        type=str,
        default="/ssd/congtj/openforge/sotab_v2/artifact/sotab_v2_test_vocabulary.csv",  # noqa: 501
        help="Path to the source data.",
    )

    parser.add_argument(
        "--train_prop", type=float, default=0, help="Training proportion."
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=12345,
        help="Random seed for sampling.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/ssd/congtj/openforge/sotab_v2/artifact/sotab_v2_test_openforge_xlarge",  # noqa: 501
        help="Path to the output directory.",
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="/home/congtj/openforge/logs/sotab_v2/openforge_sotab_xlarge",
        help="Directory to save logs.",
    )

    args = parser.parse_args()

    # Fix random seed
    random.seed(args.random_seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    logger = create_custom_logger(args.log_dir)
    logger.info(f"Running program: {__file__}\n")
    logger.info(f"{args}\n")

    # Load data
    df = pd.read_csv(args.source_data_filepath)

    num_concepts = 1  # Count the first concept
    varname_dfidx_map = {}

    for i, row in enumerate(df.itertuples()):
        if row.relation_variable_name.startswith("R_1-"):
            num_concepts += 1

        varname_dfidx_map[row.relation_variable_name] = i

    assert len(df) == len(varname_dfidx_map)
    logger.info(f"Number of concepts: {num_concepts}")

    concept_indices = list(range(1, num_concepts + 1))
    train_indices = random.sample(
        population=concept_indices,
        k=int(len(concept_indices) * args.train_prop),
    )

    valid_test_concept_indices = list(set(concept_indices) - set(train_indices))
    valid_indices = random.sample(
        population=valid_test_concept_indices,
        k=int(len(valid_test_concept_indices) / 2),
    )
    test_indices = list(set(valid_test_concept_indices) - set(valid_indices))

    train_indices.sort()
    valid_indices.sort()
    test_indices.sort()

    train_df_idxs = []
    valid_df_idxs = []
    test_df_idxs = []

    for pair in combinations(train_indices, 2):
        var_name = f"R_{pair[0]}-{pair[1]}"
        train_df_idxs.append(varname_dfidx_map[var_name])

    for pair in combinations(valid_indices, 2):
        var_name = f"R_{pair[0]}-{pair[1]}"
        valid_df_idxs.append(varname_dfidx_map[var_name])

    for pair in combinations(test_indices, 2):
        var_name = f"R_{pair[0]}-{pair[1]}"
        test_df_idxs.append(varname_dfidx_map[var_name])

    train_df = df.iloc[train_df_idxs].rename(
        columns={"relation_variable_name": "original_relation_variable_name"}
    )
    valid_df = df.iloc[valid_df_idxs].rename(
        columns={"relation_variable_name": "original_relation_variable_name"}
    )
    test_df = df.iloc[test_df_idxs].rename(
        columns={"relation_variable_name": "original_relation_variable_name"}
    )

    logger.info(f"Number of instances in the training split: {len(train_df)}")
    logger.info(
        "Number of positive instances in the training split: "
        f"{train_df['relation_variable_label'].sum()}"
    )

    logger.info(f"Number of instances in the validation split: {len(valid_df)}")
    logger.info(
        "Number of positive instances in the validation split: "
        f"{valid_df['relation_variable_label'].sum()}"
    )

    logger.info(f"Number of instances in the test split: {len(test_df)}")
    logger.info(
        "Number of positive instances in the test split: "
        f"{test_df['relation_variable_label'].sum()}"
    )

    # Add relation variable name with respect to the split
    train_var_names, valid_var_names, test_var_names = [], [], []
    new_train_indices = list(range(1, len(train_indices) + 1))
    new_valid_indices = list(range(1, len(valid_indices) + 1))
    new_test_indices = list(range(1, len(test_indices) + 1))

    for pair in combinations(new_train_indices, 2):
        var_name = f"R_{pair[0]}-{pair[1]}"
        train_var_names.append(var_name)

    for pair in combinations(new_valid_indices, 2):
        var_name = f"R_{pair[0]}-{pair[1]}"
        valid_var_names.append(var_name)

    for pair in combinations(new_test_indices, 2):
        var_name = f"R_{pair[0]}-{pair[1]}"
        test_var_names.append(var_name)

    if args.train_prop != 0:
        train_df["relation_variable_name"] = train_var_names
        train_output_filepath = os.path.join(args.output_dir, "training.csv")
        train_df.to_csv(train_output_filepath, index=False)

    valid_df["relation_variable_name"] = valid_var_names
    test_df["relation_variable_name"] = test_var_names

    valid_output_filepath = os.path.join(args.output_dir, "validation.csv")
    test_output_filepath = os.path.join(args.output_dir, "test.csv")

    valid_df.to_csv(valid_output_filepath, index=False)
    test_df.to_csv(test_output_filepath, index=False)
