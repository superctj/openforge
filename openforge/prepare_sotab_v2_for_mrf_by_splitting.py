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
        default="/home/congtj/openforge/exps/arts-context_top-40-nodes/sotab_v2_test_mrf_data_with_ml_prior.csv",  # noqa: 501
        help="Path to the source data.",
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
        default="/home/congtj/openforge/exps/arts-context_top-40-nodes",
        help="Path to the output directory.",
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="/home/congtj/openforge/logs/sotab_mrf_synthesized_data",
        help="Directory to store logs.",
    )

    args = parser.parse_args()

    # Fix random seed
    random.seed(args.random_seed)

    # if not os.path.exists(args.output_dir):
    #     os.makedirs(args.output_dir)

    # if not os.path.exists(args.log_dir):
    #     os.makedirs(args.log_dir)

    # logger = create_custom_logger(args.log_dir)
    # logger.info(args)

    # Load data
    df = pd.read_csv(args.source_data_filepath)

    num_concepts = 1  # Count the first concept
    varname_dfidx_map = {}

    for i, row in enumerate(df.itertuples()):
        if row.relation_variable_name.startswith("R_1-"):
            num_concepts += 1

        varname_dfidx_map[row.relation_variable_name] = i

    assert len(df) == len(varname_dfidx_map)
    print(f"Number of concepts: {num_concepts}")

    concept_indices = list(range(1, num_concepts + 1))
    valid_indices = random.sample(
        population=concept_indices, k=int(len(concept_indices) / 2)
    )
    test_indices = list(set(concept_indices) - set(valid_indices))
    valid_indices.sort()
    test_indices.sort()

    valid_df_idxs = []
    test_df_idxs = []

    for pair in combinations(valid_indices, 2):
        var_name = f"R_{pair[0]}-{pair[1]}"
        valid_df_idxs.append(varname_dfidx_map[var_name])

    for pair in combinations(test_indices, 2):
        var_name = f"R_{pair[0]}-{pair[1]}"
        test_df_idxs.append(varname_dfidx_map[var_name])

    valid_df = df.iloc[valid_df_idxs].rename(
        columns={"relation_variable_name": "original_relation_variable_name"}
    )
    test_df = df.iloc[test_df_idxs].rename(
        columns={"relation_variable_name": "original_relation_variable_name"}
    )

    print(f"Number of instances in the valid split: {len(valid_df)}")
    print(
        "Number of positive instances in the valid split: "
        f"{valid_df['relation_variable_label'].sum()}"
    )

    print(f"Number of instances in the test split: {len(test_df)}")
    print(
        "Number of positive instances in the test split: "
        f"{test_df['relation_variable_label'].sum()}"
    )

    # Add relation variable name with respect to the split
    valid_var_names, test_var_names = [], []
    new_valid_indices = list(range(1, len(valid_indices) + 1))
    new_test_indices = list(range(1, len(test_indices) + 1))

    for pair in combinations(new_valid_indices, 2):
        var_name = f"R_{pair[0]}-{pair[1]}"
        valid_var_names.append(var_name)

    for pair in combinations(new_test_indices, 2):
        var_name = f"R_{pair[0]}-{pair[1]}"
        test_var_names.append(var_name)

    valid_df["relation_variable_name"] = valid_var_names
    test_df["relation_variable_name"] = test_var_names

    valid_output_filepath = os.path.join(
        args.output_dir, "sotab_v2_test_mrf_data_valid_with_ml_prior.csv"
    )
    test_output_filepath = os.path.join(
        args.output_dir, "sotab_v2_test_mrf_data_test_with_ml_prior.csv"
    )

    valid_df.to_csv(valid_output_filepath, index=False)
    test_df.to_csv(test_output_filepath, index=False)
