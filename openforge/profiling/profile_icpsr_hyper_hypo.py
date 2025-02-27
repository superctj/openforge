import argparse
import os

import pandas as pd


def count_concepts(df, split: str = "training"):
    concept_1 = set(df["concept_1"])
    concept_2 = set(df["concept_2"])
    num_concepts = len(concept_1.union(concept_2))
    print(f"Number of unique concepts in {split} split: {num_concepts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        help="Path to the directory containing the data files.",
        default="/nfs/turbo/coe-jag/congtj/openforge/icpsr/artifact",
    )

    args = parser.parse_args()

    training_filepath = os.path.join(
        args.data_dir, "openforge_icpsr_hyper_hypo_training.csv"
    )
    validation_filepath = os.path.join(
        args.data_dir, "openforge_icpsr_hyper_hypo_validation.csv"
    )
    test_filepath = os.path.join(
        args.data_dir, "openforge_icpsr_hyper_hypo_test.csv"
    )

    train_df = pd.read_csv(training_filepath)
    valid_df = pd.read_csv(validation_filepath)
    test_df = pd.read_csv(test_filepath)

    count_concepts(train_df, split="training")
    count_concepts(valid_df, split="validation")
    count_concepts(test_df, split="test")
