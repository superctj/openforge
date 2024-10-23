import argparse

import pandas as pd

from openforge.utils.llm_common import load_unicorn_benchmark


def profile_entities(df: pd.DataFrame):
    entity_count_map = {}
    pos_class_count = 0

    for _, row in df.iterrows():
        if row["object_1"] not in entity_count_map:
            entity_count_map[row["object_1"]] = 1
        else:
            entity_count_map[row["object_1"]] += 1

        if row["object_2"] not in entity_count_map:
            entity_count_map[row["object_2"]] = 1
        else:
            entity_count_map[row["object_2"]] += 1

        if row["label"] == 1:
            pos_class_count += 1

    print(f"Number of positive class instances: {pos_class_count}")
    print(f"Number of unique objects: {len(entity_count_map)}")

    # Count the number of entities that appear more than once
    num_entities_mt_once = 0
    for _, entity_count in entity_count_map.items():
        if entity_count > 1:
            num_entities_mt_once += 1

    print(
        f"Number of objects that appear more than once: {num_entities_mt_once}"
    )

    # print sorted entities by count
    sorted_entities = sorted(
        entity_count_map.items(), key=lambda x: x[1], reverse=True
    )
    print(sorted_entities[:10])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the directory containing the benchmark.",
    )

    args = parser.parse_args()

    _, valid_df, test_df = load_unicorn_benchmark(args.data_dir)

    print(f"Number of pairs in the validation set: {valid_df.shape[0]}")
    profile_entities(valid_df)

    print(f"Number of pairs in the test set: {test_df.shape[0]}")
    profile_entities(test_df)
