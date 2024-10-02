import argparse

from openforge.utils.llm_common import load_unicorn_entity_matching_benchmark


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the directory containing the benchmark.",
    )

    args = parser.parse_args()

    _, _, test_df = load_unicorn_entity_matching_benchmark(args.data_dir)
    print(f"Number of pairs in the test set: {test_df.shape[0]}")

    entity_count_map = {}

    for i, row in test_df.iterrows():
        if row["entity_1"] not in entity_count_map:
            entity_count_map[row["entity_1"]] = 1
        else:
            entity_count_map[row["entity_1"]] += 1

        if row["entity_2"] not in entity_count_map:
            entity_count_map[row["entity_2"]] = 1
        else:
            entity_count_map[row["entity_2"]] += 1

    print(f"Number of unique entities: {len(entity_count_map)}")

    # Count the number of entities that appear more than once
    num_entities_mt_once = 0
    for entity, entity_count in entity_count_map.items():
        if entity_count > 1:
            num_entities_mt_once += 1

    print(
        f"Number of entities that appear more than once: {num_entities_mt_once}"
    )

    # print sorted entities by count
    sorted_entities = sorted(
        entity_count_map.items(), key=lambda x: x[1], reverse=True
    )
    print(sorted_entities[:10])
