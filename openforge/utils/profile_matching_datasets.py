import argparse
import os

from itertools import combinations

import networkx as nx
import pandas as pd

# from openforge.utils.llm_common import load_unicorn_benchmark


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


def profile_connected_components(df: pd.DataFrame):
    G = nx.Graph()

    for _, row in df.iterrows():
        lid = row["l_id"]
        rid = row["r_id"]
        prediction = row["prediction"]
        label = row["label"]

        G.add_node(lid)
        G.add_node(rid)
        G.add_edge(
            lid, rid, pair=(lid, rid), label=label, prediction=prediction
        )

    connected_components = list(nx.connected_components(G))

    # Sort the components by size in descending order
    cc_sorted = sorted(connected_components, key=len, reverse=True)

    # Iterate through each component, get the subgraph, and compute number of edges and cycles # noqa: E501
    cumulative_num_mispredictions = 0

    for i, component in enumerate(cc_sorted):
        subgraph = G.subgraph(component)

        edges = subgraph.edges(data=True)
        num_edges = subgraph.number_of_edges()
        cycles = nx.cycle_basis(subgraph)
        num_cycles = len(cycles)

        print(
            f"Component {i + 1} (size {len(component)}, edges {num_edges}, cycles {num_cycles})"  # noqa: E501
        )
        print(f"Edges: {edges}\n")
        print(f"Cycles: {cycles}\n")
        print(f"Component: {component}\n")

        num_mispredictions = 0
        for e in edges:
            if e[2]["prediction"] != e[2]["label"]:
                num_mispredictions += 1

        cumulative_num_mispredictions += num_mispredictions

        print(f"Number of mispredictions: {num_mispredictions}")
        print(
            f"Cumulative number of mispredictions in component {i}: {cumulative_num_mispredictions}"  # noqa: E501
        )
        print("=" * 80)

        # for node in subgraph.nodes:
        #     neighbors = list(subgraph.neighbors(node))
        #     print(f"Node: {node}, neighbors: {neighbors}")

        #     if len(neighbors) > 1:
        #         for pair in combinations(neighbors, 2):
        #             print(f"Pair: {pair}")

        #     exit(0)
    print("Total number of mispredictions: ", cumulative_num_mispredictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the directory containing the benchmark.",
    )

    args = parser.parse_args()

    # _, valid_df, test_df = load_unicorn_benchmark(args.data_dir)

    # print(f"Number of pairs in the validation set: {valid_df.shape[0]}")
    # profile_entities(valid_df)

    # print(f"Number of pairs in the test set: {test_df.shape[0]}")
    # profile_entities(test_df)

    test_prior_filepath = os.path.join(args.data_dir, "test.csv")
    test_df = pd.read_csv(test_prior_filepath)
    profile_connected_components(test_df)

    # valid_prior_filepath = os.path.join(args.data_dir, "validation.csv")
    # valid_df = pd.read_csv(valid_prior_filepath)
    # profile_connected_components(valid_df)
