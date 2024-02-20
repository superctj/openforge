import argparse
import os
import pickle

from sklearn.neighbors import NearestNeighbors
from openforge.ARTS.ontology import OntologyNode
from openforge.utils.custom_logging import create_custom_logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--arts_output_filepath",
        type=str,
        default="/home/congtj/openforge/data/\
column_semantics_ontology_threshold_0.9_run_nyc_gpt_3.5_merge_root.pickle",
        help="Path to the ARTS output file",
    )

    parser.add_argument(
        "--arts_level",
        type=int,
        default=2,
        help="Level of the ARTS ontology to extract concepts.",
    )

    parser.add_argument(
        "--num_head_nodes",
        type=int,
        default=100,
        help="Number of ARTS head nodes to consider",
    )

    parser.add_argument(
        "--num_neighbors",
        type=int,
        default=5,
        help="Number of neighbors to return for nearest neighbor search",
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="/home/congtj/openforge/logs",
        help="Path to the log directory",
    )

    args = parser.parse_args()

    arts_output_id = args.arts_output_filepath.split("/")[-1]
    if arts_output_id.endswith(".pickle"):
        arts_output_id = arts_output_id[: -len(".pickle")]

    log_dir = os.path.join(
        args.log_dir,
        f"{arts_output_id}_num_head_concepts-{args.num_head_nodes}_knn_search",
    )

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = create_custom_logger(log_dir)

    with open(args.arts_output_filepath, "rb") as f:
        data = pickle.load(f)
        nodeByLevel = data["nodeByLevel"]
        OntologyNode.init__device_and_threshold(
            device=data["device"], threshold=data["threshold"]
        )
        OntologyNode.embeddingByLevelAndIdx = data["embeddings"]

    nodeByLevel[args.arts_level].sort(
        key=lambda x: len(x.tbl_column_matched), reverse=True
    )
    X = []

    for i, node in enumerate(
        nodeByLevel[args.arts_level][: args.num_head_nodes]
    ):
        node_embedding = node.get_embedding().detach().cpu().numpy()
        X.append(node_embedding)

    knn = NearestNeighbors(n_neighbors=args.num_neighbors, metric="cosine")
    knn.fit(X)

    for i, node in enumerate(
        nodeByLevel[args.arts_level][: args.num_head_nodes]
    ):
        logger.info("=" * 50)
        logger.info(f"Query node: {node}")
        logger.info(f"Merged concepts: {node.texts}")

        node_embedding = node.get_embedding().detach().cpu().numpy()
        dist, idx = knn.kneighbors([node_embedding], return_distance=True)

        for k in range(args.num_neighbors):
            logger.info("-" * 30)

            neighbor_node = nodeByLevel[args.arts_level][idx[0][k]]

            logger.info(f"Neighbor node: {neighbor_node}")
            logger.info(f"Merged concepts: {neighbor_node.texts}")
            logger.info(f"Cosine similarity: {1 - dist[0][k]:.2f}")
