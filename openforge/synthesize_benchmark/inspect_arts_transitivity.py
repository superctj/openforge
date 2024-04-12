import argparse
import os
import pickle
import random

from openforge.ARTS.ontology import OntologyNode
from openforge.utils.custom_logging import create_custom_logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--arts_data_filepath",
        type=str,
        default="/ssd/congtj/openforge/arts/column_semantics_ontology_threshold_0.9_run_nyc_gpt_3.5_merge_root.pickle",  # noqa: E501
        help="Path to ARTS source data.",
    )

    parser.add_argument(
        "--arts_level",
        type=int,
        default=1,
        help="Starting level of the ARTS ontology to extract concepts.",
    )

    parser.add_argument(
        "--num_head_nodes",
        type=int,
        default=11,
        help="Number of head nodes to consider.",
    )

    # parser.add_argument(
    #     "--train_prop",
    #     type=float,
    #     default=0.5,
    #     help="Training proportion of the ARTS data.",
    # )

    # parser.add_argument(
    #     "--fasttext_model_dir",
    #     type=str,
    #     default="/ssd/congtj",
    #     help="Directory containing fasttext model weights.",
    # )

    # parser.add_argument(
    #     "--num_val_samples",
    #     type=int,
    #     default=10000,
    #     help="Maximum number of values per column for computing features.",
    # )

    # parser.add_argument(
    #     "--output_dir",
    #     type=str,
    #     default="/ssd/congtj/openforge/arts/artifact",
    #     help="Directory to save outputs.",
    # )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="/home/congtj/openforge/logs/arts_multi_relations",
        help="Directory to save logs.",
    )

    parser.add_argument(
        "--random_seed", type=int, default=12345, help="Random seed."
    )

    args = parser.parse_args()

    # fix random seed
    random.seed(args.random_seed)

    # instance_name = os.path.join(
    #     f"multi_relations_top_{args.num_head_nodes}_nodes",
    #     f"training_prop_{args.train_prop}",
    # )
    # output_dir = os.path.join(args.output_dir, instance_name)

    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    logger = create_custom_logger(args.log_dir)
    logger.info(f"Running program: {__file__}")
    logger.info(f"{args}\n")

    with open(args.arts_data_filepath, "rb") as f:
        data = pickle.load(f)
        nodeByLevel = data["nodeByLevel"]
        OntologyNode.init__device_and_threshold(
            device=data["device"], threshold=data["threshold"]
        )
        OntologyNode.embeddingByLevelAndIdx = data["embeddings"]

    nodeByLevel[args.arts_level].sort(
        key=lambda x: len(x.tbl_column_matched), reverse=True
    )

    for level_1_node in nodeByLevel[args.arts_level][: args.num_head_nodes]:
        assert str(level_1_node) == level_1_node.texts[0]
        logger.info(f"\nlevel_1 node: {str(level_1_node)}")

        if len(level_1_node.texts) > 2:
            logger.info(f"\nlevel_1 concepts: {level_1_node.texts}")

        for level_2_node in level_1_node.children:
            assert str(level_2_node) == level_2_node.texts[0]
            logger.info(f"\nlevel_2 node: {str(level_2_node)}")

            if len(level_2_node.texts) > 2:
                logger.info(f"\nlevel_2 concepts: {level_2_node.texts}")

            for level_3_node in level_2_node.children:
                assert str(level_3_node) == level_3_node.texts[0]
                logger.info(f"\nlevel_3 node: {str(level_3_node)}")

                if len(level_3_node.texts) > 2:
                    logger.info(f"\nlevel_3 concepts: {level_3_node.texts}")
