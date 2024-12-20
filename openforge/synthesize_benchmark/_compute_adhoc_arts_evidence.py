import argparse
import os
import pickle
import random

from openforge.ARTS.ontology import OntologyNode
from openforge.feature_extraction.fb_fasttext import (
    FasttextTransformer,
    compute_fasttext_signature,
)
from openforge.feature_extraction.qgram import QGramTransformer
from openforge.feature_extraction.similarity_metrics import (
    cosine_similarity,
    jaccard_index,
)
from openforge.utils.custom_logging import create_custom_logger
from openforge.utils.util import get_proj_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--arts_output_filepath",
        type=str,
        default="/home/congtj/openforge/data/\
column_semantics_ontology_threshold_0.9_run_nyc_gpt_3.5_merge_root.pickle",
        help="Path to the ARTS output file.",
    )

    parser.add_argument(
        "--arts_level",
        type=int,
        default=2,
        help="Level of the ARTS ontology to extract concepts.",
    )

    parser.add_argument(
        "--num_head_concepts",
        type=int,
        default=3,
        help="Number of head concepts to consider.",
    )

    parser.add_argument(
        "--fasttext_model_dir",
        type=str,
        default="/ssd/congtj",
        help="Directory containing fasttext model weights.",
    )

    parser.add_argument(
        "--num_val_samples",
        type=int,
        default=10000,
        help="Maximum number of values per column for computing features.",
    )

    parser.add_argument(
        "--log_level", type=str, default="INFO", help="Logging level."
    )

    parser.add_argument(
        "--random_seed", type=int, default=12345, help="Random seed."
    )

    args = parser.parse_args()

    # fix random seed
    random.seed(args.random_seed)

    # create logging directory
    proj_dir = get_proj_dir(__file__, file_level=2)
    log_dir = os.path.join(
        proj_dir,
        f"logs/arts_top-{args.num_head_concepts}-concepts_adhoc-evidence",
    )
    logger = create_custom_logger(log_dir, args.log_level)
    logger.info(args)

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

    qgram_transformer = QGramTransformer(qgram_size=3)
    fasttext_transformer = FasttextTransformer(
        cache_dir=args.fasttext_model_dir
    )
    evidence_data = []

    reference_node = nodeByLevel[args.arts_level][0]
    reference_node_name = reference_node.texts[2]

    reference_node_name_signature = set(
        qgram_transformer.transform(reference_node_name)
    )
    reference_node_fasttext_signature = compute_fasttext_signature(
        reference_node.text_to_tbl_column_matched[reference_node_name],
        fasttext_transformer,
        args.num_val_samples,
    )

    for i, node_i in enumerate(
        nodeByLevel[args.arts_level][: args.num_head_concepts]
    ):
        # node is the head concept
        assert str(node_i) == node_i.texts[0]
        # each merged concept has at least one corresponding table column
        assert len(node_i.texts) == len(node_i.text_to_tbl_column_matched)

        logger.info("=" * 50)
        logger.info(f"Concept {i}: {node_i}")
        logger.info(f"Merged concepts: {node_i.texts}")

        # use head concept as a reference point
        i_name_signature = set(qgram_transformer.transform(str(node_i)))
        i_fasttext_signature = compute_fasttext_signature(
            node_i.text_to_tbl_column_matched[str(node_i)],
            fasttext_transformer,
            args.num_val_samples,
        )

        if len(i_fasttext_signature) == 0:
            logger.info(
                f"Cannot compute value signature for concept {i}: {node_i}."
            )
            continue

        # compute name similarity
        name_sim = jaccard_index(
            i_name_signature, reference_node_name_signature
        )

        # compute value similarity
        value_sim = cosine_similarity(
            i_fasttext_signature, reference_node_fasttext_signature
        )

        logger.info(f"Name similarity: {name_sim}")
        logger.info(f"Value similarity: {value_sim}")

        evidence_data.append(([name_sim, value_sim], 1))

    evidence_save_filepath = os.path.join(
        proj_dir,
        f"data/arts_top-{args.num_head_concepts}-concepts_adhoc-evidence.pkl",
    )

    with open(evidence_save_filepath, "wb") as f:
        pickle.dump(evidence_data, f)
