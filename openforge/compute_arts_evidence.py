import argparse
import csv
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
from openforge.utils.util import create_dir, get_proj_dir


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
        default=100,
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

    data_dir = os.path.join(
        proj_dir, f"data/arts_top-{args.num_head_concepts}-concepts_evidence"
    )
    create_dir(data_dir, force=True)

    log_dir = os.path.join(
        proj_dir, f"logs/arts_top-{args.num_head_concepts}-concepts_evidence"
    )
    create_dir(log_dir, force=True)

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

    csv_output_filepath = os.path.join(data_dir, "arts_concept_dataset.csv")
    csv_file = open(csv_output_filepath, "w")
    field_names = [
        "concept 1",
        "concept 2",
        "name similarity",
        "value similarity",
        "label",
    ]

    csv_writer = csv.DictWriter(csv_file, fieldnames=field_names)
    csv_writer.writeheader()

    for i, node_i in enumerate(
        nodeByLevel[args.arts_level][: args.num_head_concepts]
    ):
        # node_i is the head concept
        assert str(node_i) == node_i.texts[0]
        # each merged concept has at least one corresponding table column
        assert len(node_i.texts) == len(node_i.text_to_tbl_column_matched)

        logger.info("=" * 50)
        logger.info(f"Node {i}: {node_i}")

        # compute evidence between the reference concept and merged concepts
        # (positive instances)
        for j, reference_concept in enumerate(node_i.texts):
            reference_concept_name_signature = set(
                qgram_transformer.transform(reference_concept)
            )
            reference_concept_fasttext_signature = compute_fasttext_signature(
                node_i.text_to_tbl_column_matched[reference_concept],
                fasttext_transformer,
                args.num_val_samples,
            )

            if len(reference_concept_fasttext_signature) == 0:
                log_msg = (
                    "Cannot compute value signature for reference concept: "
                    f"{reference_concept}."
                )
                logger.info(log_msg)
                continue

            for merged_concept in node_i.texts[j + 1 :]:
                logger.info(
                    f"Concept pair: {reference_concept} and {merged_concept}"
                )

                merged_concept_name_signature = set(
                    qgram_transformer.transform(merged_concept)
                )
                merged_concept_fasttext_signature = compute_fasttext_signature(
                    node_i.text_to_tbl_column_matched[merged_concept],
                    fasttext_transformer,
                    args.num_val_samples,
                )

                if len(merged_concept_fasttext_signature) == 0:
                    log_msg = (
                        "Cannot compute value signature for merged concept: "
                        f"{merged_concept}."
                    )
                    logger.info(log_msg)
                    continue

                # compute name similarity
                name_sim = jaccard_index(
                    reference_concept_name_signature,
                    merged_concept_name_signature,
                )

                # compute value similarity
                value_sim = cosine_similarity(
                    reference_concept_fasttext_signature,
                    merged_concept_fasttext_signature,
                )

                evidence_data.append(([name_sim, value_sim], 1))
                csv_writer.writerow(
                    {
                        "concept 1": reference_concept,
                        "concept 2": merged_concept,
                        "name similarity": name_sim,
                        "value similarity": value_sim,
                        "label": 1,
                    }
                )

            # compute evidence between the reference concept and subsequent
            # unmerged concepts (negative instances)
            for subsequent_node in nodeByLevel[args.arts_level][
                i + 1 : args.num_head_concepts
            ]:
                for unmerged_concept in subsequent_node.texts:
                    log_msg = (
                        f"Concept pair: {reference_concept} and "
                        f"{unmerged_concept}."
                    )
                    logger.info(log_msg)

                    unmerged_concept_name_signature = set(
                        qgram_transformer.transform(unmerged_concept)
                    )
                    unmerged_concept_fasttext_signature = (
                        compute_fasttext_signature(
                            subsequent_node.text_to_tbl_column_matched[
                                unmerged_concept
                            ],
                            fasttext_transformer,
                            args.num_val_samples,
                        )
                    )

                    if len(unmerged_concept_fasttext_signature) == 0:
                        log_msg = (
                            "Cannot compute value signature for unmerged "
                            f"concept: {unmerged_concept}."
                        )
                        logger.info(log_msg)
                        continue

                    # compute name similarity
                    name_sim = jaccard_index(
                        reference_concept_name_signature,
                        unmerged_concept_name_signature,
                    )

                    # compute value similarity
                    value_sim = cosine_similarity(
                        reference_concept_fasttext_signature,
                        unmerged_concept_fasttext_signature,
                    )

                    evidence_data.append(([name_sim, value_sim], 0))
                    csv_writer.writerow(
                        {
                            "concept 1": reference_concept,
                            "concept 2": unmerged_concept,
                            "name similarity": name_sim,
                            "value similarity": value_sim,
                            "label": 0,
                        }
                    )

    csv_file.close()

    evidence_save_filepath = os.path.join(data_dir, "arts_evidence.pkl")
    with open(evidence_save_filepath, "wb") as f:
        pickle.dump(evidence_data, f)
