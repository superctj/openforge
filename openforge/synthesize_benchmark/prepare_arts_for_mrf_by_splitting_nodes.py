import argparse
import os
import pickle
import random

from dataclasses import make_dataclass

import pandas as pd

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


MRFEntry = make_dataclass(
    "MRFEntry",
    [
        ("concept_1", str),
        ("concept_2", str),
        ("name_similarity", float),
        ("value_similarity", float),
        ("relation_variable_name", str),
        ("relation_variable_label", int),
    ],
)


def create_pos_and_neg_instances(
    arts_nodes: list, qgram_transformer, fasttext_transformer, logger, args
):
    global_concept_id = 1  # start from 1
    mrf_data = []

    for i, node_i in enumerate(arts_nodes):
        # 'node_i' is the head concept
        assert str(node_i) == node_i.texts[0]
        # Each merged concept has at least one corresponding table column
        assert len(node_i.texts) == len(node_i.text_to_tbl_column_matched)

        logger.info("=" * 50)
        logger.info(f"Node {i}: {node_i}")

        # Compute evidence between the reference concept and merged concepts
        # (positive instances)
        for j, reference_concept in enumerate(node_i.texts):
            reference_concept_id = global_concept_id + j

            log_msg = (
                "Reference concept (with global concept id "
                f"{reference_concept_id}): {reference_concept}"
            )
            logger.info(log_msg)

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
                    "Cannot compute fasttext signature for reference concept: "
                    f"{reference_concept}"
                )
                logger.info(log_msg)
                continue

            for k, merged_concept in enumerate(node_i.texts[j + 1 :]):
                merged_concept_id = reference_concept_id + k + 1

                log_msg = (
                    "Concept pair (with global concept id "
                    f"{reference_concept_id} and {merged_concept_id}): "
                    f"{reference_concept} and {merged_concept}"
                )
                logger.info(log_msg)

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
                        "Cannot compute fasttext signature for merged concept: "
                        f"{merged_concept}"
                    )
                    logger.info(log_msg)
                    continue

                # Compute name similarity
                name_sim = jaccard_index(
                    reference_concept_name_signature,
                    merged_concept_name_signature,
                )

                # Compute fasttext similarity
                fasttext_sim = cosine_similarity(
                    reference_concept_fasttext_signature,
                    merged_concept_fasttext_signature,
                )

                relation_variable_name = (
                    f"R_{reference_concept_id}-{merged_concept_id}"
                )

                mrf_data.append(
                    MRFEntry(
                        reference_concept,
                        merged_concept,
                        name_sim,
                        fasttext_sim,
                        relation_variable_name,
                        1,
                    )
                )

            local_concept_id = global_concept_id + len(node_i.texts)
            unmerged_concept_id = local_concept_id

            # Compute evidence between the reference concept and subsequent
            # unmerged concepts (negative instances)
            for subsequent_node in arts_nodes[i + 1 :]:
                for unmerged_concept in subsequent_node.texts:
                    log_msg = (
                        "Concept pair (with global concept id "
                        f"{reference_concept_id} and {unmerged_concept_id}): "
                        f"{reference_concept} and {unmerged_concept}"
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
                            f"concept: {unmerged_concept}"
                        )
                        logger.info(log_msg)
                        continue

                    # compute name similarity
                    name_sim = jaccard_index(
                        reference_concept_name_signature,
                        unmerged_concept_name_signature,
                    )

                    # compute value similarity
                    fasttext_sim = cosine_similarity(
                        reference_concept_fasttext_signature,
                        unmerged_concept_fasttext_signature,
                    )

                    relation_variable_name = (
                        f"R_{reference_concept_id}-{unmerged_concept_id}"
                    )

                    mrf_data.append(
                        MRFEntry(
                            reference_concept,
                            unmerged_concept,
                            name_sim,
                            fasttext_sim,
                            relation_variable_name,
                            0,
                        )
                    )

                    unmerged_concept_id += 1

        global_concept_id += len(node_i.texts)

    return mrf_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--arts_data_filepath",
        type=str,
        default="/home/congtj/openforge/data/\
column_semantics_ontology_threshold_0.9_run_nyc_gpt_3.5_merge_root.pickle",
        help="Path to the ARTS source data.",
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
        default=40,
        help="Number of head nodes to consider.",
    )

    parser.add_argument(
        "--train_prop",
        type=float,
        default=0.6,
        help="Training proportion of the ARTS data.",
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
    instance_name = f"arts-context_top-{args.num_head_nodes}-nodes"

    data_save_dir = os.path.join(proj_dir, f"exps/{instance_name}")
    create_dir(data_save_dir, force=True)

    log_dir = os.path.join(proj_dir, f"logs/{instance_name}")
    create_dir(log_dir, force=True)

    logger = create_custom_logger(log_dir, args.log_level)
    logger.info(args)

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

    qgram_transformer = QGramTransformer(qgram_size=3)
    fasttext_transformer = FasttextTransformer(
        cache_dir=args.fasttext_model_dir
    )

    mrf_data = {"train": [], "valid": [], "test": []}

    # Split by ARTS nodes
    base_num_nodes = 20
    assert args.num_head_nodes >= base_num_nodes

    base_indices = list(range(base_num_nodes))
    train_indices = random.sample(
        population=base_indices, k=int(args.train_prop * base_num_nodes)
    )

    valid_test_indices = list(set(base_indices) - set(train_indices))
    valid_indices = random.sample(
        population=valid_test_indices, k=len(valid_test_indices) // 2
    )

    test_indices = list(set(valid_test_indices) - set(valid_indices))

    if args.num_head_nodes > base_num_nodes:
        extra_indices = list(range(base_num_nodes, args.num_head_nodes))

        extra_train_indices = random.sample(
            population=extra_indices,
            k=int(args.train_prop * len(extra_indices)),
        )
        train_indices.extend(extra_train_indices)
        logger.info(f"Train indices: {train_indices}")

        extra_valid_test_indices = list(
            set(extra_indices) - set(extra_train_indices)
        )

        extra_valid_indices = random.sample(
            population=extra_valid_test_indices,
            k=len(extra_valid_test_indices) // 2,
        )
        valid_indices.extend(extra_valid_indices)
        logger.info(f"Valid indices: {valid_indices}")

        extra_test_indices = list(
            set(extra_valid_test_indices) - set(extra_valid_indices)
        )
        test_indices.extend(extra_test_indices)
        logger.info(f"Test indices: {test_indices}")

    train_nodes = [nodeByLevel[args.arts_level][i] for i in train_indices]
    valid_nodes = [nodeByLevel[args.arts_level][i] for i in valid_indices]
    test_nodes = [nodeByLevel[args.arts_level][i] for i in test_indices]

    logger.info("Creating instances for training split...")
    mrf_data["train"] = create_pos_and_neg_instances(
        train_nodes, qgram_transformer, fasttext_transformer, logger, args
    )

    logger.info("Creating instances for validation split...")
    mrf_data["valid"] = create_pos_and_neg_instances(
        valid_nodes, qgram_transformer, fasttext_transformer, logger, args
    )

    logger.info("Creating instances for test split...")
    mrf_data["test"] = create_pos_and_neg_instances(
        test_nodes, qgram_transformer, fasttext_transformer, logger, args
    )

    for split in mrf_data:
        output_filepath = os.path.join(
            data_save_dir, f"arts_mrf_data_{split}.csv"
        )

        mrf_df = pd.DataFrame(mrf_data[split])
        mrf_df.to_csv(output_filepath, index=False)

        logger.info(f"Saved MRF data for {split} split.")
