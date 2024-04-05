import argparse
import os
import pickle
import random

from dataclasses import make_dataclass

import pandas as pd

from openforge.ARTS.helpers.mongodb_helper import readCSVFileWithTableID
from openforge.ARTS.ontology import OntologyNode
from openforge.feature_extraction.fb_fasttext import FasttextTransformer
from openforge.feature_extraction.qgram import QGramTransformer
from openforge.feature_extraction.similarity_metrics import (
    cosine_similarity,
    edit_distance,
    jaccard_index,
)
from openforge.utils.custom_logging import create_custom_logger


VALUE_SIGNATURE_ATTEMPTS = 100

MRFEntry = make_dataclass(
    "MRFEntry",
    [
        ("concept_1", str),
        ("concept_2", str),
        ("concept_1_table_id", str),
        ("concept_2_table_id", str),
        ("concept_1_col_name", str),
        ("concept_2_col_name", str),
        ("name_qgram_similarity", float),
        ("name_jaccard_similarity", float),
        ("name_edit_distance", int),
        ("name_fasttext_similarity", float),
        ("name_word_count_ratio", float),
        ("name_char_count_ratio", float),
        ("value_jaccard_similarity", float),
        ("value_fasttext_similarity", float),
        ("relation_variable_name", str),
        ("relation_variable_label", int),
    ],
)


def get_value_fasttext_signature(
    col_list, fasttext_model, num_val_samples: int
):
    # randomly pick a corresponding table column to compute value signature
    count = 0

    while count < VALUE_SIGNATURE_ATTEMPTS:
        rnd_idx = random.randrange(len(col_list))
        table_id, col_name = col_list[rnd_idx]

        df = readCSVFileWithTableID(
            table_id, usecols=[col_name], nrows=num_val_samples
        ).astype(str)
        column_values = df[col_name].tolist()
        fasttext_signature = fasttext_model.transform(column_values)

        if len(fasttext_signature) != 0:
            break
        else:
            count += 1

    return fasttext_signature, column_values, table_id, col_name


def get_concept_signatures(
    concept: str,
    qgram_model,
    fasttext_model,
    tblid_colname_pairs: list,
    num_val_samples: int,
):
    name_qgram_signature = set(qgram_model.transform(concept))
    name_fasttext_signature = fasttext_model.transform([concept])

    col_values, value_fasttext_signature, table_id, col_name = (
        get_value_fasttext_signature(
            tblid_colname_pairs, fasttext_model, num_val_samples
        )
    )

    return (
        name_qgram_signature,
        name_fasttext_signature,
        value_fasttext_signature,
        col_values,
        table_id,
        col_name,
    )


def create_pos_and_neg_instances(
    arts_nodes: list, qgram_model, fasttext_model, logger, args
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

            logger.info(
                "Reference concept (with global concept id "
                f"{reference_concept_id}): {reference_concept}"
            )

            (
                reference_concept_name_qgram_signature,
                reference_concept_name_fasttext_signature,
                reference_concept_value_fasttext_signature,
                reference_concept_col_values,
                reference_concept_table_id,
                reference_concept_col_name,
            ) = get_concept_signatures(
                reference_concept,
                qgram_model,
                fasttext_model,
                node_i.text_to_tbl_column_matched[reference_concept],
                args.num_val_samples,
            )

            if len(reference_concept_value_fasttext_signature) == 0:
                logger.info(
                    "Cannot compute fasttext signature for reference concept: "
                    f"{reference_concept}"
                )
                continue

            for k, merged_concept in enumerate(node_i.texts[j + 1 :]):
                merged_concept_id = reference_concept_id + k + 1

                logger.info(
                    "Concept pair (with global concept id "
                    f"{reference_concept_id} and {merged_concept_id}): "
                    f"{reference_concept} and {merged_concept}"
                )

                (
                    merged_concept_name_qgram_signature,
                    merged_concept_name_fasttext_signature,
                    merged_concept_value_fasttext_signature,
                    merged_concept_col_values,
                    merged_concept_table_id,
                    merged_concept_col_name,
                ) = get_concept_signatures(
                    merged_concept,
                    qgram_model,
                    fasttext_model,
                    node_i.text_to_tbl_column_matched[merged_concept],
                    args.num_val_samples,
                )

                if len(merged_concept_value_fasttext_signature) == 0:
                    logger.info(
                        "Cannot compute fasttext signature for merged concept: "
                        f"{merged_concept}"
                    )
                    continue

                name_qgram_sim = jaccard_index(
                    reference_concept_name_qgram_signature,
                    merged_concept_name_qgram_signature,
                )
                name_jaccard_sim = jaccard_index(
                    set(reference_concept.split()), set(merged_concept.split())
                )
                name_edit_dist = edit_distance(
                    reference_concept, merged_concept
                )
                name_fasttext_sim = cosine_similarity(
                    reference_concept_name_fasttext_signature,
                    merged_concept_name_fasttext_signature,
                )
                name_word_count_ratio = len(reference_concept.split()) / len(
                    merged_concept.split()
                )
                name_char_count_ratio = len(reference_concept) / len(
                    merged_concept
                )

                value_jaccard_sim = jaccard_index(
                    set(reference_concept_col_values),
                    set(merged_concept_col_values),
                )
                value_fasttext_sim = cosine_similarity(
                    reference_concept_value_fasttext_signature,
                    merged_concept_value_fasttext_signature,
                )

                relation_variable_name = (
                    f"R_{reference_concept_id}-{merged_concept_id}"
                )

                mrf_data.append(
                    MRFEntry(
                        reference_concept,
                        merged_concept,
                        reference_concept_table_id,
                        merged_concept_table_id,
                        reference_concept_col_name,
                        merged_concept_col_name,
                        name_qgram_sim,
                        name_jaccard_sim,
                        name_edit_dist,
                        name_fasttext_sim,
                        name_word_count_ratio,
                        name_char_count_ratio,
                        value_jaccard_sim,
                        value_fasttext_sim,
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
                    logger.info(
                        "Concept pair (with global concept id "
                        f"{reference_concept_id} and {unmerged_concept_id}): "
                        f"{reference_concept} and {unmerged_concept}"
                    )

                    # unmerged_concept_name_signature = set(
                    #     qgram_transformer.transform(unmerged_concept)
                    # )
                    # unmerged_concept_fasttext_signature = (
                    #     compute_fasttext_signature(
                    #         subsequent_node.text_to_tbl_column_matched[
                    #             unmerged_concept
                    #         ],
                    #         fasttext_transformer,
                    #         args.num_val_samples,
                    #     )
                    # )

                    (
                        unmerged_concept_name_qgram_signature,
                        unmerged_concept_name_fasttext_signature,
                        unmerged_concept_value_fasttext_signature,
                        unmerged_concept_col_values,
                        unmerged_concept_table_id,
                        unmerged_concept_col_name,
                    ) = get_concept_signatures(
                        unmerged_concept,
                        qgram_model,
                        fasttext_model,
                        subsequent_node.text_to_tbl_column_matched[
                            unmerged_concept
                        ],
                        args.num_val_samples,
                    )

                    if len(unmerged_concept_value_fasttext_signature) == 0:
                        logger.info(
                            "Cannot compute value signature for unmerged "
                            f"concept: {unmerged_concept}"
                        )
                        continue

                    name_qgram_sim = jaccard_index(
                        reference_concept_name_qgram_signature,
                        unmerged_concept_name_qgram_signature,
                    )

                    name_jaccard_sim = jaccard_index(
                        set(reference_concept.split()),
                        set(unmerged_concept.split()),
                    )

                    name_edit_dist = edit_distance(
                        reference_concept, unmerged_concept
                    )

                    name_fasttext_sim = cosine_similarity(
                        reference_concept_name_fasttext_signature,
                        unmerged_concept_name_fasttext_signature,
                    )

                    name_word_count_ratio = len(
                        reference_concept.split()
                    ) / len(unmerged_concept.split())

                    name_char_count_ratio = len(reference_concept) / len(
                        unmerged_concept
                    )

                    value_jaccard_sim = jaccard_index(
                        set(reference_concept_col_values),
                        set(unmerged_concept_col_values),
                    )

                    value_fasttext_sim = cosine_similarity(
                        reference_concept_value_fasttext_signature,
                        unmerged_concept_value_fasttext_signature,
                    )

                    relation_variable_name = (
                        f"R_{reference_concept_id}-{unmerged_concept_id}"
                    )

                    mrf_data.append(
                        MRFEntry(
                            reference_concept,
                            unmerged_concept,
                            reference_concept_table_id,
                            unmerged_concept_table_id,
                            reference_concept_col_name,
                            unmerged_concept_col_name,
                            name_qgram_sim,
                            name_jaccard_sim,
                            name_edit_dist,
                            name_fasttext_sim,
                            name_word_count_ratio,
                            name_char_count_ratio,
                            value_jaccard_sim,
                            value_fasttext_sim,
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
        default="/ssd/congtj/openforge/arts/column_semantics_ontology_threshold_0.9_run_nyc_gpt_3.5_merge_root.pickle",  # noqa: E501
        help="Path to ARTS source data.",
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
        default=0.5,
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
        "--output_dir",
        type=str,
        default="/ssd/congtj/openforge/arts/artifact",
        help="Directory to save outputs.",
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="/home/congtj/openforge/logs/arts",
        help="Directory to save logs.",
    )

    parser.add_argument(
        "--random_seed", type=int, default=12345, help="Random seed."
    )

    args = parser.parse_args()

    # fix random seed
    random.seed(args.random_seed)

    instance_name = os.path.join(
        f"top_{args.num_head_nodes}_nodes", f"training_prop_{args.train_prop})"
    )
    output_dir = os.path.join(args.output_dir, instance_name)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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

    qgram_model = QGramTransformer(qgram_size=3)
    fasttext_model = FasttextTransformer(cache_dir=args.fasttext_model_dir)

    mrf_data = {"training": [], "validation": [], "test": []}

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
    mrf_data["training"] = create_pos_and_neg_instances(
        train_nodes, qgram_model, fasttext_model, logger, args
    )

    logger.info("Creating instances for validation split...")
    mrf_data["validation"] = create_pos_and_neg_instances(
        valid_nodes, qgram_model, fasttext_model, logger, args
    )

    logger.info("Creating instances for test split...")
    mrf_data["test"] = create_pos_and_neg_instances(
        test_nodes, qgram_model, fasttext_model, logger, args
    )

    for split in mrf_data:
        output_filepath = os.path.join(
            output_dir, f"openforge_arts_{split}.csv"
        )

        mrf_df = pd.DataFrame(mrf_data[split])
        mrf_df.to_csv(output_filepath, index=False)

        logger.info(f"Saved MRF data for {split} split.")
