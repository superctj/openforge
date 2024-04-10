import argparse
import os
import pickle
import random

from dataclasses import make_dataclass
from enum import Enum

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


class RelationType(Enum):
    NULL = 0
    EQUIV = 1  # Equivalent
    HYPER = 2  # Hypernymy
    HYPO = 3  # Hyponymy


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


def collect_concept_relations(split_nodes: list, only_hypernymy: bool = False):
    concepts = set()
    conceptpair_relation_map = {}
    concept_collist_map = {}

    for level_1_node in split_nodes:
        assert str(level_1_node) == level_1_node.texts[0]
        logger.info(f"\nLevel 1 node: {str(level_1_node)}")

        for i, level_1_concept in enumerate(level_1_node.texts):
            # Collect equivalent relations
            if not only_hypernymy:
                for level_1_merged_concept in level_1_node.texts[i + 1 :]:
                    logger.info(
                        f"Equivalent concepts: ({level_1_concept}, "
                        f"{level_1_merged_concept})"
                    )

                    concepts.add(level_1_concept)
                    concepts.add(level_1_merged_concept)

                    conceptpair_relation_map[
                        (level_1_concept, level_1_merged_concept)
                    ] = RelationType.EQUIV

                    if level_1_concept not in concept_collist_map:
                        concept_collist_map[level_1_concept] = (
                            level_1_node.text_to_tbl_column_matched[
                                level_1_concept
                            ]
                        )

                    if level_1_merged_concept not in concept_collist_map:
                        concept_collist_map[level_1_merged_concept] = (
                            level_1_node.text_to_tbl_column_matched[
                                level_1_merged_concept
                            ]
                        )

            for level_2_node in level_1_node.children[:10]:
                assert str(level_2_node) == level_2_node.texts[0]

                logger.info(f"Level 2 node: {str(level_2_node)}")
                # logger.info(f"Level 2 merged concepts: {level_2_node.texts}")

                # Collect equivalent relations
                if not only_hypernymy:
                    for j, level_2_concept in enumerate(level_2_node.texts):
                        for level_2_merged_concept in level_2_node.texts[
                            j + 1 :
                        ]:
                            logger.info(
                                f"Equivalent concepts: ({level_2_concept}, "
                                f"{level_2_merged_concept})"
                            )

                            concepts.add(level_2_concept)
                            concepts.add(level_2_merged_concept)

                            conceptpair_relation_map[
                                (level_2_concept, level_2_merged_concept)
                            ] = RelationType.EQUIV

                            if level_2_concept not in concept_collist_map:
                                concept_collist_map[level_2_concept] = (
                                    level_2_node.text_to_tbl_column_matched[
                                        level_2_concept
                                    ]
                                )

                        if level_2_merged_concept not in concept_collist_map:
                            concept_collist_map[level_2_merged_concept] = (
                                level_2_node.text_to_tbl_column_matched[
                                    level_2_merged_concept
                                ]
                            )

                # Collect hypernymy relations
                for level_2_concept in level_2_node.texts:
                    logger.info(
                        f"Hypernymy concepts: ({level_1_concept}, "
                        f"{level_2_concept})"
                    )

                    concepts.add(level_1_concept)
                    concepts.add(level_2_concept)

                    conceptpair_relation_map[
                        (level_1_concept, level_2_concept)
                    ] = RelationType.HYPER

                    if level_1_concept not in concept_collist_map:
                        concept_collist_map[level_1_concept] = (
                            level_1_node.text_to_tbl_column_matched[
                                level_1_concept
                            ]
                        )

                    if level_2_concept not in concept_collist_map:
                        concept_collist_map[level_2_concept] = (
                            level_2_node.text_to_tbl_column_matched[
                                level_2_concept
                            ]
                        )

    concepts = list(concepts)
    logger.info(f"Number of concepts: {len(concepts)}")

    return concepts, conceptpair_relation_map, concept_collist_map


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

    value_fasttext_signature, col_values, table_id, col_name = (
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


def create_relation_instances(
    concepts: list,
    conceptpair_relation_map: dict,
    concept_collist_map: dict,
    qgram_model,
    fasttext_model,
    logger,
    args,
):
    mrf_data = []

    for i, concept_i in enumerate(concepts):
        (
            concept_i_name_qgram_signature,
            concept_i_name_fasttext_signature,
            concept_i_value_fasttext_signature,
            concept_i_col_values,
            concept_i_table_id,
            concept_i_col_name,
        ) = get_concept_signatures(
            concept_i,
            qgram_model,
            fasttext_model,
            concept_collist_map[concept_i],
            args.num_val_samples,
        )

        if len(concept_i_value_fasttext_signature) == 0:
            logger.info(
                "Cannot compute fasttext signature for concept i: "
                f"{concept_i}"
            )
            continue

        for j, concept_j in enumerate(concepts[i + 1 :]):
            (
                concept_j_name_qgram_signature,
                concept_j_name_fasttext_signature,
                concept_j_value_fasttext_signature,
                concept_j_col_values,
                concept_j_table_id,
                concept_j_col_name,
            ) = get_concept_signatures(
                concept_j,
                qgram_model,
                fasttext_model,
                concept_collist_map[concept_j],
                args.num_val_samples,
            )

            if len(concept_j_value_fasttext_signature) == 0:
                logger.info(
                    "Cannot compute fasttext signature for concept j: "
                    f"{concept_j}"
                )
                continue

            name_qgram_sim = jaccard_index(
                concept_i_name_qgram_signature,
                concept_j_name_qgram_signature,
            )
            name_jaccard_sim = jaccard_index(
                set(concept_i.split()), set(concept_j.split())
            )
            name_edit_dist = edit_distance(concept_i, concept_j)
            name_fasttext_sim = cosine_similarity(
                concept_i_name_fasttext_signature,
                concept_j_name_fasttext_signature,
            )
            name_word_count_ratio = len(concept_i.split()) / len(
                concept_j.split()
            )
            name_char_count_ratio = len(concept_i) / len(concept_j)

            value_jaccard_sim = jaccard_index(
                set(concept_i_col_values),
                set(concept_j_col_values),
            )
            value_fasttext_sim = cosine_similarity(
                concept_i_value_fasttext_signature,
                concept_j_value_fasttext_signature,
            )

            relation_variable_name = f"R_{i+1}-{j+1}"
            if (concept_i, concept_j) in conceptpair_relation_map:
                relation_variable_label = conceptpair_relation_map[
                    (concept_i, concept_j)
                ].value
            else:
                relation_variable_label = 0

            mrf_data.append(
                MRFEntry(
                    concept_i,
                    concept_j,
                    concept_i_table_id,
                    concept_j_table_id,
                    concept_i_col_name,
                    concept_j_col_name,
                    name_qgram_sim,
                    name_jaccard_sim,
                    name_edit_dist,
                    name_fasttext_sim,
                    name_word_count_ratio,
                    name_char_count_ratio,
                    value_jaccard_sim,
                    value_fasttext_sim,
                    relation_variable_name,
                    relation_variable_label,
                )
            )

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
        "--only_hypernymy",
        type=int,
        default=1,
        help="Whether only collects hypernymy and empty relations.",
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
        default="/home/congtj/openforge/logs/arts_multi_relations",
        help="Directory to save logs.",
    )

    parser.add_argument(
        "--random_seed", type=int, default=12345, help="Random seed."
    )

    args = parser.parse_args()

    # fix random seed
    random.seed(args.random_seed)

    if bool(args.only_hypernymy):
        instance_name = os.path.join(
            f"hypernymy_top_{args.num_head_nodes}_nodes",
            f"training_prop_{args.train_prop}",
        )
    else:
        instance_name = os.path.join(
            f"multi_relations_top_{args.num_head_nodes}_nodes",
            f"training_prop_{args.train_prop}",
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
    base_num_nodes = 11
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

        extra_valid_test_indices = list(
            set(extra_indices) - set(extra_train_indices)
        )

        extra_valid_indices = random.sample(
            population=extra_valid_test_indices,
            k=len(extra_valid_test_indices) // 2,
        )
        valid_indices.extend(extra_valid_indices)

        extra_test_indices = list(
            set(extra_valid_test_indices) - set(extra_valid_indices)
        )
        test_indices.extend(extra_test_indices)

    logger.info(f"Train indices: {train_indices}")
    logger.info(f"Valid indices: {valid_indices}")
    logger.info(f"Test indices: {test_indices}")

    train_nodes = [nodeByLevel[args.arts_level][i] for i in train_indices]
    valid_nodes = [nodeByLevel[args.arts_level][i] for i in valid_indices]
    test_nodes = [nodeByLevel[args.arts_level][i] for i in test_indices]

    logger.info("Collect concept relations for training split...")
    (
        train_concepts,
        train_conceptpair_relation_map,
        train_concept_collist_map,
    ) = collect_concept_relations(train_nodes, bool(args.only_hypernymy))

    logger.info("Collect concept relations for validation split...")
    (
        valid_concepts,
        valid_conceptpair_relation_map,
        valid_concept_collist_map,
    ) = collect_concept_relations(valid_nodes, bool(args.only_hypernymy))

    logger.info("Collect concept relations for test split...")
    test_concepts, test_conceptpair_relation_map, test_concept_collist_map = (
        collect_concept_relations(test_nodes, bool(args.only_hypernymy))
    )

    min_len = min(len(valid_concepts), len(test_concepts))
    valid_concepts = valid_concepts[:min_len]
    test_concepts = test_concepts[:min_len]

    logger.info("Creating instances for training split...")
    mrf_data["training"] = create_relation_instances(
        train_concepts,
        train_conceptpair_relation_map,
        train_concept_collist_map,
        qgram_model,
        fasttext_model,
        logger,
        args,
    )

    logger.info("Creating instances for validation split...")
    mrf_data["validation"] = create_relation_instances(
        valid_concepts,
        valid_conceptpair_relation_map,
        valid_concept_collist_map,
        qgram_model,
        fasttext_model,
        logger,
        args,
    )

    logger.info("Creating instances for test split...")
    mrf_data["test"] = create_relation_instances(
        test_concepts,
        test_conceptpair_relation_map,
        test_concept_collist_map,
        qgram_model,
        fasttext_model,
        logger,
        args,
    )

    for split in mrf_data:
        output_filepath = os.path.join(
            output_dir, f"openforge_arts_{split}.csv"
        )

        mrf_df = pd.DataFrame(mrf_data[split])
        mrf_df.to_csv(output_filepath, index=False)

        logger.info(f"Saved MRF data for {split} split.")
