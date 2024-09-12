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


def collect_concept_relations(split_nodes: list):
    hypernymy_pairs = []
    concept_collist_map = {}

    for level_1_node in split_nodes:
        if str(level_1_node) == "borough":
            continue

        assert str(level_1_node) == level_1_node.texts[0]
        logger.info(f"\nLevel 1 node: {str(level_1_node)}")

        for level_1_concept in level_1_node.texts[:1]:
            level_1_concept = level_1_concept.lower()

            for level_2_node in level_1_node.children[:10]:
                assert str(level_2_node) == level_2_node.texts[0]
                logger.info(f"Level 2 node: {str(level_2_node)}")

                # Collect hypernymy relations
                for level_2_concept in level_2_node.texts:
                    level_2_concept = level_2_concept.lower()

                    logger.info(
                        f"Hypernymy concepts: ({level_1_concept}, "
                        f"{level_2_concept})"
                    )

                    hypernymy_pairs.append((level_1_concept, level_2_concept))

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

    logger.info(f"Number of hypernymy pairs: {len(hypernymy_pairs)}")

    return hypernymy_pairs, concept_collist_map


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
    hypernymy_pairs: dict,
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

        for j, concept_j in enumerate(concepts):
            if j <= i:
                continue

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
            if (concept_i, concept_j) in hypernymy_pairs:
                relation_variable_label = 1
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


def get_concepts_from_relation_pairs(relation_pairs):
    # Keeping the partial order is important

    concepts = set()
    for pair in relation_pairs:
        concepts.add(pair[0])

    concepts = list(concepts)

    for pair in relation_pairs:
        if pair[1] not in concepts:
            concepts.append(pair[1])

    return concepts


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
        default="/home/congtj/openforge/logs/arts_hyper",
        help="Directory to save logs.",
    )

    parser.add_argument(
        "--random_seed", type=int, default=12345, help="Random seed."
    )

    args = parser.parse_args()

    # fix random seed
    random.seed(args.random_seed)

    instance_name = os.path.join(
        f"hypernymy_top_{args.num_head_nodes}_nodes",
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

    top_nodes = [
        nodeByLevel[args.arts_level][i] for i in range(args.num_head_nodes)
    ]

    hypernymy_pairs, concept_collist_map = collect_concept_relations(top_nodes)
    hypernymy_pairs = random.sample(hypernymy_pairs, 200)

    train_pairs = random.sample(
        hypernymy_pairs, int(args.train_prop * len(hypernymy_pairs))
    )
    train_concepts = get_concepts_from_relation_pairs(train_pairs)
    logger.info(f"Number of concepts in training split: {len(train_concepts)}")

    valid_test_pairs = list(set(hypernymy_pairs) - set(train_pairs))
    valid_pairs = random.sample(valid_test_pairs, len(valid_test_pairs) // 2)
    valid_concepts = get_concepts_from_relation_pairs(valid_pairs)
    logger.info(
        f"Number of concepts in validation split: {len(valid_concepts)}"
    )

    test_pairs = list(set(valid_test_pairs) - set(valid_pairs))
    test_concepts = get_concepts_from_relation_pairs(test_pairs)
    logger.info(f"Number of concepts in test split: {len(test_concepts)}")

    mrf_data = {"training": [], "validation": [], "test": []}
    qgram_model = QGramTransformer(qgram_size=3)
    fasttext_model = FasttextTransformer(cache_dir=args.fasttext_model_dir)

    logger.info("Creating instances for training split...")
    mrf_data["training"] = create_relation_instances(
        train_concepts,
        train_pairs,
        concept_collist_map,
        qgram_model,
        fasttext_model,
        logger,
        args,
    )

    logger.info("Creating instances for validation split...")
    mrf_data["validation"] = create_relation_instances(
        valid_concepts,
        valid_pairs,
        concept_collist_map,
        qgram_model,
        fasttext_model,
        logger,
        args,
    )

    logger.info("Creating instances for test split...")
    mrf_data["test"] = create_relation_instances(
        test_concepts,
        test_pairs,
        concept_collist_map,
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
