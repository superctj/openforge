import argparse
import os
import pickle
import random

from dataclasses import make_dataclass
from enum import Enum

import pandas as pd

from sklearn.model_selection import train_test_split

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
        ("relation_variable_label", int),
        ("relation_variable_name", str),
    ],
)


def collect_concept_collists(nodes: list):
    concept_collist_map = {}

    for level_1_node in nodes:
        level_1_concept = str(level_1_node)
        assert level_1_concept == level_1_node.texts[0]

        if level_1_concept not in concept_collist_map:
            concept_collist_map[level_1_concept] = (
                level_1_node.text_to_tbl_column_matched[level_1_concept]
            )

        for level_2_node in level_1_node.children:
            assert str(level_2_node) == level_2_node.texts[0]

            for level_2_concept in level_2_node.texts:
                if level_2_concept not in concept_collist_map:
                    concept_collist_map[level_2_concept] = (
                        level_2_node.text_to_tbl_column_matched[level_2_concept]
                    )

            for level_3_node in level_2_node.children:
                assert str(level_3_node) == level_3_node.texts[0]

                for level_3_concept in level_3_node.texts:
                    if level_3_concept not in concept_collist_map:
                        concept_collist_map[level_3_concept] = (
                            level_3_node.text_to_tbl_column_matched[
                                level_3_concept
                            ]
                        )

    return concept_collist_map


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
                    relation_variable_label,
                    relation_variable_name,
                )
            )

    return mrf_data


def create_equiv_relation_split(df: pd.DataFrame):
    num_instances = df.shape[0]

    train_indices = random.sample(range(num_instances), num_instances // 3)
    valid_test_indices = list(set(range(num_instances)) - set(train_indices))

    valid_indices = random.sample(valid_test_indices, num_instances // 3)
    test_indices = list(set(valid_test_indices) - set(valid_indices))

    assert len(train_indices) == len(valid_indices)
    assert len(train_indices) == len(test_indices)

    train_df = df.iloc[train_indices].reset_index(drop=True)
    valid_df = df.iloc[valid_indices].reset_index(drop=True)
    test_df = df.iloc[test_indices].reset_index(drop=True)

    return train_df, valid_df, test_df


def create_hyper_relation_split(
    df: pd.DataFrame, train_prop: float, random_seed: int
):
    train_df, valid_test_df = train_test_split(
        df, train_size=train_prop, random_state=random_seed
    )

    valid_df, test_df = train_test_split(
        valid_test_df, test_size=0.5, random_state=args.random_seed
    )

    train_df.reset_index(drop=True, inplace=True)
    valid_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    return train_df, valid_df, test_df


def collect_concepts_and_relation_instances(
    equiv_df: pd.DataFrame, hyper_df: pd.DataFrame
):
    concepts = []
    pair_relation_map = {}

    for row in hyper_df.itertuples():
        root_concept = row.root_concept
        concept_1 = row.concept_1
        concept_2 = row.concept_2
        concept_3 = row.concept_3

        if root_concept not in concepts:
            concepts.append(root_concept)

        if concept_1 not in concepts:
            concepts.append(concept_1)

        if concept_2 not in concepts:
            concepts.append(concept_2)

        if concept_3 not in concepts:
            concepts.append(concept_3)

        pair_relation_map[(concept_1, concept_2)] = RelationType.HYPER
        pair_relation_map[(concept_2, concept_3)] = RelationType.HYPER
        pair_relation_map[(concept_1, concept_3)] = RelationType.HYPER

    for row in equiv_df.itertuples():
        root_concept = row.root_concept
        concept_1 = row.concept_1
        concept_2 = row.concept_2
        concept_3 = row.concept_3

        if root_concept not in concepts:
            concepts.append(root_concept)

        if concept_1 not in concepts:
            concepts.append(concept_1)

        if concept_2 not in concepts:
            concepts.append(concept_2)

        if concept_3 not in concepts:
            concepts.append(concept_3)

        pair_relation_map[(root_concept, concept_1)] = RelationType.HYPER
        pair_relation_map[(root_concept, concept_2)] = RelationType.HYPER
        pair_relation_map[(root_concept, concept_3)] = RelationType.HYPER

        pair_relation_map[(concept_1, concept_2)] = RelationType.EQUIV
        pair_relation_map[(concept_2, concept_3)] = RelationType.EQUIV
        pair_relation_map[(concept_1, concept_3)] = RelationType.EQUIV

    return concepts, pair_relation_map


def create_splits(
    filepath: str, train_prop: float, random_seed: int, logger=None
):
    df = pd.read_csv(filepath, header=0, delimiter=",")

    equiv_df = df[df["relation_label"] == 1].reset_index(drop=True)
    hyper_df = df[df["relation_label"] == 2].reset_index(drop=True)

    equiv_train_df, equiv_valid_df, equiv_test_df = create_equiv_relation_split(
        equiv_df
    )

    hyper_train_df, hyper_valid_df, hyper_test_df = create_hyper_relation_split(
        hyper_df, train_prop, random_seed
    )

    train_concepts, train_pair_relation_map = (
        collect_concepts_and_relation_instances(equiv_train_df, hyper_train_df)
    )
    logger.info(f"\n Number of training concepts: len{train_concepts}")
    logger.info(f"Number of training instances: len{train_pair_relation_map}")

    valid_concepts, valid_pair_relation_map = (
        collect_concepts_and_relation_instances(equiv_valid_df, hyper_valid_df)
    )
    logger.info(f"\n Number of validation concepts: len{valid_concepts}")
    logger.info(f"Number of validation instances: len{valid_pair_relation_map}")

    test_concepts, test_pair_relation_map = (
        collect_concepts_and_relation_instances(equiv_test_df, hyper_test_df)
    )
    logger.info(f"\n Number of test concepts: len{test_concepts}")
    logger.info(f"Number of test instances: len{test_pair_relation_map}")

    return (
        train_concepts,
        train_pair_relation_map,
        valid_concepts,
        valid_pair_relation_map,
        test_concepts,
        test_pair_relation_map,
    )


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
        default=6,
        help="Number of head nodes to consider.",
    )

    parser.add_argument(
        "--verified_relation_filepath",
        type=str,
        default="/ssd/congtj/openforge/arts/artifact/verified_arts_multi_relations.csv",  # noqa: E501
        help="Path to verified relations file.",
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

    instance_name = os.path.join(
        f"verified_multi_relations_top_{args.num_head_nodes}_nodes",
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
    concept_collist_map = collect_concept_collists(nodeByLevel[args.arts_level])

    (
        train_concepts,
        train_pair_relation_map,
        valid_concepts,
        valid_pair_relation_map,
        test_concepts,
        test_pair_relation_map,
    ) = create_splits(
        args.verified_relation_filepath,
        train_prop=args.train_prop,
        random_seed=args.random_seed,
    )

    qgram_model = QGramTransformer(qgram_size=3)
    fasttext_model = FasttextTransformer(cache_dir=args.fasttext_model_dir)

    mrf_data = {"training": [], "validation": [], "test": []}

    logger.info("Creating instances for training split...")
    mrf_data["training"] = create_relation_instances(
        train_concepts,
        train_pair_relation_map,
        concept_collist_map,
        qgram_model,
        fasttext_model,
        logger,
        args,
    )

    logger.info("Creating instances for validation split...")
    mrf_data["validation"] = create_relation_instances(
        valid_concepts,
        valid_pair_relation_map,
        concept_collist_map,
        qgram_model,
        fasttext_model,
        logger,
        args,
    )

    logger.info("Creating instances for test split...")
    mrf_data["test"] = create_relation_instances(
        test_concepts,
        test_pair_relation_map,
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
