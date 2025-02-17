import argparse
import logging
import os
import random

from dataclasses import make_dataclass

import pandas as pd
import numpy as np

from openforge.feature_extraction.fb_fasttext import FasttextTransformer
from openforge.feature_extraction.qgram import QGramTransformer
from openforge.feature_extraction.similarity_metrics import (
    cosine_similarity,
    edit_distance,
    jaccard_index,
)
from openforge.utils.custom_logging import create_custom_logger


DataEntry = make_dataclass(
    "DataEntry",
    [
        ("concept_1", str),
        ("concept_2", str),
        ("name_qgram_similarity", float),
        ("name_jaccard_similarity", float),
        ("name_edit_distance", int),
        ("name_fasttext_similarity", float),
        ("name_word_count_ratio", float),
        ("name_char_count_ratio", float),
        ("relation_variable_label", int),
        ("relation_variable_name", str),
    ],
)


def create_splits(data_dir: str):
    """
    Create train, valid, and test splits.
    """
   
    train_tuples = pd.read_csv(os.path.join(data_dir, "training_instances.csv"), header=0, delimiter=",")[
        ["concept_1", "concept_2", "concept_3"]
    ].drop_duplicates()
    valid_tuples = pd.read_csv(os.path.join(data_dir, "validation_instances.csv"), header=0, delimiter=",")[
        ["concept_1", "concept_2", "concept_3"]
    ].drop_duplicates()
    test_tuples = pd.read_csv(os.path.join(data_dir, "test_instances.csv"), header=0, delimiter=",")[
        ["concept_1", "concept_2", "concept_3"]
    ].drop_duplicates()

    train_hyper_tuples = train_tuples[:train_tuples.shape[0] // 2]
    train_hypo_tuples = train_tuples[train_tuples.shape[0] // 2:]
    assert train_hyper_tuples.shape[0] == train_hypo_tuples.shape[0]

    valid_hyper_tuples = valid_tuples[:valid_tuples.shape[0] // 2]
    valid_hypo_tuples = valid_tuples[valid_tuples.shape[0] // 2:]
    assert valid_hyper_tuples.shape[0] == valid_hypo_tuples.shape[0]

    test_hyper_tuples = test_tuples[:test_tuples.shape[0] // 2]
    test_hypo_tuples = test_tuples[test_tuples.shape[0] // 2:]
    assert test_hyper_tuples.shape[0] == test_hypo_tuples.shape[0]

    return (train_hyper_tuples, train_hypo_tuples), (valid_hyper_tuples, valid_hypo_tuples), (test_hyper_tuples, test_hypo_tuples)


def collect_concepts_and_relation_instances(hyper_tuples: pd.DataFrame, hypo_tuples: pd.DataFrame):
    """
    Collect concepts and relation instances.
    """

    concepts = []
    relation_instances = {}

    concepts.extend(hyper_tuples["concept_1"].unique().tolist())
    concepts.extend(hyper_tuples["concept_2"].unique().tolist())
    concepts.extend(hyper_tuples["concept_3"].unique().tolist())

    for i, row in enumerate(hyper_tuples.itertuples()):
        concept_1 = row.concept_1
        concept_2 = row.concept_2
        concept_3 = row.concept_3

        relation_instances[(concept_1, concept_2)] = 1
        relation_instances[(concept_2, concept_3)] = 1
        relation_instances[(concept_1, concept_3)] = 1

    concepts.extend(hypo_tuples["concept_3"].unique().tolist())
    concepts.extend(hypo_tuples["concept_2"].unique().tolist())
    concepts.extend(hypo_tuples["concept_1"].unique().tolist())

    for i, row in enumerate(hypo_tuples.itertuples()):
        concept_1 = row.concept_1
        concept_2 = row.concept_2
        concept_3 = row.concept_3

        relation_instances[(concept_3, concept_2)] = 2
        relation_instances[(concept_2, concept_1)] = 2
        relation_instances[(concept_3, concept_1)] = 2
        
    return concepts, relation_instances


def get_concept_signatures(
    concept: str,
    qgram_transformer,
    fasttext_transformer,
):
    name_qgram_signature = set(qgram_transformer.transform(concept))
    name_fasttext_signature = fasttext_transformer.transform([concept])

    return name_qgram_signature, name_fasttext_signature


def synthesize_split_data(
    concepts: list,
    relation_instances: dict,
    fasttext_model_dir: str,
    logger: logging.Logger,
) -> list:

    data_entries = []
    qgram_transformer = QGramTransformer(qgram_size=3)
    fasttext_transformer = FasttextTransformer(cache_dir=fasttext_model_dir)

    for i in range(len(concepts)):
        concept_i = concepts[i]
        logger.info(f"Concept {i+1}: {concept_i}")

        (
            concept_i_name_qgram_signature,
            concept_i_name_fasttext_signature,
        ) = get_concept_signatures(
            concept_i, qgram_transformer, fasttext_transformer
        )

        if len(concept_i_name_fasttext_signature) == 0:
            logger.info(
                "Cannot compute name fasttext signature for concept: "
                f"{concept_i}."
            )
            continue

        for j in range(len(concepts)):
            if j <= i:
                continue

            concept_j = concepts[j]
            logger.info(f"\tConcept {j+1}: {concept_j}")

            (
                concept_j_name_qgram_signature,
                concept_j_name_fasttext_signature,
            ) = get_concept_signatures(
                concept_j, qgram_transformer, fasttext_transformer
            )

            if len(concept_j_name_fasttext_signature) == 0:
                logger.info(
                    "Cannot compute name fasttext signature for concept: "
                    f"{concept_j}."
                )
                continue

            name_qgram_sim = jaccard_index(
                concept_i_name_qgram_signature, concept_j_name_qgram_signature
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

            relation_variable_name = f"R_{i+1}-{j+1}"

            if (concept_i, concept_j) in relation_instances:
                relation_variable_label = relation_instances[
                    (concept_i, concept_j)
                ]
            else:
                relation_variable_label = 0

            data_entries.append(
                DataEntry(
                    concept_1=concept_i,
                    concept_2=concept_j,
                    name_qgram_similarity=name_qgram_sim,
                    name_jaccard_similarity=name_jaccard_sim,
                    name_edit_distance=name_edit_dist,
                    name_fasttext_similarity=name_fasttext_sim,
                    name_word_count_ratio=name_word_count_ratio,
                    name_char_count_ratio=name_char_count_ratio,
                    relation_variable_label=relation_variable_label,
                    relation_variable_name=relation_variable_name,
                )
            )

    return data_entries


def save_data(
    split_data: list, split: str, output_dir: str, logger: logging.Logger
):
    logger.info(f"Saving {split} split data...")

    output_filepath = os.path.join(
        output_dir, f"openforge_icpsr_hyper_hypo_{split}.csv"
    )
    split_df = pd.DataFrame(split_data)

    y = split_df["relation_variable_label"]
    logger.info(f"\tNumber of empty relation instances: {np.sum(y == 0)}")
    logger.info(f"\tNumber of hypernymy instances: {np.sum(y == 1)}")
    logger.info(f"\tNumber of hyponymy instances: {np.sum(y == 2)}")

    split_df.to_csv(output_filepath, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        default="/ssd/congtj/openforge/icpsr/artifact/openforge_icpsr_hyper_hypo",  # noqa: E501
        help="Directory containing source data.",
    )

    parser.add_argument(
        "--fasttext_model_dir",
        type=str,
        default="/ssd/congtj",
        help="Directory containing fasttext model weights.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/ssd/congtj/openforge/icpsr/artifact/openforge_icpsr_hyper_hypo",
        help="Directory to save outputs.",
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="/home/congtj/openforge/logs/icpsr",
        help="Directory to save logs.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    logger = create_custom_logger(args.log_dir)
    logger.info(f"Running program: {__file__}\n")
    logger.info(f"{args}\n")

    train_tuples, valid_tuples, test_tuples = create_splits(
        args.data_dir,
    )

    train_concepts, train_instances = collect_concepts_and_relation_instances(
        train_tuples[0], train_tuples[1]
    )

    logger.info(f"Number of training concepts: {len(train_concepts)}")
    logger.info(f"Number of training instances: {len(train_instances)}")

    valid_concepts, valid_instances = collect_concepts_and_relation_instances(
        valid_tuples[0], valid_tuples[1]
    )

    logger.info(f"Number of validation concepts: {len(valid_concepts)}")
    logger.info(f"Number of validation instances: {len(valid_instances)}")

    test_concepts, test_instances = collect_concepts_and_relation_instances(
        test_tuples[0], test_tuples[1]
    )

    logger.info(f"Number of test concepts: {len(test_concepts)}")
    logger.info(f"Number of test instances: {len(test_instances)}")

    train_data = synthesize_split_data(
        train_concepts, train_instances, args.fasttext_model_dir, logger
    )

    valid_data = synthesize_split_data(
        valid_concepts, valid_instances, args.fasttext_model_dir, logger
    )

    test_data = synthesize_split_data(
        test_concepts, test_instances, args.fasttext_model_dir, logger
    )

    save_data(train_data, "training", args.output_dir, logger)
    save_data(valid_data, "validation", args.output_dir, logger)
    save_data(test_data, "test", args.output_dir, logger)
