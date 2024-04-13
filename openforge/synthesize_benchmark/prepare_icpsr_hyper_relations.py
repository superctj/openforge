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


def create_splits(
    source_data: pd.DataFrame,
    num_train_instances: int,
    num_valid_instances: int,
    num_test_instances: int,
    random_seed: int,
):
    """
    Create train, valid, and test splits.
    """
    # Create a list of unique hypernym pairs
    tuples = source_data[
        ["concept_1", "concept_2", "concept_3"]
    ].drop_duplicates()

    # Sample train instances
    train_tuples = tuples.sample(
        n=num_train_instances, random_state=random_seed
    )

    # Remove train instances from the list
    tuples = tuples[~tuples.isin(train_tuples)].dropna()

    # Sample valid instances
    valid_tuples = tuples.sample(
        n=num_valid_instances, random_state=random_seed
    )

    # Remove valid instances from the list
    tuples = tuples[~tuples.isin(valid_tuples)].dropna()

    # Sample test instances
    test_tuples = tuples.sample(n=num_test_instances, random_state=random_seed)

    return train_tuples, valid_tuples, test_tuples


def collect_concepts_and_relation_instances(split_tuples: pd.DataFrame):
    """
    Collect concepts and relation instances.
    """

    concepts = []
    relation_instances = set()

    for row in split_tuples.itertuples():
        concept_1 = row.concept_1
        concept_2 = row.concept_2
        concept_3 = row.concept_3

        if concept_1 not in concepts:
            concepts.append(concept_1)

        if concept_2 not in concepts:
            concepts.append(concept_2)

        if concept_3 not in concepts:
            concepts.append(concept_3)

        # Add relation instances
        relation_instances.add((concept_1, concept_2))
        relation_instances.add((concept_2, concept_3))
        relation_instances.add((concept_1, concept_3))

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
                relation_variable_label = 1
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
        output_dir, f"openforge_icpsr_hyper_{split}.csv"
    )
    split_df = pd.DataFrame(split_data)

    y = split_df["relation_variable_label"]
    logger.info(f"\tNumber of empty relation instances: {np.sum(y == 0)}")
    logger.info(f"\tNumber of hypernymy instances: {np.sum(y == 1)}")

    split_df.to_csv(output_filepath, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source_filepath",
        type=str,
        default="/ssd/congtj/openforge/icpsr/artifact/hypernymy_transitivity.csv",  # noqa: E501
        help="Path to the source file.",
    )

    parser.add_argument(
        "--num_train_intances",
        type=str,
        default=60,
        help="The number of transitive instances to create the training split.",
    )

    parser.add_argument(
        "--num_valid_intances",
        type=str,
        default=20,
        help="The number of transitive instances to create the validation split.",  # noqa: E501
    )

    parser.add_argument(
        "--num_test_intances",
        type=str,
        default=20,
        help="The number of transitive instances to create the test split.",
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
        default="/ssd/congtj/openforge/icpsr/artifact",
        help="Directory to save outputs.",
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="/home/congtj/openforge/logs/icpsr",
        help="Directory to save logs.",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=12345,
        help="Random seed for sampling.",
    )

    args = parser.parse_args()

    # Fix random seed
    random.seed(args.random_seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    logger = create_custom_logger(args.log_dir)
    logger.info(f"Running program: {__file__}\n")
    logger.info(f"{args}\n")

    source_data = pd.read_csv(args.source_filepath, header=0, delimiter=",")
    train_tuples, valid_tuples, test_tuples = create_splits(
        source_data,
        args.num_train_intances,
        args.num_valid_intances,
        args.num_test_intances,
        args.random_seed,
    )

    train_concepts, train_instances = collect_concepts_and_relation_instances(
        train_tuples
    )

    logger.info(f"Number of training concepts: {len(train_concepts)}")
    logger.info(f"Number of training instances: {len(train_instances)}")

    valid_concepts, valid_instances = collect_concepts_and_relation_instances(
        valid_tuples
    )

    logger.info(f"Number of validation concepts: {len(valid_concepts)}")
    logger.info(f"Number of validation instances: {len(valid_instances)}")

    test_concepts, test_instances = collect_concepts_and_relation_instances(
        test_tuples
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
