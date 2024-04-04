import argparse
import logging
import os
import random

from collections import defaultdict
from dataclasses import make_dataclass
from enum import Enum

import pandas as pd

from openforge.feature_extraction.fb_fasttext import FasttextTransformer
from openforge.feature_extraction.qgram import QGramTransformer
from openforge.feature_extraction.similarity_metrics import (
    cosine_similarity,
    edit_distance,
    jaccard_index,
)
from openforge.utils.custom_logging import create_custom_logger


class RelationType(Enum):
    NULL = 0
    EQUIV = 1  # Equivalent
    HYPER = 2  # Hypernymy
    HYPON = 3  # Hyponymy


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
        ("relation_variable_name", str),
        ("relation_variable_label", int),
    ],
)


def collect_relation_instances(term_relations: pd.DataFrame) -> dict:
    relation_instances = {}

    for row in term_relations.itertuples():
        subject_id = int(row.SUBJECT_ID)
        object_id = int(row.OBJECT_ID)
        relationship = int(row.RELATIONSHIP)

        if subject_id < object_id:
            # subject is broader than object
            if relationship == 1:
                relation_instances[(subject_id, object_id)] = RelationType.HYPER
            # subject is narrower than object
            elif relationship == 2:
                relation_instances[(subject_id, object_id)] = RelationType.HYPON
            # subject is a referred or nonpreferred term of object
            elif relationship == 4 or relationship == 5:
                relation_instances[(subject_id, object_id)] = RelationType.EQUIV

        return relation_instances


def create_splits(
    sorted_concept_ids: list[int], num_concepts: int, train_prop: float
) -> tuple[list[int], list[int], list[int]]:
    top_concept_ids = sorted_concept_ids[:num_concepts]

    num_train_concepts = int(num_concepts * train_prop)
    num_valid_concepts = int(num_concepts * (1 - train_prop) / 2)
    # num_test_concepts = num_concepts - num_train_concepts - num_valid_concepts
    # assert num_valid_concepts == num_test_concepts, (
    #     "The number of validation concepts should be equal to the number of "
    #     "test concepts."
    # )

    train_ids = random.sample(population=top_concept_ids, k=num_train_concepts)
    valid_test_ids = list(set(top_concept_ids) - set(train_ids))
    valid_ids = random.sample(
        population=valid_test_ids,
        k=num_valid_concepts,
    )
    test_ids = list(set(valid_test_ids) - set(valid_ids))
    assert len(valid_ids) == len(test_ids), (
        "The number of validation concepts should be equal to the number of "
        "test concepts."
    )

    train_ids.sort()
    valid_ids.sort()
    test_ids.sort()

    return train_ids, valid_ids, test_ids


def get_concept_signatures(
    concept: str,
    qgram_transformer,
    fasttext_transformer,
):
    name_qgram_signature = set(qgram_transformer.transform(concept))
    name_fasttext_signature = fasttext_transformer.transform([concept])

    return name_qgram_signature, name_fasttext_signature


def synthesize_split_data(
    split_ids: list[int],
    id_term_map: dict,
    relation_instances: dict,
    fasttext_model_dir: str,
    logger: logging.Logger,
) -> list:
    data_entries = []
    qgram_transformer = QGramTransformer(qgram_size=3)
    fasttext_transformer = FasttextTransformer(cache_dir=fasttext_model_dir)

    for i in range(len(split_ids)):
        concept_i_id = split_ids[i]
        concept_i = id_term_map[concept_i_id]
        logger.info("\n" + "=" * 50)
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

        for j in range(i + 1, len(split_ids)):
            concept_j_id = split_ids[j]
            concept_j = id_term_map[concept_j_id]
            logger.info(f"Concept {j+1}: {concept_j}")

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

            if (concept_i_id, concept_j_id) in relation_instances:
                relation_type = relation_instances[(concept_i_id, concept_j_id)]

                if relation_type == RelationType.EQUIV:
                    relation_variable_label = 1
                elif relation_type == RelationType.HYPER:
                    relation_variable_label = 2
                elif relation_type == RelationType.HYPON:
                    relation_variable_label = 3
                else:
                    raise ValueError(f"Unknown relation type: {relation_type}.")
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
                    relation_variable_name=relation_variable_name,
                    relation_variable_label=relation_variable_label,
                )
            )

    return data_entries


def save_data(split_data: list, split: str, output_dir: str):
    output_filepath = os.path.join(output_dir, f"openforge_icpsr_{split}.csv")
    split_df = pd.DataFrame(split_data)
    split_df.to_csv(output_filepath, index=False)


def get_sorted_concept_ids(
    term_relations: pd.DataFrame, concept_ids: list[int], logger: logging.Logger
) -> dict:
    concept_relations_visibility = {}

    for concept in concept_ids:
        """
        1st 0: no equivalent concepts
        2nd 0: no broader concepts
        3rd 0: no narrower concepts
        """
        concept_relations_visibility[concept] = [0, 0, 0]

    for row in term_relations.itertuples():
        subject_id = int(row.SUBJECT_ID)
        assert subject_id in concept_relations_visibility
        object_id = int(row.OBJECT_ID)
        relationship = int(row.RELATIONSHIP)

        if subject_id < object_id:
            # subject is broader than object
            if relationship == 1:
                concept_relations_visibility[subject_id][2] = 1
            # subject is narrower than object
            elif relationship == 2:
                concept_relations_visibility[subject_id][1] = 1
            # subject is a referred or nonpreferred term of object
            elif relationship == 4 or relationship == 5:
                concept_relations_visibility[subject_id][0] = 1

    temp_dict = defaultdict(int)
    for key, val in concept_relations_visibility.items():
        temp_dict[key] = sum(val)

    sorted_dict = sorted(temp_dict.items(), key=lambda x: x[1], reverse=True)
    sorted_concept_ids = [key for key, _ in sorted_dict]

    return sorted_concept_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source_data_dir",
        type=str,
        default="/nfs/turbo/coe-jag/congtj/icpsr_data",
        help="Path to the source data directory.",
    )

    parser.add_argument(
        "--num_concepts",
        type=str,
        default=300,
        help="Total number of concepts in the synthesized vocabulary.",
    )

    parser.add_argument(
        "--train_prop", type=float, default=0.6, help="Training proportion."
    )

    parser.add_argument(
        "--fasttext_model_dir",
        type=str,
        default="/nfs/turbo/coe-jag/congtj",
        help="Directory containing fasttext model weights.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/nfs/turbo/coe-jag/congtj/icpsr_data/artifact",
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

    subject_terms_filepath = os.path.join(
        args.source_data_dir, "subject_terms.xlsx"
    )
    relation_filepath = os.path.join(
        args.source_data_dir, "term_relations.xlsx"
    )

    subject_terms = pd.read_excel(subject_terms_filepath)
    term_relations = pd.read_excel(relation_filepath)

    concept_ids = subject_terms["TERM_ID"].to_list()
    sorted_concept_ids = get_sorted_concept_ids(
        term_relations, concept_ids, logger
    )

    id_term_map = {}
    for row in subject_terms.itertuples():
        id_term_map[row.TERM_ID] = row.TERM

    relation_instances = collect_relation_instances(term_relations)

    train_ids, valid_ids, test_ids = create_splits(
        sorted_concept_ids, args.num_concepts, args.train_prop
    )

    train_data = synthesize_split_data(
        train_ids,
        id_term_map,
        relation_instances,
        args.fasttext_model_dir,
        logger,
    )

    valid_data = synthesize_split_data(
        valid_ids,
        id_term_map,
        relation_instances,
        args.fasttext_model_dir,
        logger,
    )

    test_data = synthesize_split_data(
        test_ids,
        id_term_map,
        relation_instances,
        args.fasttext_model_dir,
        logger,
    )

    save_data(train_data, "training", args.output_dir)
    save_data(valid_data, "validation", args.output_dir)
    save_data(test_data, "test", args.output_dir)
