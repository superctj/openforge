import argparse
import logging
import os
import random

from collections import defaultdict
from dataclasses import make_dataclass
from enum import Enum

import numpy as np
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
    HYPO = 3  # Hyponymy


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


def collect_concept_relations(
    term_relations: pd.DataFrame, concept_ids: list[int], logger: logging.Logger
) -> tuple[dict, dict, dict]:
    concept_relations_visibility = {}
    concept_relation_map = {}
    pair_relation_map = {}

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

                if subject_id not in concept_relation_map:
                    concept_relation_map[subject_id] = [
                        (object_id, RelationType.HYPER)
                    ]
                else:
                    concept_relation_map[subject_id].append(
                        (object_id, RelationType.HYPER)
                    )

                pair_relation_map[(subject_id, object_id)] = RelationType.HYPER
            # subject is narrower than object
            elif relationship == 2:
                concept_relations_visibility[subject_id][1] = 1

                if subject_id not in concept_relation_map:
                    concept_relation_map[subject_id] = [
                        (object_id, RelationType.HYPO)
                    ]
                else:
                    concept_relation_map[subject_id].append(
                        (object_id, RelationType.HYPO)
                    )

                pair_relation_map[(subject_id, object_id)] = RelationType.HYPO
            # subject is a referred or nonpreferred term of object
            elif relationship == 4 or relationship == 5:
                concept_relations_visibility[subject_id][0] = 1

                if subject_id not in concept_relation_map:
                    concept_relation_map[subject_id] = [
                        (object_id, RelationType.EQUIV)
                    ]
                else:
                    concept_relation_map[subject_id].append(
                        (object_id, RelationType.EQUIV)
                    )

                pair_relation_map[(subject_id, object_id)] = RelationType.EQUIV

    temp_dict = defaultdict(int)
    for key, val in concept_relations_visibility.items():
        temp_dict[key] = sum(val)

    sorted_concept_ids = sorted(
        temp_dict.items(), key=lambda x: x[1], reverse=True
    )

    return sorted_concept_ids, concept_relation_map, pair_relation_map


def expand_split_concept_vocabulary(
    split_top_concept_ids: list[int], concept_relation_map: dict
) -> list:
    split_top_concept_ids.sort()
    split_concept_ids = set(split_top_concept_ids)

    for concept_id in split_top_concept_ids:
        if concept_id in concept_relation_map:
            for object_concept_id, _ in concept_relation_map[concept_id]:
                split_concept_ids.add(object_concept_id)

    return list(split_top_concept_ids)


def collect_ordered_relation_instances(term_relations: pd.DataFrame):
    ordered_relation_instances = {
        RelationType.HYPER: [],
        RelationType.HYPO: [],
        RelationType.EQUIV: [],
    }
    pair_relation_map = {}

    for row in term_relations.itertuples():
        subject_id = int(row.SUBJECT_ID)
        object_id = int(row.OBJECT_ID)
        relationship = int(row.RELATIONSHIP)

        if subject_id < object_id:
            # subject is broader than object
            if relationship == 1:
                ordered_relation_instances[RelationType.HYPER].append(
                    (subject_id, object_id)
                )
                pair_relation_map[(subject_id, object_id)] = RelationType.HYPER
            # subject is narrower than object
            elif relationship == 2:
                ordered_relation_instances[RelationType.HYPO].append(
                    (subject_id, object_id)
                )
                pair_relation_map[(subject_id, object_id)] = RelationType.HYPO
            # subject is a referred or nonpreferred term of object
            elif relationship == 4 or relationship == 5:
                ordered_relation_instances[RelationType.EQUIV].append(
                    (subject_id, object_id)
                )
                pair_relation_map[(subject_id, object_id)] = RelationType.EQUIV

    return ordered_relation_instances, pair_relation_map


def create_splits(
    sorted_concept_ids: list[int], concept_relation_map: dict, train_prop: float
) -> tuple[list[int], list[int], list[int]]:
    # Select top concepts with the most relations
    top_concept_ids = []
    for concept_id, relation_count in sorted_concept_ids:
        if relation_count == 3:
            top_concept_ids.append(concept_id)
        else:
            assert relation_count < 3
            break

    num_top_train_concepts = int(len(top_concept_ids) * train_prop)
    num_top_valid_concepts = int(len(top_concept_ids) * (1 - train_prop) / 2)

    top_train_concept_ids = random.sample(
        population=top_concept_ids, k=num_top_train_concepts
    )
    top_valid_test_concept_ids = list(
        set(top_concept_ids) - set(top_train_concept_ids)
    )
    top_valid_concept_ids = random.sample(
        population=top_valid_test_concept_ids,
        k=num_top_valid_concepts,
    )
    top_test_concept_ids = list(
        set(top_valid_test_concept_ids) - set(top_valid_concept_ids)
    )

    train_concept_ids = expand_split_concept_vocabulary(
        top_train_concept_ids, concept_relation_map
    )
    valid_concept_ids = expand_split_concept_vocabulary(
        top_valid_concept_ids, concept_relation_map
    )
    test_concept_ids = expand_split_concept_vocabulary(
        top_test_concept_ids, concept_relation_map
    )

    return train_concept_ids, valid_concept_ids, test_concept_ids


def create_splits_for_single_relation(
    relation_instances: list, train_prop: float, logger: logging.Logger
) -> tuple[list, list, list]:
    num_relation_instances = len(relation_instances)
    num_train_instances = int(num_relation_instances * train_prop)

    train_instances = random.sample(
        population=relation_instances, k=num_train_instances
    )
    valid_test_instances = list(set(relation_instances) - set(train_instances))

    num_valid_instances = len(valid_test_instances) // 2
    valid_instances = random.sample(
        population=valid_test_instances,
        k=num_valid_instances,
    )
    test_instances = list(set(valid_test_instances) - set(valid_instances))

    logger.info(f"Number of training instances: {len(train_instances)}")
    logger.info(f"Number of validation instances: {len(valid_instances)}")
    logger.info(f"Number of test instances: {len(test_instances)}")

    return train_instances, valid_instances, test_instances


def collect_split_vocabulary(
    equiv_instances: list, hyper_instances: list, hypo_instances: list
) -> list[int]:
    split_vocabulary = set()

    for instance in equiv_instances:
        split_vocabulary.add(instance[0])
        split_vocabulary.add(instance[1])

    for instance in hyper_instances:
        split_vocabulary.add(instance[0])
        split_vocabulary.add(instance[1])

    for instance in hypo_instances:
        split_vocabulary.add(instance[0])
        split_vocabulary.add(instance[1])

    split_vocabulary = list(split_vocabulary)
    split_vocabulary.sort()

    return split_vocabulary


def create_splits_from_relation_instances(
    ordered_relation_instances: dict,
    num_intances_per_relation: int,
    train_prop: float,
    logger: logging.Logger,
):
    for key in ordered_relation_instances:
        logger.info(f"\nRelation type: {key}")
        logger.info(
            f"Number of instances: {len(ordered_relation_instances[key])}"
        )

    sampled_equiv_instances = random.sample(
        population=ordered_relation_instances[RelationType.EQUIV],
        k=num_intances_per_relation,
    )
    sampled_hyper_instances = random.sample(
        population=ordered_relation_instances[RelationType.HYPER],
        k=num_intances_per_relation,
    )
    sampled_hypo_instances = random.sample(
        population=ordered_relation_instances[RelationType.HYPO],
        k=num_intances_per_relation,
    )

    logger.info("\nEquivalent relation:")
    train_equiv_instances, valid_equiv_instances, test_equiv_instances = (
        create_splits_for_single_relation(
            sampled_equiv_instances, train_prop, logger
        )
    )

    logger.info("\nHypernymy relation:")
    train_hyper_instances, valid_hyper_instances, test_hyper_instances = (
        create_splits_for_single_relation(
            sampled_hyper_instances, train_prop, logger
        )
    )

    logger.info("\nHyponymy relation:")
    train_hypo_instances, valid_hypo_instances, test_hypo_instances = (
        create_splits_for_single_relation(
            sampled_hypo_instances, train_prop, logger
        )
    )

    train_ids = collect_split_vocabulary(
        train_equiv_instances, train_hyper_instances, train_hypo_instances
    )
    valid_ids = collect_split_vocabulary(
        valid_equiv_instances, valid_hyper_instances, valid_hypo_instances
    )
    test_ids = collect_split_vocabulary(
        test_equiv_instances, test_hyper_instances, test_hypo_instances
    )

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

            if (concept_i_id, concept_j_id) in relation_instances:
                relation_type = relation_instances[(concept_i_id, concept_j_id)]

                if relation_type == RelationType.EQUIV:
                    relation_variable_label = 1
                elif relation_type == RelationType.HYPER:
                    relation_variable_label = 2
                elif relation_type == RelationType.HYPO:
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


def save_data(
    split_data: list, split: str, output_dir: str, logger: logging.Logger
):
    output_filepath = os.path.join(output_dir, f"openforge_icpsr_{split}.csv")
    split_df = pd.DataFrame(split_data)

    y = split_df["relation_variable_label"]
    logger.info(f"\tNumber of null relation instances: {np.sum(y == 0)}")
    logger.info(f"\tNumber of equivalent instances: {np.sum(y == 1)}")
    logger.info(f"\tNumber of hypernymy instances: {np.sum(y == 2)}")
    logger.info(f"\tNumber of hyponymy instances: {np.sum(y == 3)}\n")

    split_df.to_csv(output_filepath, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source_data_dir",
        type=str,
        default="/nfs/turbo/coe-jag/congtj/icpsr_data",
        help="Path to the source data directory.",
    )

    parser.add_argument(
        "--num_intances_per_relation",
        type=str,
        default=40,
        help="The number of instances to consider per relation type.",
    )

    parser.add_argument(
        "--train_prop", type=float, default=0.5, help="Training proportion."
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

    logger.info(f"\nSubject terms: {subject_terms.head()}")
    logger.info(f"\nTerm relations: {term_relations.head()}\n")

    concept_ids = subject_terms["TERM_ID"].to_list()
    ordered_relation_instances, pair_relation_map = (
        collect_ordered_relation_instances(term_relations)
    )

    id_term_map = {}
    for row in subject_terms.itertuples():
        id_term_map[row.TERM_ID] = row.TERM

    train_ids, valid_ids, test_ids = create_splits_from_relation_instances(
        ordered_relation_instances,
        args.num_intances_per_relation,
        args.train_prop,
        logger,
    )

    train_data = synthesize_split_data(
        train_ids,
        id_term_map,
        pair_relation_map,
        args.fasttext_model_dir,
        logger,
    )

    valid_data = synthesize_split_data(
        valid_ids,
        id_term_map,
        pair_relation_map,
        args.fasttext_model_dir,
        logger,
    )

    test_data = synthesize_split_data(
        test_ids,
        id_term_map,
        pair_relation_map,
        args.fasttext_model_dir,
        logger,
    )

    logger.info("Training split:")
    save_data(train_data, "training", args.output_dir, logger)

    logger.info("Validation split:")
    save_data(valid_data, "validation", args.output_dir, logger)

    logger.info("Test split:")
    save_data(test_data, "test", args.output_dir, logger)
