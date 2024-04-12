import argparse
import os
import pickle
import random

from dataclasses import make_dataclass

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


def create_equiv_relation_split(df: pd.DataFrame, random_seed: int):
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


def collect_relation_instances(
    filepath: str, train_prop: float, random_seed: int, logger=None
):
    df = pd.read_csv(filepath, header=0, delimiter=",")

    equiv_df = df[df["relation_label"] == 1].reset_index(drop=True)
    hyper_df = df[df["relation_label"] == 2].reset_index(drop=True)

    equiv_train_df, equiv_valid_df, equiv_test_df = create_equiv_relation_split(
        equiv_df, random_seed
    )

    hyper_train_df, hyper_valid_df, hyper_test_df = create_hyper_relation_split(
        hyper_df, train_prop, random_seed
    )

    print(hyper_train_df.head())
    print(hyper_train_df.shape)
    print(hyper_valid_df.head())
    print(hyper_valid_df.shape)
    print(hyper_test_df.head())
    print(hyper_test_df.shape)


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

    collect_relation_instances(
        args.verified_relation_filepath,
        train_prop=args.train_prop,
        random_seed=args.random_seed,
    )

    # # fix random seed
    # random.seed(args.random_seed)

    # instance_name = os.path.join(
    #     f"verified_multi_relations_top_{args.num_head_nodes}_nodes",
    #     f"training_prop_{args.train_prop}",
    # )
    # output_dir = os.path.join(args.output_dir, instance_name)

    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    # if not os.path.exists(args.log_dir):
    #     os.makedirs(args.log_dir)

    # logger = create_custom_logger(args.log_dir)
    # logger.info(f"Running program: {__file__}")
    # logger.info(f"{args}\n")

    # with open(args.arts_data_filepath, "rb") as f:
    #     data = pickle.load(f)
    #     nodeByLevel = data["nodeByLevel"]
    #     OntologyNode.init__device_and_threshold(
    #         device=data["device"], threshold=data["threshold"]
    #     )
    #     OntologyNode.embeddingByLevelAndIdx = data["embeddings"]

    # nodeByLevel[args.arts_level].sort(
    #     key=lambda x: len(x.tbl_column_matched), reverse=True
    # )

    # qgram_model = QGramTransformer(qgram_size=3)
    # fasttext_model = FasttextTransformer(cache_dir=args.fasttext_model_dir)

    # mrf_data = {"training": [], "validation": [], "test": []}
