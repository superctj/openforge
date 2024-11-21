import argparse
import os

import numpy as np
import pandas as pd

from openforge.feature_extraction.fb_fasttext import FasttextTransformer
from openforge.feature_extraction.qgram import QGramTransformer
from openforge.feature_extraction.similarity_metrics import (
    cosine_similarity,
    jaccard_index,
)
from openforge.utils.custom_logging import create_custom_logger


def compute_features(
    input_filepath, output_filepath, fasttext_model, qgram_model, logger
):
    input_df = pd.read_json(input_filepath)
    all_features = []

    for i, row in input_df.iterrows():
        l_entity = row["l_entity"]
        r_entity = row["r_entity"]

        l_qgrams = qgram_model.transform(l_entity)
        r_qgrams = qgram_model.transform(r_entity)
        qgram_similarity = jaccard_index(set(l_qgrams), set(r_qgrams))

        l_words = l_entity.split()
        r_words = r_entity.split()
        jaccard_similarity = jaccard_index(set(l_words), set(r_words))

        l_embeddings = fasttext_model.transform(l_words)
        r_embeddings = fasttext_model.transform(r_words)
        embedding_similarity = cosine_similarity(l_embeddings, r_embeddings)

        word_count_ratio = min(len(l_words), len(r_words)) / max(
            len(l_words), len(r_words)
        )
        character_count_ratio = min(len(l_entity), len(r_entity)) / max(
            len(l_entity), len(r_entity)
        )

        features = [
            qgram_similarity,
            jaccard_similarity,
            embedding_similarity,
            word_count_ratio,
            character_count_ratio,
        ]
        all_features.append(features)

        logger.info(f"{i+1}/{input_df.shape[0]}: {features}")

    all_features = np.array(all_features)
    np.save(output_filepath, all_features)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute features for Walmart-Amazon dataset"
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        default="/ssd2/congtj/openforge/Structured/Walmart-Amazon/artifact",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/ssd2/congtj/openforge/Structured/Walmart-Amazon/artifact/hand_designed_features",  # noqa: E501
    )

    parser.add_argument(
        "--fasttext_model_dir",
        type=str,
        default="/ssd/congtj",  # noqa: E501
    )

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logger = create_custom_logger(args.output_dir)
    logger.info(f"Running program: {__file__}\n")
    logger.info(f"{args}\n")

    fasttext_model = FasttextTransformer(cache_dir=args.fasttext_model_dir)
    qgram_model = QGramTransformer(qgram_size=3)

    training_input_filepath = os.path.join(
        args.input_dir, "preprocessed_training.json"
    )
    training_output_filepath = os.path.join(args.output_dir, "training.npy")

    validation_input_filepath = os.path.join(
        args.input_dir, "preprocessed_validation.json"
    )
    validation_output_filepath = os.path.join(args.output_dir, "validation.npy")

    test_input_filepath = os.path.join(args.input_dir, "preprocessed_test.json")
    test_output_filepath = os.path.join(args.output_dir, "test.npy")

    logger.info("=" * 80)
    logger.info("Computing features for the training split:")
    compute_features(
        training_input_filepath,
        training_output_filepath,
        fasttext_model,
        qgram_model,
        logger,
    )

    logger.info("=" * 80)
    logger.info("Computing features for the validation split:")
    compute_features(
        validation_input_filepath,
        validation_output_filepath,
        fasttext_model,
        qgram_model,
        logger,
    )

    logger.info("=" * 80)
    logger.info("Computing features for the test split:")
    compute_features(
        test_input_filepath,
        test_output_filepath,
        fasttext_model,
        qgram_model,
        logger,
    )
