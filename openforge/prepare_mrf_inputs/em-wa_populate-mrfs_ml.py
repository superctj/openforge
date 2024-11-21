"""
Populating predictions and confidence scores for MRFs created in
'em-wa_create-mrfs_nv-embed-v2.py'.

Input directory and output directory are the same as we want to overwrite the 
placeholder predictions and confidence scores in the input files.
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.utils import extmath

from openforge.feature_extraction.fb_fasttext import FasttextTransformer
from openforge.feature_extraction.qgram import QGramTransformer
from openforge.feature_extraction.similarity_metrics import (
    cosine_similarity,
    jaccard_index,
)
from openforge.utils.custom_logging import create_custom_logger
from openforge.utils.util import parse_config


def get_ridge_prior_predictions(model, features: np.ndarray):
    d = model.decision_function(features)
    d_2d = np.c_[-d, d]

    pred_proba = extmath.softmax(d_2d)
    confdc_scores = np.max(pred_proba, axis=1)
    preds = np.argmax(pred_proba, axis=1)
    assert preds.shape[0] == features.shape[0]
    assert confdc_scores.shape[0] == features.shape[0]

    return preds, confdc_scores


def get_prior_predictions(model, features: np.ndarray):
    pred_proba = model.predict_proba(features)
    pred = (pred_proba[:, 1] >= 0.5).astype(int)
    confdc_scores = np.max(pred_proba, axis=1)
    assert pred.shape[0] == features.shape[0]
    assert confdc_scores.shape[0] == features.shape[0]

    return pred, confdc_scores


def populate_predictions(
    input_dir: str,
    output_dir: str,
    prior_model,
    fasttext_model,
    qgram_model,
    logger,
):

    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            input_filepath = os.path.join(input_dir, filename)
            input_df = pd.read_json(input_filepath)
            input_df = input_df.drop(columns=["prediction", "confidence_score"])
            all_features = []

            for _, row in input_df.iterrows():
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
                embedding_similarity = cosine_similarity(
                    l_embeddings, r_embeddings
                )

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

            all_features = np.array(all_features)
            preds, confdc_scores = get_ridge_prior_predictions(
                prior_model, all_features
            )

            input_df["prediction"] = preds
            input_df["confidence_score"] = confdc_scores

            output_filepath = os.path.join(output_dir, filename)
            input_df.to_json(output_filepath, orient="records", indent=4)

            logger.info(f"Populated predictions for {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to the experiment configuration file",
    )

    args = parser.parse_args()

    config = parse_config(args.config_path)

    # Create logger
    output_dir = config.get("io", "output_dir")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = create_custom_logger(output_dir)
    logger.info(f"Running program: {__file__}\n")
    logger.info(f"{args}\n")

    printable_config = {section: dict(config[section]) for section in config}
    logger.info(f"Experiment configuration:\n{printable_config}\n")

    prior_model_filepath = config.get("io", "prior_model_filepath")
    fasttext_model_dir = config.get("io", "fasttext_model_dir")
    with open(prior_model_filepath, "rb") as f:
        prior_model = pickle.load(f)

    input_dir = config.get("io", "input_dir")
    valid_input_dir = os.path.join(input_dir, "validation")
    test_input_dir = os.path.join(input_dir, "test")

    valid_output_dir = os.path.join(output_dir, "validation")
    if not os.path.exists(valid_output_dir):
        os.makedirs(valid_output_dir)

    test_output_dir = os.path.join(output_dir, "test")
    if not os.path.exists(test_output_dir):
        os.makedirs(test_output_dir)

    fasttext_model = FasttextTransformer(cache_dir=fasttext_model_dir)
    qgram_model = QGramTransformer(qgram_size=3)

    logger.info("Populating predictions for validation set")
    populate_predictions(
        valid_input_dir,
        valid_output_dir,
        prior_model,
        fasttext_model,
        qgram_model,
        logger,
    )

    logger.info("Populating predictions for test set")
    populate_predictions(
        test_input_dir,
        test_output_dir,
        prior_model,
        fasttext_model,
        qgram_model,
        logger,
    )
