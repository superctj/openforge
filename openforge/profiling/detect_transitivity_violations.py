import argparse
import os
import pickle

import numpy as np
import pandas as pd
import torch

from sentence_transformers import SentenceTransformer
from sklearn.utils import extmath

from openforge.utils.custom_logging import create_custom_logger
from openforge.utils.util import parse_config


ENTITY_MATCHING_INSTRUCTION = "Instruct: Classify a given pair of entities as either equivalent or non-equivalent\nQuery: "  # noqa: E501


def collect_entity_info(df: pd.DataFrame):
    l_id_entity_map = {}
    r_id_entity_map = {}

    for _, row in df.iterrows():
        l_id = row["l_id"]
        r_id = row["r_id"]
        l_entity = row["l_entity"]
        r_entity = row["r_entity"]

        l_id_entity_map[l_id] = l_entity
        r_id_entity_map[r_id] = r_entity

    return l_id_entity_map, r_id_entity_map


def generate_embeddings_for_extrapolated_pairs(
    l_entity: str,
    r_id_skip: int,
    r_id_entity_map: dict,
    model,
    batch_size: int,
    device: torch.device,
):
    batch_inputs = []
    all_rids = []
    all_embeddings = []

    for r_id in r_id_entity_map:
        if r_id == r_id_skip:
            continue
        else:
            if len(batch_inputs) != 0 and len(batch_inputs) % batch_size == 0:
                with torch.no_grad():
                    batch_embeddings = model.encode(
                        batch_inputs,
                        batch_size=batch_size,
                        prompt=ENTITY_MATCHING_INSTRUCTION,
                        device=device,
                    )
                    all_embeddings.append(batch_embeddings)
                    batch_inputs = []

            r_entity = r_id_entity_map[r_id]
            batch_inputs.append(
                f"Entity 1: {l_entity}; Entity 2: {r_entity}.{model.tokenizer.eos_token}"  # noqa: E501
            )
            all_rids.append(r_id)

    if len(batch_inputs) > 0:
        with torch.no_grad():
            batch_embeddings = model.encode(
                batch_inputs,
                batch_size=batch_size,
                prompt=ENTITY_MATCHING_INSTRUCTION,
                device=device,
            )
        all_embeddings.append(batch_embeddings)

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    assert all_embeddings.shape[0] == len(r_id_entity_map) - 1

    return all_embeddings, all_rids


def get_ridge_prior_predictions(model, features: np.ndarray):
    d = model.decision_function(features)
    d_2d = np.c_[-d, d]

    pred_proba = extmath.softmax(d_2d)
    confdc_scores = np.max(pred_proba, axis=1)
    pred = np.argmax(pred_proba, axis=1)
    assert pred.shape[0] == features.shape[0]
    assert confdc_scores.shape[0] == features.shape[0]

    return pred, confdc_scores


def detect_transitivity_violations(
    df: pd.DataFrame,
    l_id_entity_map: dict,
    r_id_entity_map: dict,
    embedding_model,
    batch_size: int,
    device: torch.device,
    prior_model,
    logger,
):
    for _, row in df.iterrows():
        if row["prediction"] == 1:
            l_id = row["l_id"]
            r_id = row["r_id"]
            l_entity = row["l_entity"]
            # r_entity = row["r_entity"]

            r_extrapolated_embeddings, r_extrapolated_ids = (
                generate_embeddings_for_extrapolated_pairs(
                    l_entity,
                    r_id,
                    r_id_entity_map,
                    embedding_model,
                    batch_size,
                    device,
                )
            )
            r_extrapolated_predictions, r_extrapolated_confdc_scores = (
                get_ridge_prior_predictions(
                    prior_model, r_extrapolated_embeddings
                )
            )

            logger.info(f"Pair in the test set: ({l_id}, {r_id})")
            logger.info(f"Prediction: {row['prediction']}")
            logger.info(f"Confidence score: {row['confidence_score']:.2f}")

            for j, r_id_extrapolated in enumerate(r_extrapolated_ids):
                if r_extrapolated_predictions[j] == 1:
                    logger.info("-" * 80)
                    logger.info(
                        f"Transitivity violation detected: ({l_id}, {r_id_extrapolated})"  # noqa: E501
                    )
                    logger.info(
                        f"Entities: ({l_entity}, {r_id_entity_map[r_id_extrapolated]})"  # noqa: E501
                    )
                    logger.info(f"Prediction: {r_extrapolated_predictions[j]}")
                    logger.info(
                        f"Confidence score: {r_extrapolated_confdc_scores[j]}"
                    )

            logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_path",
        type=str,
        default="",
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

    prior_dir = config.get("io", "prior_dir")
    prior_model_filepath = config.get("io", "prior_model_filepath")
    model_id = config.get("encoding", "model_id")
    max_length = config.getint("encoding", "max_length")
    batch_size = config.getint("encoding", "batch_size")

    test_prior_filepath = os.path.join(prior_dir, "test.csv")
    test_df = pd.read_csv(test_prior_filepath)
    l_id_entity_map, r_id_entity_map = collect_entity_info(test_df)

    embedding_model = SentenceTransformer(
        model_id,
        trust_remote_code=True,
        model_kwargs={"torch_dtype": "bfloat16"},
    )
    embedding_model.max_seq_length = max_length
    embedding_model.tokenizer.padding_side = "right"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_model = embedding_model.to(device)

    with open(prior_model_filepath, "rb") as f:
        prior_model = pickle.load(f)

    detect_transitivity_violations(
        test_df,
        l_id_entity_map,
        r_id_entity_map,
        embedding_model,
        batch_size,
        device,
        prior_model,
        logger,
    )
