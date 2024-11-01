import argparse
import os
import pickle

from itertools import combinations

import numpy as np
import pandas as pd
import torch

from llm2vec import LLM2Vec
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import extmath

from openforge.utils.custom_logging import create_custom_logger
from openforge.utils.util import parse_config


ENTITY_MATCHING_INSTRUCTION = "Classify a given pair of entities as either equivalent or non-equivalent: "  # noqa: E501


def get_ridge_prior_predictions(model, features: np.ndarray):
    d = model.decision_function(features)
    d_2d = np.c_[-d, d]

    pred_proba = extmath.softmax(d_2d)
    confdc_scores = np.max(pred_proba, axis=1)
    pred = np.argmax(pred_proba, axis=1)
    assert pred.shape[0] == features.shape[0]
    assert confdc_scores.shape[0] == features.shape[0]

    return pred, confdc_scores


def get_prior_predictions(model, features: np.ndarray):
    pred_proba = model.predict_proba(features)
    pred = (pred_proba[:, 1] >= 0.5).astype(int)
    confdc_scores = np.max(pred_proba, axis=1)
    assert pred.shape[0] == features.shape[0]
    assert confdc_scores.shape[0] == features.shape[0]

    return pred, confdc_scores


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
    all_rid: list,
    r_id_entity_map: dict,
    model,
    batch_size: int,
):
    batch_inputs = []
    all_embeddings = []

    for r_id in all_rid:
        if len(batch_inputs) != 0 and len(batch_inputs) % batch_size == 0:
            with torch.no_grad():
                batch_embeddings = (
                    model.encode(
                        batch_inputs,
                        batch_size=batch_size,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
                all_embeddings.append(batch_embeddings)
                batch_inputs = []

        r_entity = r_id_entity_map[r_id]
        batch_inputs.append(
            [
                ENTITY_MATCHING_INSTRUCTION,
                f"Entity 1: {l_entity}; Entity 2: {r_entity}.",
            ]
        )

    if len(batch_inputs) > 0:
        with torch.no_grad():
            batch_embeddings = (
                model.encode(
                    batch_inputs,
                    batch_size=batch_size,
                )
                .detach()
                .cpu()
                .numpy()
            )
        all_embeddings.append(batch_embeddings)

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    assert all_embeddings.shape[0] == len(all_rid)

    return all_embeddings


def prepare_mrf_inputs(
    df: pd.DataFrame,
    l_id_entity_map: dict,
    r_id_entity_map: dict,
    embedding_model,
    batch_size: int,
    n_neighbors: int,
    prior_model,
    output_dir: str,
    logger,
):
    all_rid = list(r_id_entity_map.keys())
    lid_with_positive_prediction = set(
        df[df["prediction"] == 1]["l_id"].unique().tolist()
    )
    logger.info(
        f"l_ids with positive predictions: {lid_with_positive_prediction}"
    )

    for l_id, group in df.groupby("l_id"):
        if l_id not in lid_with_positive_prediction:
            continue

        l_entity = l_id_entity_map[l_id]
        r_extrapolated_embeddings = generate_embeddings_for_extrapolated_pairs(
            l_entity,
            all_rid,
            r_id_entity_map,
            embedding_model,
            batch_size,
        )

        for _, row in group.iterrows():
            if row["prediction"] == 1:
                r_id = row["r_id"]

                logger.info(f"Pair: ({l_id}, {r_id})")
                logger.info(f"Prediction: {row['prediction']}")
                logger.info(f"Confidence score: {row['confidence_score']:.2f}")

                # +1 to account for the query itself
                knn = NearestNeighbors(
                    n_neighbors=n_neighbors + 1, metric="cosine"
                )
                knn.fit(r_extrapolated_embeddings)

                query_embedding = r_extrapolated_embeddings[all_rid.index(r_id)]
                dist, idx = knn.kneighbors(
                    [query_embedding], return_distance=True
                )

                mrf_l_id = []
                mrf_r_id = []
                mrf_left_entities = []
                mrf_right_entities = []
                prior_features = []

                for i in range(n_neighbors):
                    mrf_l_id.append(f"l_{l_id}")
                    mrf_r_id.append(f"r_{all_rid[idx[0][i]]}")
                    mrf_left_entities.append(l_entity)
                    mrf_right_entities.append(
                        r_id_entity_map[all_rid[idx[0][i]]]
                    )
                    prior_features.append(r_extrapolated_embeddings[idx[0][i]])

                    # logger.info(
                    #     f"Neighbor {i + 1}: {(l_id, all_rid[idx[0][i]])}"
                    # )
                    # logger.info(f"Cosine similarity: {1 - dist[0][i]:.2f}")

                extra_inputs = []

                for pair in combinations(mrf_r_id, 2):
                    mrf_l_id.append(pair[0])
                    mrf_r_id.append(pair[1])

                    l_rid = int(pair[0][2:])
                    r_rid = int(pair[1][2:])
                    mrf_left_entities.append(r_id_entity_map[l_rid])
                    mrf_right_entities.append(r_id_entity_map[r_rid])

                    extra_inputs.append(
                        [
                            ENTITY_MATCHING_INSTRUCTION,
                            f"Entity 1: {r_id_entity_map[l_rid]}; Entity 2: {r_id_entity_map[r_rid]}.",  # noqa: E501
                        ]
                    )

                with torch.no_grad():
                    extra_embeddings = (
                        embedding_model.encode(
                            extra_inputs,
                            batch_size=batch_size,
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )

                all_prior_features = np.concatenate(
                    [np.array(prior_features), extra_embeddings], axis=0
                )
                assert all_prior_features.shape[0] == len(mrf_l_id)

                prior_pred, prior_confdc_scores = get_prior_predictions(
                    prior_model, all_prior_features
                )

                output_filepath = os.path.join(
                    output_dir, f"mrf-input_{l_id}-{r_id}.json"
                )
                feature_output_filepath = os.path.join(
                    output_dir, f"features_{l_id}-{r_id}.npy"
                )

                mrf_input_df = pd.DataFrame(
                    {
                        "l_id": mrf_l_id,
                        "r_id": mrf_r_id,
                        "l_entity": mrf_left_entities,
                        "r_entity": mrf_right_entities,
                        "prior_prediction": prior_pred,
                        "prior_confidence_score": prior_confdc_scores,
                    }
                )

                mrf_input_df.to_json(
                    output_filepath, orient="records", indent=4
                )
                np.save(feature_output_filepath, all_prior_features)

                logger.info("=" * 80)


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

    prior_dir = config.get("io", "prior_dir")
    prior_model_filepath = config.get("io", "prior_model_filepath")
    base_model_id = config.get("encoding", "base_model_id")
    lora_model_id = config.get("encoding", "lora_model_id")
    pooling_mode = config.get("encoding", "pooling_mode")
    max_length = config.getint("encoding", "max_length")
    batch_size = config.getint("encoding", "batch_size")
    n_neighbors = config.getint("knn", "n_neighbors")

    valid_prior_filepath = os.path.join(prior_dir, "validation.csv")
    test_prior_filepath = os.path.join(prior_dir, "test.csv")
    valid_df = pd.read_csv(valid_prior_filepath)
    test_df = pd.read_csv(test_prior_filepath)

    valid_l_id_entity_map, valid_r_id_entity_map = collect_entity_info(valid_df)
    test_l_id_entity_map, test_r_id_entity_map = collect_entity_info(test_df)

    embedding_model = LLM2Vec.from_pretrained(
        base_model_id,
        peft_model_name_or_path=lora_model_id,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.bfloat16,
        pooling_mode=pooling_mode,
        max_length=max_length,
    )

    with open(prior_model_filepath, "rb") as f:
        prior_model = pickle.load(f)

    valid_mrf_output_dir = os.path.join(output_dir, "validation")
    if not os.path.exists(valid_mrf_output_dir):
        os.makedirs(valid_mrf_output_dir)

    test_mrf_output_dir = os.path.join(output_dir, "test")
    if not os.path.exists(test_mrf_output_dir):
        os.makedirs(test_mrf_output_dir)

    logger.info("Preparing MRF inputs for the validation dataset")
    prepare_mrf_inputs(
        valid_df,
        valid_l_id_entity_map,
        valid_r_id_entity_map,
        embedding_model,
        batch_size,
        n_neighbors,
        prior_model,
        valid_mrf_output_dir,
        logger,
    )

    logger.info("Preparing MRF inputs for the test dataset")
    prepare_mrf_inputs(
        test_df,
        test_l_id_entity_map,
        test_r_id_entity_map,
        embedding_model,
        batch_size,
        n_neighbors,
        prior_model,
        test_mrf_output_dir,
        logger,
    )
