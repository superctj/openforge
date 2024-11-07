"""
Prepare MRF inputs based on prior predictions of a model and k-nearest
neighbors given by the nv-embed-v2 embedding model. The difference between this
program and 'em-wa_create-mrfs_nv-embed-v2.py' is that we this program builds a
shared KNN index of embeddings for all l_ids with positive predictions instead
of building a KNN index for each l_id with positive predictions. This is done
to reduce the computation overhead.

Due to conflicted environments, populating predictions and confidence scores for
MRFs is done in 'em-wa_populate-mrfs_llm.py'.
"""

import argparse
import os

from itertools import combinations

import numpy as np
import pandas as pd
import torch

from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

from openforge.utils.custom_logging import create_custom_logger
from openforge.utils.util import parse_config

EMBEDDING_MODEL_INSTRUCTION = "Instruct: Classify a given pair of entities as either equivalent or non-equivalent\nQuery: "  # noqa: E501


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


def generate_embeddings_for_entities(
    all_id: list,
    id_entity_map: dict,
    model,
    batch_size: int,
    device: torch.device,
):
    batch_inputs = []
    all_embeddings = []

    for e_id in all_id:
        if len(batch_inputs) != 0 and len(batch_inputs) % batch_size == 0:
            with torch.no_grad():
                batch_embeddings = model.encode(
                    batch_inputs,
                    batch_size=batch_size,
                    device=device,
                )
                all_embeddings.append(batch_embeddings)
                batch_inputs = []

        entity = id_entity_map[e_id]
        batch_inputs.append(f"{entity}.{model.tokenizer.eos_token}")

    if len(batch_inputs) > 0:
        with torch.no_grad():
            batch_embeddings = model.encode(
                batch_inputs,
                batch_size=batch_size,
                device=device,
            )
        all_embeddings.append(batch_embeddings)

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    assert all_embeddings.shape[0] == len(all_id)

    return all_embeddings


def create_mrfs(
    df: pd.DataFrame,
    l_id_entity_map: dict,
    r_id_entity_map: dict,
    embedding_model,
    batch_size: int,
    device: torch.device,
    n_neighbors: int,
    output_dir: str,
    logger,
):
    if "prior_prediction" in df.columns:
        df.rename(
            columns={
                "prior_prediction": "prediction",
                "prior_confidence_score": "confidence_score",
            },
            inplace=True,
        )

    all_rid = list(r_id_entity_map.keys())
    lid_with_positive_prediction = set(
        df[df["prediction"] == 1]["l_id"].unique().tolist()
    )
    logger.info(
        f"l_ids with positive predictions: {lid_with_positive_prediction}"
    )

    r_entity_embeddings = generate_embeddings_for_entities(
        all_rid, r_id_entity_map, embedding_model, batch_size, device
    )
    # +1 to account for rid in the query row
    knn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="cosine")
    knn.fit(r_entity_embeddings)

    for l_id, group in df.groupby("l_id"):
        if l_id not in lid_with_positive_prediction:
            continue

        l_entity = l_id_entity_map[l_id]
        query_embedding = embedding_model.encode(
            [f"{l_entity}.{embedding_model.tokenizer.eos_token}"],
            batch_size=batch_size,
            device=device,
        )[0]
        _, idx = knn.kneighbors([query_embedding], return_distance=True)

        for _, row in group.iterrows():
            if row["prediction"] == 1:
                r_id = row["r_id"]

                logger.info(f"Pair: ({l_id}, {r_id})")
                logger.info(f"Prediction: {row['prediction']}")
                logger.info(f"Confidence score: {row['confidence_score']:.2f}")

                mrf_l_id = [f"l_{l_id}"]
                mrf_r_id = [f"r_{r_id}"]
                mrf_left_entities = [l_entity]
                mrf_right_entities = [r_id_entity_map[r_id]]

                for i in range(len(idx[0])):
                    neighbor_id = all_rid[idx[0][i]]
                    if neighbor_id == r_id:
                        continue

                    mrf_l_id.append(f"l_{l_id}")
                    mrf_r_id.append(f"r_{neighbor_id}")
                    mrf_left_entities.append(l_entity)
                    mrf_right_entities.append(r_id_entity_map[neighbor_id])

                for pair in combinations(mrf_r_id, 2):
                    mrf_l_id.append(pair[0])
                    mrf_r_id.append(pair[1])

                    l_rid = int(pair[0][2:])
                    r_rid = int(pair[1][2:])
                    mrf_left_entities.append(r_id_entity_map[l_rid])
                    mrf_right_entities.append(r_id_entity_map[r_rid])

                output_filepath = os.path.join(
                    output_dir, f"mrf-input_{l_id}-{r_id}.json"
                )
                mrf_input_df = pd.DataFrame(
                    {
                        "l_id": mrf_l_id,
                        "r_id": mrf_r_id,
                        "l_entity": mrf_left_entities,
                        "r_entity": mrf_right_entities,
                        "prediction": [row["prediction"]] * len(mrf_l_id),
                        "confidence_score": [row["confidence_score"]]
                        * len(mrf_l_id),
                    }
                )
                mrf_input_df.to_json(
                    output_filepath, orient="records", indent=4
                )

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

    valid_output_dir = os.path.join(output_dir, "validation")
    if not os.path.exists(valid_output_dir):
        os.makedirs(valid_output_dir)

    test_output_dir = os.path.join(output_dir, "test")
    if not os.path.exists(test_output_dir):
        os.makedirs(test_output_dir)

    logger = create_custom_logger(output_dir)
    logger.info(f"Running program: {__file__}\n")
    logger.info(f"{args}\n")

    printable_config = {section: dict(config[section]) for section in config}
    logger.info(f"Experiment configuration:\n{printable_config}\n")

    prior_dir = config.get("io", "prior_dir")
    valid_prior_filepath = os.path.join(prior_dir, "validation.json")
    test_prior_filepath = os.path.join(prior_dir, "test.json")

    embedding_model_id = config.get("encoding", "model_id")
    max_length = config.getint("encoding", "max_length")
    batch_size = config.getint("encoding", "batch_size")
    n_neighbors = config.getint("knn", "n_neighbors")

    embedding_model = SentenceTransformer(
        embedding_model_id,
        trust_remote_code=True,
        model_kwargs={"torch_dtype": "bfloat16"},
    )
    embedding_model.max_seq_length = max_length
    embedding_model.tokenizer.padding_side = "right"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_model = embedding_model.to(device)

    # Create MRFs based on prior predictions and knn
    valid_df = pd.read_json(valid_prior_filepath)
    valid_l_id_entity_map, valid_r_id_entity_map = collect_entity_info(valid_df)
    create_mrfs(
        valid_df,
        valid_l_id_entity_map,
        valid_r_id_entity_map,
        embedding_model,
        batch_size,
        device,
        n_neighbors,
        valid_output_dir,
        logger,
    )

    test_df = pd.read_json(test_prior_filepath)
    test_l_id_entity_map, test_r_id_entity_map = collect_entity_info(test_df)
    create_mrfs(
        test_df,
        test_l_id_entity_map,
        test_r_id_entity_map,
        embedding_model,
        batch_size,
        device,
        n_neighbors,
        test_output_dir,
        logger,
    )
