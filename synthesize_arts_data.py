import argparse
import os
import pickle
import random

from datetime import datetime

import numpy as np

from d3l.indexing.feature_extraction.schema.qgram_transformer import QGramTransformer
from d3l.indexing.feature_extraction.values.fasttext_embedding_transformer import FasttextTransformer

from ARTS.helpers.mongodb_helper import readCSVFileWithTableID
from ARTS.ontology import OntologyNode
from utils.customized_logging import get_logger, logging_level_map

CONSTANT = 1e-9
VALUE_SIGNATURE_ATTEMPTS = 100


def compute_value_signature(col_list, feature_extractor):
    # randomly pick a corresponding table column to compute value signature
    count = 0

    while count < VALUE_SIGNATURE_ATTEMPTS:
        rnd_idx = random.randrange(len(col_list))
        table_id, col_name = col_list[rnd_idx]
        df = readCSVFileWithTableID(table_id, usecols=[col_name], nrows=args.num_val_samples).astype(str)
        fasttext_signature = feature_extractor.transform(df[col_name].tolist())
        if len(fasttext_signature) != 0:
            break
        else:
            count += 1

    return fasttext_signature


def compute_name_similarity(name_sig1, name_sig2):
    return len(name_sig1.intersection(name_sig2)) / len(name_sig1.union(name_sig2))


def compute_value_similarity(value_sig1, value_sig2):
    return np.dot(value_sig1, value_sig2) / (np.linalg.norm(value_sig1) * np.linalg.norm(value_sig2) + CONSTANT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arts_output_filepath", type=str, default="/home/jjxing/ssd/openforge/ARTS/output/column_semantics_ontology_threshold_0.9_run_nyc_gpt_3.5_merge_root.pickle", help="Path to the ARTS output file.")

    parser.add_argument("--arts_level", type=int, default=2, help="Level of the ARTS ontology to extract concepts.")

    parser.add_argument("--num_head_concepts", type=int, default=200, help="Number of head concepts to consider.")

    parser.add_argument("--num_val_samples", type=int, default=10000, help="Number of maximum sample values per column for computing features.")

    parser.add_argument("--log_dir", type=str, default="./logs/arts_synthesized_data", help="Directory to store logs.")

    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level.")

    parser.add_argument("--random_seed", type=int, default=12345, help="Random seed.")
    args = parser.parse_args()

    random.seed(args.random_seed)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    
    cur_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_filepath = os.path.join(args.log_dir, f"{cur_datetime}.log")
    logger = get_logger(log_filepath)
    logger.setLevel(logging_level_map[args.log_level])
    logger.info(args)

    with open(args.arts_output_filepath, "rb") as f:
        data = pickle.load(f)
        nodeByLevel = data["nodeByLevel"]
        OntologyNode.init__device_and_threshold(device=data["device"], threshold=data["device"])
        OntologyNode.embeddingByLevelAndIdx = data["embeddings"]

    qgram_transformer = QGramTransformer(qgram_size=3)
    fasttext_transformer = FasttextTransformer()
    tr_te_data = []

    nodeByLevel[args.arts_level].sort(key=lambda x:len(x.tbl_column_matched), reverse=True)
    for i, node in enumerate(nodeByLevel[args.arts_level][:args.num_head_concepts]):
        # node is the head concept
        assert str(node) == node.texts[0]
        # each merged concept has at least one corresponding table column
        assert(len(node.texts) == len(node.text_to_tbl_column_matched))

        logger.info("=" * 50)
        logger.info(f"Concept: {node}")
        logger.info(f"Number of merged concepts: {len(node.texts)}")

        # use head concept as a reference point
        name_signature = set(qgram_transformer.transform(str(node)))
        fasttext_signature = compute_value_signature(node.text_to_tbl_column_matched[str(node)], fasttext_transformer)
        if len(fasttext_signature) == 0:
            logger.info(f"Cannot compute value signature for reference concept: {node}.")
            continue

        for merged_concept in node.texts:
            merged_name_signature = set(qgram_transformer.transform(merged_concept))
            merged_value_signature = compute_value_signature(node.text_to_tbl_column_matched[merged_concept], fasttext_transformer)
            if len(merged_value_signature) == 0:
                logger.info(f"Cannot compute value signature for merged concept: {merged_concept}.")
                continue

            # compute name similarity
            name_sim = compute_name_similarity(name_signature, merged_name_signature)

            # compute value similarity
            value_sim = compute_value_similarity(fasttext_signature, merged_value_signature)

            tr_te_data.append(([name_sim, value_sim], 1))

        neg_node = nodeByLevel[args.arts_level][i+1]
        for neg_concept in neg_node.texts:
            neg_name_signature = set(qgram_transformer.transform(neg_concept))
            neg_value_signature = compute_value_signature(neg_node.text_to_tbl_column_matched[neg_concept], fasttext_transformer)
            if len(neg_value_signature) == 0:
                logger.info(f"Cannot compute value signature for negative concept: {neg_concept}.")
                continue

            # compute name similarity
            name_sim = compute_name_similarity(name_signature, neg_name_signature)

            # compute value similarity
            value_sim = compute_value_similarity(fasttext_signature, neg_value_signature)

            tr_te_data.append(([name_sim, value_sim], 0))
        
        # for (table_id, col_name) in node.tbl_column_matched:
        #     df = readCSVFileWithTableID(table_id, nrows=args.num_val_samples)
        #     # logger.info(df[col_name].tolist())

    with open(f"./data/arts_num-head-concepts-{args.num_head_concepts}.pkl", "wb") as save_f:
        pickle.dump(tr_te_data, save_f)
