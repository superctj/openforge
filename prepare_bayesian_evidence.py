import argparse
import os
import pickle
import random

from datetime import datetime

from d3l.indexing.feature_extraction.schema.qgram_transformer import QGramTransformer
from d3l.indexing.feature_extraction.values.fasttext_embedding_transformer import FasttextTransformer

from ARTS.helpers.mongodb_helper import readCSVFileWithTableID
from ARTS.ontology import OntologyNode
from synthesize_arts_data import compute_value_signature, compute_name_similarity, compute_value_similarity
from utils.customized_logging import get_logger, logging_level_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arts_output_filepath", type=str, default="./data/column_semantics_ontology_threshold_0.9_run_nyc_gpt_3.5_merge_root.pickle", help="Path to the ARTS output file.")

    parser.add_argument("--arts_level", type=int, default=2, help="Level of the ARTS ontology to extract concepts.")

    parser.add_argument("--num_head_concepts", type=int, default=100, help="Number of head concepts to consider.")

    parser.add_argument("--num_val_samples", type=int, default=10000, help="Number of maximum sample values per column for computing features.")

    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level.")

    parser.add_argument("--random_seed", type=int, default=12345, help="Random seed.")
    args = parser.parse_args()

    random.seed(args.random_seed)

    log_dir = f"./logs/arts_top-{args.num_head_concepts}-concepts_evidence"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    cur_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_filepath = os.path.join(log_dir, f"{cur_datetime}.log")
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
    evidence_data = []

    nodeByLevel[args.arts_level].sort(key=lambda x:len(x.tbl_column_matched), reverse=True)
    for i, node_i in enumerate(nodeByLevel[args.arts_level][:args.num_head_concepts]):
        # node is the head concept
        assert str(node_i) == node_i.texts[0]
        # each merged concept has at least one corresponding table column
        assert(len(node_i.texts) == len(node_i.text_to_tbl_column_matched))

        logger.info("=" * 50)
        logger.info(f"Concept {i}: {node_i}")

        # use head concept as a reference point
        i_name_signature = set(qgram_transformer.transform(str(node_i)))
        i_fasttext_signature = compute_value_signature(
            node_i.text_to_tbl_column_matched[str(node_i)], fasttext_transformer, args.num_val_samples)
        
        if len(i_fasttext_signature) == 0:
            logger.info(f"Cannot compute value signature for concept {i}: {node_i}.")
            continue

        for j, node_j in enumerate(nodeByLevel[args.arts_level][:args.num_head_concepts]):
            if j == i:
                continue

            logger.info(f"Concept {j}: {node_j}")
            j_name_signature = set(qgram_transformer.transform(str(node_j)))
            j_fasttext_signature = compute_value_signature(
                node_j.text_to_tbl_column_matched[str(node_j)], fasttext_transformer, args.num_val_samples)
            
            if len(j_fasttext_signature) == 0:
                logger.info(f"Cannot compute value signature for concept {j}: {node_j}.")
                continue
                
            # compute name similarity
            name_sim = compute_name_similarity(i_name_signature, j_name_signature)

            # compute value similarity
            value_sim = compute_value_similarity(i_fasttext_signature, j_fasttext_signature)

            evidence_data.append(((i, j), [name_sim, value_sim]))

    evidence_save_filepath = f"./data/arts_top-{args.num_head_concepts}-concepts_evidence.pkl"
    with open(evidence_save_filepath, "wb") as f:
        pickle.dump(evidence_data, f)
