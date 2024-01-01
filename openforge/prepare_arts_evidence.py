import argparse
import os
import pickle
import random

from openforge.ARTS.ontology import OntologyNode
from openforge.utils.custom_logging import get_custom_logger
from openforge.feature_extraction.fb_fasttext import FasttextTransformer, compute_fasttext_signature
from openforge.feature_extraction.qgram import QGramTransformer
from openforge.feature_extraction.similarity_metrics import cosine_similarity, jaccard_index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arts_output_filepath", type=str, default="/home/tianji/openforge/data/column_semantics_ontology_threshold_0.9_run_nyc_gpt_3.5_merge_root.pickle", help="Path to the ARTS output file.")

    parser.add_argument("--arts_level", type=int, default=2, help="Level of the ARTS ontology to extract concepts.")

    parser.add_argument("--num_head_concepts", type=int, default=4, help="Number of head concepts to consider.")

    parser.add_argument("--num_val_samples", type=int, default=10000, help="Number of maximum sample values per column for computing features.")

    parser.add_argument("--log_dir", type=str, default="/home/tianji/openforge/logs", help="Logging directory.")

    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level.")

    parser.add_argument("--random_seed", type=int, default=12345, help="Random seed.")
    args = parser.parse_args()

    # fix random seed
    random.seed(args.random_seed)

    # create logging directory
    log_dir = os.path.join(args.log_dir, f"arts_top-{args.num_head_concepts}-concepts_evidence")
    logger = get_custom_logger(log_dir, args.log_level)
    logger.info(args)

    with open(args.arts_output_filepath, "rb") as f:
        data = pickle.load(f)
        nodeByLevel = data["nodeByLevel"]
        OntologyNode.init__device_and_threshold(device=data["device"], threshold=data["device"])
        OntologyNode.embeddingByLevelAndIdx = data["embeddings"]

    nodeByLevel[args.arts_level].sort(key=lambda x:len(x.tbl_column_matched), reverse=True)

    qgram_transformer = QGramTransformer(qgram_size=3)
    fasttext_transformer = FasttextTransformer()
    evidence_data = []

    for i, node_i in enumerate(nodeByLevel[args.arts_level][:args.num_head_concepts]):
        # node is the head concept
        assert str(node_i) == node_i.texts[0]
        # each merged concept has at least one corresponding table column
        assert(len(node_i.texts) == len(node_i.text_to_tbl_column_matched))

        logger.info("=" * 50)
        logger.info(f"Concept {i}: {node_i}")

        # use head concept as a reference point
        i_name_signature = set(qgram_transformer.transform(str(node_i)))
        i_fasttext_signature = compute_fasttext_signature(
            node_i.text_to_tbl_column_matched[str(node_i)], fasttext_transformer, args.num_val_samples)
        
        if len(i_fasttext_signature) == 0:
            logger.info(f"Cannot compute value signature for concept {i}: {node_i}.")
            continue

        for j, node_j in enumerate(nodeByLevel[args.arts_level][:args.num_head_concepts]):
            if j == i:
                continue

            logger.info(f"Concept {j}: {node_j}")
            j_name_signature = set(qgram_transformer.transform(str(node_j)))
            j_fasttext_signature = compute_fasttext_signature(
                node_j.text_to_tbl_column_matched[str(node_j)], fasttext_transformer, args.num_val_samples)
            
            if len(j_fasttext_signature) == 0:
                logger.info(f"Cannot compute value signature for concept {j}: {node_j}.")
                continue
                
            # compute name similarity
            name_sim = jaccard_index(i_name_signature, j_name_signature)

            # compute value similarity
            value_sim = cosine_similarity(i_fasttext_signature, j_fasttext_signature)

            evidence_data.append(((i, j), [name_sim, value_sim]))

    evidence_save_filepath = f"./data/arts_top-{args.num_head_concepts}-concepts_evidence.pkl"
    with open(evidence_save_filepath, "wb") as f:
        pickle.dump(evidence_data, f)
