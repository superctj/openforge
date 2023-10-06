import argparse
import os
import pickle

import numpy as np

from d3l.indexing.feature_extraction.schema.qgram_transformer import QGramTransformer
from d3l.indexing.feature_extraction.values.fasttext_embedding_transformer import FasttextTransformer

from ARTS.helpers.mongodb_helper import readCSVFileWithTableID
from ARTS.ontology import OntologyNode


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arts_output_filepath", type=str, default="/home/jjxing/ssd/openforge/ARTS/output/column_semantics_ontology_threshold_0.9_run_nyc_gpt_3.5_merge_root.pickle", help="Path to the ARTS output file")

    parser.add_argument("--num_head_concepts", type=int, default=200, help="Number of head concepts to consider")

    parser.add_argument("--num_val_samples", type=int, default=10000, help="Number of sample values per column for manual inspection")

    args = parser.parse_args()

    with open(args.arts_output_filepath, "rb") as f:
        data = pickle.load(f)
        nodeByLevel = data["nodeByLevel"]
        OntologyNode.init__device_and_threshold(device=data["device"], threshold=data["device"])
        OntologyNode.embeddingByLevelAndIdx = data["embeddings"]

    qgram_transformer = QGramTransformer(qgram_size=3)
    fasttext_transformer = FasttextTransformer()
    tr_te_data = []

    nodeByLevel[2].sort(key=lambda x:len(x.tbl_column_matched), reverse=True)
    for i, node in enumerate(nodeByLevel[2][:args.num_head_concepts]):
        # log_filepath = os.path.join(log_dir, f"{node}.log")
        # logger = get_logger(log_filepath)
        # logger.info(f"Concept: {node}")

        # print(type(node.text_to_tbl_column_matched))
        # print(len(node.text_to_tbl_column_matched))
        name_signature = set(qgram_transformer.transform(str(node)))
        for (table_id, col_name) in node.tbl_column_matched:
            df = readCSVFileWithTableID(table_id, usecols=[col_name], nrows=args.num_val_samples)
            fasttext_signature = fasttext_transformer.transform(df[col_name].tolist())
            if len(fasttext_signature) == 0:
                continue
            else:
                break

        for key in node.text_to_tbl_column_matched.keys():
            for merged in node.text_to_tbl_column_matched[key]:
                merged_concept = merged[1]
                merged_df = readCSVFileWithTableID(merged[0], usecols=[merged_concept], nrows=args.num_val_samples)

                merged_name_signature = set(qgram_transformer.transform(merged_concept))
                merged_fasttext_signature = fasttext_transformer.transform(merged_df[merged_concept].tolist())

                name_sim = len(name_signature.intersection(merged_name_signature)) / len(name_signature.union(merged_name_signature))
                value_sim = np.dot(fasttext_signature, merged_fasttext_signature) #/ (np.linalg.norm(fasttext_signature) * np.linalg.norm(merged_fasttext_signature))

                tr_te_data.append(([name_sim, value_sim], 1))
        
        neg_node = nodeByLevel[2][i+1]
        for key in neg_node.text_to_tbl_column_matched.keys():
            for merged in neg_node.text_to_tbl_column_matched[key]:
                merged_concept = merged[1]
                merged_df = readCSVFileWithTableID(merged[0], usecols=[merged_concept], nrows=args.num_val_samples)

                merged_name_signature = set(qgram_transformer.transform(merged_concept))
                merged_fasttext_signature = fasttext_transformer.transform(merged_df[merged_concept].tolist())

                name_sim = len(name_signature.intersection(merged_name_signature)) / len(name_signature.union(merged_name_signature))
                value_sim = np.dot(fasttext_signature, merged_fasttext_signature) #/ (np.linalg.norm(fasttext_signature) * np.linalg.norm(merged_fasttext_signature))

                tr_te_data.append(([name_sim, value_sim], 0))

        # for (table_id, col_name) in node.tbl_column_matched:
        #     df = readCSVFileWithTableID(table_id, nrows=args.num_val_samples)
        #     # logger.info(df[col_name].tolist())

    save_f = open(f"./data/arts_num-head-concepts-{args.num_head_concepts}.pkl", "wb")
    pickle.dump(tr_te_data, save_f)
