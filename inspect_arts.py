import argparse
import os
import pickle

from ARTS.ontology import OntologyNode
from ARTS.helpers.mongodb_helper import readCSVFileWithTableID
from utils.customized_logging import get_logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arts_output_filepath", type=str, default="./ARTS/output/column_semantics_ontology_threshold_0.9_run_nyc_gpt_3.5_column_semantics_all_root_token_as_root.pickle", help="Path to the ARTS output file")

    parser.add_argument("--num_head_concepts", type=int, default=100, help="Number of head concepts to consider")

    parser.add_argument("--num_val_samples", type=int, default=100, help="Number of sample values per column for manual inspection")

    parser.add_argument("--log_dir", type=str, default="./logs", help="Path to the log directory")
    
    args = parser.parse_args()

    arts_output_id = args.arts_output_filepath.split("/")[-1]
    if arts_output_id.endswith(".pickle"):
        arts_output_id = arts_output_id[:-len(".pickle")]
    log_dir = os.path.join(args.log_dir, f"{arts_output_id}#num_head_concepts-{args.num_head_concepts}#num_val_samples-{args.num_val_samples}")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    with open(args.arts_output_filepath, "rb") as f:
        data = pickle.load(f)
        nodeByLevel = data["nodeByLevel"]
        OntologyNode.init__device_and_threshold(device=data["device"], threshold=data["device"])
        OntologyNode.embeddingByLevelAndIdx = data["embeddings"]

    nodeByLevel[1].sort(key=lambda x:len(x.tbl_column_matched), reverse=True)
    for node in nodeByLevel[1][:args.num_head_concepts]:
        log_filepath = os.path.join(log_dir, f"{node}.log")
        logger = get_logger(log_filepath)

        logger.info(f"Concept: {node}")
        for (table_id, col_name) in node.tbl_column_matched:
            df = readCSVFileWithTableID(table_id, nrows=args.num_val_samples)
            logger.info(df[col_name].tolist())
