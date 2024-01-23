import argparse
import os
import pickle

from openforge.ARTS.ontology import OntologyNode
from openforge.ARTS.helpers.mongodb_helper import convertTableIDToTableName, readCSVFileWithTableID
from openforge.utils.custom_logging import get_custom_logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--arts_output_filepath", type=str, default="/home/congtj/openforge/data/column_semantics_ontology_threshold_0.9_run_nyc_gpt_3.5_merge_root.pickle", help="Path to the ARTS output file")

    parser.add_argument("--arts_level", type=int, default=2, help="Level of the ARTS ontology to extract concepts.")

    parser.add_argument("--num_head_concepts", type=int, default=50, help="Number of head concepts to consider")

    parser.add_argument("--num_val_samples", type=int, default=100, help="Number of sample values per column for manual inspection")

    parser.add_argument("--log_dir", type=str, default="/home/congtj/openforge/logs", help="Path to the log directory")
    
    args = parser.parse_args()

    arts_output_id = args.arts_output_filepath.split("/")[-1]
    if arts_output_id.endswith(".pickle"):
        arts_output_id = arts_output_id[:-len(".pickle")]
    
    log_dir = os.path.join(
        args.log_dir,
        f"{arts_output_id}_num_head_concepts-{args.num_head_concepts}_num_val_samples-{args.num_val_samples}"
    )
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    with open(args.arts_output_filepath, "rb") as f:
        data = pickle.load(f)
        nodeByLevel = data["nodeByLevel"]
        OntologyNode.init__device_and_threshold(device=data["device"], threshold=data["device"])
        OntologyNode.embeddingByLevelAndIdx = data["embeddings"]

    nodeByLevel[args.arts_level].sort(key=lambda x:len(x.tbl_column_matched), reverse=True)

    for i, node in enumerate(nodeByLevel[args.arts_level][:args.num_head_concepts]):
        log_filepath = os.path.join(log_dir, f"{i}_{node}.log")
        logger = get_custom_logger(log_filepath)

        logger.info(f"Concept: {node}")
        logger.info(f"Merged concepts: {node.texts}")

        for concept_name in node.text_to_tbl_column_matched:
            logger.info(f"Concept name: {concept_name}")

            for (table_id, col_name) in node.text_to_tbl_column_matched[concept_name]:
                table_name = convertTableIDToTableName(table_id)

                logger.info(f"Table ID: {table_id}\tTable name: {table_name}\tColumn name: {col_name}")
            
            logger.info("="*50)
        # logger.info(f"Corresponding columns: {node.text_to_tbl_column_matched}")
        # for (table_id, col_name) in node.tbl_column_matched:
        #     df = readCSVFileWithTableID(table_id, nrows=args.num_val_samples)
        #     logger.info(df[col_name].tolist())
