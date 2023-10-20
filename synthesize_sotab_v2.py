import argparse
import os
import pickle
import random

from dataclasses import make_dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from d3l.indexing.feature_extraction.schema.qgram_transformer import QGramTransformer
from d3l.indexing.feature_extraction.values.fasttext_embedding_transformer import FasttextTransformer

from synthesize_arts_data import compute_name_similarity, compute_value_similarity
from utils.customized_logging import get_logger, logging_level_map


DELIMITER = "!ctj!"


def synthesize_pos_and_neg_pairs(data_dir: str, output_filepath: str):
    schemaorg_metadata_filepath = os.path.join(data_dir, "cta_schemaorg/sotab_v2_cta_test_set.csv")
    dbpedia_metadata_filepath = os.path.join(data_dir, "cta_dbpedia/sotab_cta_test_dbpedia.csv")

    schemaorg_df = pd.read_csv(schemaorg_metadata_filepath)
    dbpedia_df = pd.read_csv(dbpedia_metadata_filepath)

    schemaorg_cols = set()
    schemaorg_col_label_mapping = {}
    
    PosEntry = make_dataclass("PosEntry", [
        ("table_name", str), ("column_index", int), ("schemaorg_label", str), ("dbpedia_label", str)])
    NegEntry = make_dataclass("NegEntry", [
        ("schemaorg_table_name", str), ("schemaorg_column_index", int), ("schemaorg_label", str), ("dbpedia_table_name", str), ("dbpedia_column_index", int), ("dbpedia_label", str)])
    pos_data = []
    neg_data = []

    for row in schemaorg_df.itertuples():
        unique_id = row.table_name + DELIMITER + str(row.column_index)
        schemaorg_cols.add(unique_id)
        schemaorg_col_label_mapping[unique_id] = row.label

    # collect positive data
    for row in dbpedia_df.itertuples():
        unique_id = row.table_name + DELIMITER + str(row.column_index)
        if unique_id in schemaorg_cols:
            pos_data.append(PosEntry(
                row.table_name, row.column_index, schemaorg_col_label_mapping[unique_id], row.label))

    # collect negative data
    schemaorg_cols = list(schemaorg_cols)
    for row in dbpedia_df.itertuples():
        unique_id = row.table_name + DELIMITER + str(row.column_index)
        while True:
            schemaorg_col = random.choice(schemaorg_cols)
            if schemaorg_col != unique_id:
                schemaorg_table_name, schemaorg_column_index = schemaorg_col.split(DELIMITER)
                schemaorg_label = schemaorg_col_label_mapping[schemaorg_col]

                neg_data.append(NegEntry(
                    schemaorg_table_name, int(schemaorg_column_index), schemaorg_label, row.table_name, row.column_index, row.label))
                break

    # print("Number of positive pairs:", len(pos_data))
    # print("Number of negative pairs:", len(neg_data))
    cutoff = min(len(pos_data), len(neg_data))
    pos_data = pos_data[:cutoff]
    neg_data = neg_data[:cutoff]
    #  

    pos_df = pd.DataFrame(pos_data)
    neg_df = pd.DataFrame(neg_data)
    pos_df.to_csv(f"{output_filepath}_pos.csv", index=False)
    neg_df.to_csv(f"{output_filepath}_neg.csv", index=False)
    return pos_df, neg_df


def synthesize_evaluation_dataset(pos_df: pd.DataFrame, neg_df: pd.DataFrame, schemaorg_table_dir: str, dbpedia_table_dir: str, logger, args: argparse.Namespace):
    qgram_transformer = QGramTransformer(qgram_size=3)
    fasttext_transformer = FasttextTransformer()
    evaluation_data = []
    
    for row in pos_df.itertuples():
        logger.info("=" * 50)
        logger.info(f"Table name in positive data: {row.table_name}")

        table_name = row.table_name
        col_idx = row.column_index

        table = pd.read_json(os.path.join(dbpedia_table_dir, table_name), 
                             compression="gzip", lines=True)
        table = table.astype(str)
        
        if table.shape[0] < args.value_sample_size * 2:
            sample_values = table.iloc[:, col_idx].tolist()
            sample_size = int(len(sample_values) / 2)
            schemaorg_values = sample_values[:sample_size]
            dbpedia_values = sample_values[sample_size:]
        else:
            sample_values = table.iloc[:, col_idx].sample(n=args.value_sample_size*2, random_state=args.random_seed).tolist()

            schemaorg_values = sample_values[:args.value_sample_size]
            dbpedia_values = sample_values[args.value_sample_size:]

        schemaorg_name_signature = set(qgram_transformer.transform(row.schemaorg_label))
        schemaorg_fasttext_signature = fasttext_transformer.transform(schemaorg_values)

        if len(schemaorg_fasttext_signature) == 0:
            logger.info("Cannot compute value signature for schemaorg values.")
            logger.info("schemaorg values: ", " ".join(map(str, schemaorg_values)))
            continue

        dbpedia_label = row.dbpedia_label.split("/")[-1]
        dbpedia_name_signature = set(qgram_transformer.transform(dbpedia_label))
        dbpedia_fasttext_signature = fasttext_transformer.transform(dbpedia_values)
        if len(dbpedia_fasttext_signature) == 0:
            logger.info("Cannot compute value signature for dbpedia values.")
            logger.info("dbpedia values: ", " ".join(map(str, dbpedia_values)))
            continue

        name_sim = compute_name_similarity(schemaorg_name_signature, dbpedia_name_signature)
        value_sim = compute_value_similarity(schemaorg_fasttext_signature, dbpedia_fasttext_signature)

        evaluation_data.append(([name_sim, value_sim], 1))
    
    for row in neg_df.itertuples():
        logger.info("=" * 50)
        logger.info(f"Table name in negative data: {row.schemaorg_table_name}")

        schemaorg_table_name = row.schemaorg_table_name
        schemaorg_col_idx = row.schemaorg_column_index
        schemaorg_label = row.schemaorg_label
        dbpedia_table_name = row.dbpedia_table_name
        dbpedia_col_idx = row.dbpedia_column_index
        dbpedia_label = row.dbpedia_label

        schemaorg_name_signature = set(qgram_transformer.transform(schemaorg_label))
        dbpedia_name_signature = set(qgram_transformer.transform(dbpedia_label))

        schemaorg_table = pd.read_json(os.path.join(schemaorg_table_dir, schemaorg_table_name), compression="gzip", lines=True)
        schemaorg_table = schemaorg_table.astype(str)

        if schemaorg_table.shape[0] < args.value_sample_size:
            schemaorg_values = schemaorg_table.iloc[:, schemaorg_col_idx].tolist()
        else:
            schemaorg_values = schemaorg_table.iloc[:, schemaorg_col_idx].sample(n=args.value_sample_size, random_state=args.random_seed).tolist()
        
        schemaorg_fasttext_signature = fasttext_transformer.transform(schemaorg_values)
        if len(schemaorg_fasttext_signature) == 0:
            logger.info("Cannot compute value signature for schemaorg values.")
            logger.info("schemaorg values: ", " ".join(map(str, schemaorg_values)))
            continue

        dbpedia_table = pd.read_json(os.path.join(dbpedia_table_dir, dbpedia_table_name), compression="gzip", lines=True)
        dbpedia_table = dbpedia_table.astype(str)

        if dbpedia_table.shape[0] < args.value_sample_size:
            dbpedia_values = dbpedia_table.iloc[:, dbpedia_col_idx].tolist()
        else:
            dbpedia_values = dbpedia_table.iloc[:, dbpedia_col_idx].sample(n=args.value_sample_size, random_state=args.random_seed).tolist()
        
        dbpedia_fasttext_signature = fasttext_transformer.transform(dbpedia_values)
        if len(dbpedia_fasttext_signature) == 0:
            logger.info("Cannot compute value signature for dbpedia values.")
            logger.info("dbpedia values: ", " ".join(map(str, dbpedia_values)))
            continue

        name_sim = compute_name_similarity(schemaorg_name_signature, dbpedia_name_signature)
        value_sim = compute_value_similarity(schemaorg_fasttext_signature, dbpedia_fasttext_signature)

        evaluation_data.append(([name_sim, value_sim], 0))
    
    with open(os.path.join(args.output_dir, "sotab_v2_cta_test_synthesized_microbenchmark.pkl"), "wb") as f:
        pickle.dump(evaluation_data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/ssd/congtj/openforge/sotab_v2", help="Path to the root data directory.")

    parser.add_argument("--split", type=str, default="Test", help="Split of raw data for synthesizing the microbenchmark.")
    
    parser.add_argument("--value_sample_size", type=int, default=10, help="Number of sample values per column for feature extraction.")
    
    parser.add_argument("--random_seed", type=int, default=12345, help="Random seed for sampling.")
    
    parser.add_argument("--output_dir", type=str, default="/ssd/congtj/openforge/sotab_v2/artifact", help="Path to the output directory.")

    parser.add_argument("--log_dir", type=str, default="./logs/sotab_synthesized_data", help="Directory to store logs.")
    args = parser.parse_args()

    random.seed(args.random_seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    
    cur_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_filepath = os.path.join(args.log_dir, f"{cur_datetime}.log")
    logger = get_logger(log_filepath)
    logger.info(args)
    
    label_mapping_filepath = os.path.join(args.output_dir, "sotab_v2_cta_test_synthesized")
    pos_df, neg_df = synthesize_pos_and_neg_pairs(args.data_dir, label_mapping_filepath)

    schemaorg_table_dir = os.path.join(args.data_dir, f"cta_schemaorg/{args.split}")
    dbpedia_table_dir = os.path.join(args.data_dir, f"cta_dbpedia/{args.split}")
    synthesize_evaluation_dataset(pos_df, neg_df, schemaorg_table_dir, dbpedia_table_dir, logger, args)
