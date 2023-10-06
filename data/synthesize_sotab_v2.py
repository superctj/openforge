import argparse
import os
import pickle

from dataclasses import make_dataclass

import numpy as np
import pandas as pd

from d3l.indexing.feature_extraction.schema.qgram_transformer import QGramTransformer
from d3l.indexing.feature_extraction.values.fasttext_embedding_transformer import FasttextTransformer


def find_label_mapping(data_dir: str, output_filepath: str):
    schemaorg_metadata_filepath = os.path.join(data_dir, "cta_schemaorg/sotab_v2_cta_test_set.csv")
    dbpedia_metadata_filepath = os.path.join(data_dir, "cta_dbpedia/sotab_cta_test_dbpedia.csv")

    schemaorg_df = pd.read_csv(schemaorg_metadata_filepath)
    dbpedia_df = pd.read_csv(dbpedia_metadata_filepath)

    schemaorg_cols = set()
    schemaorg_col_label_mapping = {}
    OutputEntry = make_dataclass("OutputEntry", [("table_name", str), ("column_index", int), ("schemaorg_label", str), ("dbpedia_label", str)])
    output_data = []

    for row in schemaorg_df.itertuples():
        unique_id = row.table_name + "_" + str(row.column_index)
        schemaorg_cols.add(unique_id)
        schemaorg_col_label_mapping[unique_id] = row.label

    for row in dbpedia_df.itertuples():
        unique_id = row.table_name + "_" + str(row.column_index)
        if unique_id in schemaorg_cols:
            output_data.append(OutputEntry(row.table_name, row.column_index, schemaorg_col_label_mapping[unique_id], row.label))

    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_filepath, index=False)


def synthesize_evaluation_dataset(label_mapping_filepath: str, table_dir: str,  args: argparse.Namespace):
    label_mapping_df = pd.read_csv(label_mapping_filepath)
    qgram_transformer = QGramTransformer(qgram_size=3)
    fasttext_transformer = FasttextTransformer()
    evaluation_data = []
    
    for row in label_mapping_df.itertuples():
        table_name = row.table_name
        col_idx = row.column_index

        table = pd.read_json(os.path.join(table_dir, table_name), 
                             compression="gzip", lines=True)
        
        if table.shape[0] < args.value_sample_size * 2:
            sample_values = table.iloc[:, col_idx].tolist()
            sample_size = int(len(sample_values) / 2)
            schemaorg_values = sample_values[:sample_size]
            dbpedia_values = sample_values[sample_size:]
        else:
            sample_values = table.iloc[:, col_idx].sample(n=args.value_sample_size*2, random_state=args.sample_random_seed).tolist()

            schemaorg_values = sample_values[:args.value_sample_size]
            dbpedia_values = sample_values[args.value_sample_size:]

        schemaorg_name_signature = set(qgram_transformer.transform(row.schemaorg_label))
        schemaorg_fasttext_signature = fasttext_transformer.transform(schemaorg_values)

        dbpedia_label = row.dbpedia_label.split("/")[-1]
        dbpedia_name_signature = set(qgram_transformer.transform(dbpedia_label))
        dbpedia_fasttext_signature = fasttext_transformer.transform(dbpedia_values)

        name_sim = len(schemaorg_name_signature.intersection(dbpedia_name_signature)) / len(schemaorg_name_signature.union(dbpedia_name_signature))
        value_sim = np.dot(schemaorg_fasttext_signature, dbpedia_fasttext_signature) #/ (np.linalg.norm(schemaorg_fasttext_signature) * np.linalg.norm(dbpedia_fasttext_signature))

        evaluation_data.append(([name_sim, value_sim], 1))
    
    with open(os.path.join(args.output_dir, "sotab_v2_cta_test_synthesized_benchmark.pkl"), "wb") as f:
        pickle.dump(evaluation_data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/ssd/congtj/openforge/sotab_v2", help="Path to the root data directory")
    
    parser.add_argument("--value_sample_size", type=int, default=10, help="Number of sample values per column for feature extraction")
    
    parser.add_argument("--sample_random_seed", type=int, default=12345, help="Random seed for sampling")
    
    parser.add_argument("--output_dir", type=str, default="/ssd/congtj/openforge/sotab_v2/artifact", help="Path to the output directory")
    args = parser.parse_args()

    label_mapping_filepath = os.path.join(args.output_dir, "sotab_v2_cta_test_both_vocab.csv")
    if not os.path.exists(label_mapping_filepath):
        os.makedirs(args.output_dir, exist_ok=True)
        find_label_mapping(args.data_dir, label_mapping_filepath)
    else:
        table_dir = os.path.join(args.data_dir, "cta_dbpedia/Test")
        synthesize_evaluation_dataset(label_mapping_filepath, table_dir, args)
