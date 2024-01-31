import argparse
import os
import random

from dataclasses import make_dataclass

import pandas as pd

from openforge.feature_extraction.fb_fasttext import FasttextTransformer
from openforge.feature_extraction.qgram import QGramTransformer
from openforge.feature_extraction.similarity_metrics import cosine_similarity, jaccard_index
from openforge.utils.custom_logging import get_custom_logger


def split_on_uppercase(s: str, lower: bool=True, keep_contiguous: bool=True):
    """
    Args:
        s (str): string
        keep_contiguous (bool): flag to indicate we want to 
                                keep contiguous uppercase chars together

    Returns:
    """

    string_length = len(s)
    is_lower_around = (lambda: s[i-1].islower() or 
                       string_length > (i + 1) and s[i + 1].islower())

    start = 0
    parts = []
    
    for i in range(1, string_length):
        if s[i].isupper() and (not keep_contiguous or is_lower_around()):
            if lower:
                parts.append(s[start: i].lower())
            else:
                parts.append(s[start: i])
            start = i
    
    if lower:
        parts.append(s[start:].lower())
    else:
        parts.append(s[start:])

    return parts


def process_schemaorg_label(label: str) -> str:
    processed_label = ""
    parts = label.split("/")

    for i, part in enumerate(parts):
        decomposed = split_on_uppercase(part)
        
        if i == 0:
            processed_label += " ".join(decomposed)
        else:
            processed_label += (" " + " ".join(decomposed))

    return processed_label


def get_column_values(table_path: str, column_index: int, sample_size: int, random_seed: int) -> list:
    table = pd.read_json(table_path, compression="gzip", lines=True)
    table = table.astype(str)

    if table.shape[0] <= sample_size:
        column_values = table.iloc[:, column_index].tolist()
    else:
        column_values = table.iloc[:, column_index].sample(n=sample_size, random_state=random_seed).tolist()
    
    return column_values


def compute_label_signatures(label: str, schemaorg_table_dir: str, dbpedia_table_dir: str, schemaorg_label_col_map: dict, dbpedia_label_col_map: dict, qgram_transformer, fasttext_transformer, args):
    # Check if label comes from dbpedia
    if label.startswith("https"): 
        table_name, column_index = dbpedia_label_col_map[label]
        table_path = os.path.join(dbpedia_table_dir, table_name)

        label = label.split("/")[-1].lower()
    else:
        table_name, column_index = schemaorg_label_col_map[label]
        table_path = os.path.join(schemaorg_table_dir, table_name)

        label = process_schemaorg_label(label)

    name_signature = set(qgram_transformer.transform(label))

    column_values = get_column_values(
        table_path,
        column_index,
        args.value_sample_size,
        args.random_seed
    )

    try:
        fasttext_signature = fasttext_transformer.transform(column_values)
    except:
        fasttext_signature = []
    
    return name_signature, fasttext_signature


def find_equivalent_labels(source_data_dir: str, logger) -> list:
    schemaorg_metadata_filepath = os.path.join(
        source_data_dir, "cta_schemaorg/sotab_v2_cta_test_set.csv")
    dbpedia_metadata_filepath = os.path.join(
        source_data_dir, "cta_dbpedia/sotab_cta_test_dbpedia.csv")

    schemaorg_df = pd.read_csv(schemaorg_metadata_filepath)
    dbpedia_df = pd.read_csv(dbpedia_metadata_filepath)

    EquivalentEntry = make_dataclass("EquivalentEntry", [
        ("table_name", str),
        ("column_index", int),
        ("schemaorg_label", str),
        ("dbpedia_label", str)
    ])
    equivalent_entries = []

    schemaorg_cols = set()
    schemaorg_col_label_mapping = {}
    dbpedia_labels = set()

    for row in schemaorg_df.itertuples():
        unique_id = row.table_name + str(row.column_index)
        schemaorg_cols.add(unique_id)
        schemaorg_col_label_mapping[unique_id] = row.label

    for row in dbpedia_df.itertuples():
        unique_id = row.table_name + str(row.column_index)
        
        if unique_id in schemaorg_cols:
            dbpedia_labels.add(row.label)
            equivalent_entries.append(EquivalentEntry(
                row.table_name,
                row.column_index,
                schemaorg_col_label_mapping[unique_id],
                row.label
            ))

            logger.info(f"schemaorg label: {schemaorg_col_label_mapping[unique_id]}")
            logger.info(f"dbpedia equivalent label: {row.label}")

    num_unique_schemaorg_labels = len(set(schemaorg_col_label_mapping.values()))
    
    logger.info(f"Number of unique schemaorg labels: {num_unique_schemaorg_labels}")
    logger.info(f"Number of unique equivalent dbpedia labels: {len(dbpedia_labels)}")

    return equivalent_entries


def synthesize_sotab_v2_mrf_data(equivalent_entries: list, args, logger):
    schemaorg_table_dir = os.path.join(args.source_data_dir, f"cta_schemaorg/{args.split}")
    dbpedia_table_dir = os.path.join(args.source_data_dir, f"cta_dbpedia/{args.split}")

    all_labels = []
    equivalent_pairs = set()
    schemaorg_label_col_map = {}
    dbpedia_label_col_map = {}

    for entry in equivalent_entries:
        if entry.schemaorg_label not in all_labels:
            all_labels.append(entry.schemaorg_label)
            schemaorg_label_col_map[entry.schemaorg_label] = (entry.table_name, entry.column_index)
        
        if entry.dbpedia_label not in all_labels:
            all_labels.append(entry.dbpedia_label)
            dbpedia_label_col_map[entry.dbpedia_label] = (entry.table_name, entry.column_index)
        
        equivalent_pairs.add((entry.schemaorg_label, entry.dbpedia_label))

    qgram_transformer = QGramTransformer(qgram_size=3)
    fasttext_transformer = FasttextTransformer(
        cache_dir=args.fasttext_model_dir
    )

    csv_output_filepath = os.path.join(
        args.output_dir, "sotab_v2_test_mrf_data.csv"
    )

    MRFEntry = make_dataclass("MRFEntry", [
        ("label_1", str),
        ("label_2", str),
        ("name_similarity", float),
        ("value_similarity", float),
        ("relation_variable_name", str),
        ("relation_variable_label", int)
    ])

    mrf_data = []

    for i, label_i in enumerate(all_labels):
        logger.info("\n" + "="*50)
        logger.info(f"Concept {i+1}: {label_i}")

        label_i_name_signature, label_i_fasttext_signature = compute_label_signatures(
            label_i,
            schemaorg_table_dir,
            dbpedia_table_dir,
            schemaorg_label_col_map,
            dbpedia_label_col_map,
            qgram_transformer,
            fasttext_transformer,
            args
        )
        
        if len(label_i_fasttext_signature) == 0:
            logger.info(f"Cannot compute fasttext signature for label: {label_i}.")
            continue

        for j in range(i+1, len(all_labels)):
            label_j = all_labels[j]
            logger.info(f"Concept {j+1}: {label_j}")

            label_j_name_signature, label_j_fasttext_signature = compute_label_signatures(
                label_j,
                schemaorg_table_dir,
                dbpedia_table_dir,
                schemaorg_label_col_map,
                dbpedia_label_col_map,
                qgram_transformer,
                fasttext_transformer,
                args
            )

            if len(label_j_fasttext_signature) == 0:
                logger.info(f"Cannot compute fasttext signature for label: {label_j}.")
                continue

            name_sim = jaccard_index(
                label_i_name_signature,
                label_j_name_signature
            )

            fasttext_sim = cosine_similarity(
                label_i_fasttext_signature,
                label_j_fasttext_signature
            )

            if (label_i, label_j) in equivalent_pairs or (label_j, label_i) in equivalent_pairs:
                relation_variable_label = 1
            else:
                relation_variable_label = 0
            
            relation_variable_name = f"R_{i+1}_{j+1}"
            mrf_data.append(MRFEntry(
                label_i,
                label_j,
                name_sim,
                fasttext_sim,
                relation_variable_name,
                relation_variable_label
            ))

    mrf_df = pd.DataFrame(mrf_data)
    mrf_df.to_csv(csv_output_filepath, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--source_data_dir", type=str, default="/ssd/congtj/openforge/sotab_v2", help="Path to the source data directory.")

    parser.add_argument("--split", type=str, default="Test", help="Split of source data.")

    parser.add_argument("--fasttext_model_dir", type=str, default="/ssd/congtj", help="Directory containing fasttext model weights.")
    
    parser.add_argument("--value_sample_size", type=int, default=10, help="Number of sample values per column for feature extraction.")
    
    parser.add_argument("--random_seed", type=int, default=12345, help="Random seed for sampling.")
    
    parser.add_argument("--output_dir", type=str, default="/ssd/congtj/openforge/sotab_v2/artifact", help="Path to the output directory.")

    parser.add_argument("--log_dir", type=str, default="/home/congtj/openforge/logs/sotab_synthesized_data", help="Directory to store logs.")
    
    args = parser.parse_args()

    # Fix random seed
    random.seed(args.random_seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    
    logger = get_custom_logger(args.log_dir)
    logger.info(args)

    equivalent_entries = find_equivalent_labels(args.source_data_dir, logger)
    synthesize_sotab_v2_mrf_data(equivalent_entries, args, logger)
