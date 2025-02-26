import argparse
import os
import random

from dataclasses import make_dataclass

import pandas as pd

from openforge.feature_extraction.fb_fasttext import FasttextTransformer
from openforge.feature_extraction.qgram import QGramTransformer
from openforge.feature_extraction.similarity_metrics import (
    cosine_similarity,
    edit_distance,
    jaccard_index,
)
from openforge.utils.custom_logging import create_custom_logger


VALUE_SIGNATURE_ATTEMPTS = 100


def split_on_uppercase(
    s: str, lower: bool = True, keep_contiguous: bool = True
) -> list[str]:
    """
    Args:
        s: string
        keep_contiguous: flag to indicate we want to keep contiguous uppercase
            chars together

    Returns:
        List of substrings split on uppercase.
    """

    string_length = len(s)
    is_lower_around = (
        lambda: s[i - 1].islower()
        or string_length > (i + 1)
        and s[i + 1].islower()
    )

    start = 0
    parts = []

    for i in range(1, string_length):
        if s[i].isupper() and (not keep_contiguous or is_lower_around()):
            if lower:
                parts.append(s[start:i].lower())
            else:
                parts.append(s[start:i])
            start = i

    if lower:
        parts.append(s[start:].lower())
    else:
        parts.append(s[start:])

    return parts


def process_label(label: str) -> str:
    processed_label = ""
    parts = label.split("/")

    for i, part in enumerate(parts):
        decomposed = split_on_uppercase(part)

        # Remove suffix "at" that stands for addtional type in SOTAB benchmark
        if decomposed[-1] == "at":
            decomposed = decomposed[:-1]

        if i == 0:
            processed_label += " ".join(decomposed)
        else:
            processed_label += " " + " ".join(decomposed)

    return processed_label


def get_value_signatures(
    tblname_colidx_pairs: list[tuple], table_dir: str, fasttext_model
):
    # randomly pick a corresponding table column to compute value signatures
    count = 0

    while count < VALUE_SIGNATURE_ATTEMPTS:
        rnd_idx = random.randrange(len(tblname_colidx_pairs))
        table_name, col_idx = tblname_colidx_pairs[rnd_idx]

        table_path = os.path.join(table_dir, table_name)
        column_values = get_column_values(table_path, col_idx)
        fasttext_signature = fasttext_model.transform(column_values)

        if len(fasttext_signature) != 0:
            break
        else:
            count += 1

    return column_values, fasttext_signature, table_path, col_idx


def get_column_values(table_path: str, column_index: int) -> list:
    table = pd.read_json(table_path, compression="gzip", lines=True)
    table = table.astype(str)

    # if table.shape[0] <= sample_size:
    #     column_values = table.iloc[:, column_index].tolist()
    # else:
    #     column_values = (
    #         table.iloc[:, column_index]
    #         .sample(n=sample_size, random_state=random_seed)
    #         .tolist()
    #     )
    column_values = table.iloc[:, column_index].tolist()

    return column_values


def get_label_signatures(
    label: str,
    schemaorg_table_dir: str,
    dbpedia_table_dir: str,
    schemaorg_label_col_map: dict,
    dbpedia_label_col_map: dict,
    qgram_transformer,
    fasttext_transformer,
    logger,
):
    # Check if label comes from dbpedia
    if label.startswith("https"):
        tblname_colidx_pairs = dbpedia_label_col_map[label]
        table_dir = dbpedia_table_dir

        label = process_label(label.split("/")[-1])
    else:
        tblname_colidx_pairs = schemaorg_label_col_map[label]
        table_dir = schemaorg_table_dir

        label = process_label(label)

    logger.info(f"Processed label: {label}")

    name_qgram_signature = set(qgram_transformer.transform(label))
    name_fasttext_signature = fasttext_transformer.transform([label])

    col_values, value_fasttext_signature, table_path, col_idx = (
        get_value_signatures(
            tblname_colidx_pairs, table_dir, fasttext_transformer
        )
    )

    return (
        label,
        col_values,
        table_path,
        col_idx,
        name_qgram_signature,
        name_fasttext_signature,
        value_fasttext_signature,
    )


def find_equivalent_labels(
    source_data_dir: str, dataset_split: str, logger
) -> list:
    schemaorg_metadata_filepath = os.path.join(
        source_data_dir,
        f"cta_{dataset_split}_schemaorg/sotab_v2_cta_{dataset_split}_set.csv",
    )
    dbpedia_metadata_filepath = os.path.join(
        source_data_dir,
        f"cta_{dataset_split}_dbpedia/sotab_cta_{dataset_split}_dbpedia.csv",
    )

    schemaorg_df = pd.read_csv(schemaorg_metadata_filepath)
    dbpedia_df = pd.read_csv(dbpedia_metadata_filepath)

    EquivalentEntry = make_dataclass(
        "EquivalentEntry",
        [
            ("table_name", str),
            ("column_index", int),
            ("schemaorg_label", str),
            ("dbpedia_label", str),
        ],
    )

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
            equivalent_entries.append(
                EquivalentEntry(
                    row.table_name,
                    row.column_index,
                    schemaorg_col_label_mapping[unique_id],
                    row.label,
                )
            )

            logger.info(
                f"schemaorg label: {schemaorg_col_label_mapping[unique_id]}"
            )
            logger.info(f"dbpedia equivalent label: {row.label}")

    num_unique_schemaorg_labels = len(set(schemaorg_col_label_mapping.values()))

    logger.info(
        f"Number of unique schemaorg labels: {num_unique_schemaorg_labels}"
    )
    logger.info(
        f"Number of unique equivalent dbpedia labels: {len(dbpedia_labels)}"
    )

    return equivalent_entries


def synthesize_sotab_v2_mrf_data(
    equivalent_entries: list, args: argparse.Namespace, logger
):
    schemaorg_table_dir = os.path.join(
        args.source_data_dir, f"cta_{args.split}_schemaorg/{args.split}"
    )
    dbpedia_table_dir = os.path.join(
        args.source_data_dir, f"cta_{args.split}_dbpedia/{args.split}"
    )

    all_labels = []
    equivalent_pairs = set()
    schemaorg_label_col_map = {}
    dbpedia_label_col_map = {}

    for entry in equivalent_entries:
        if entry.schemaorg_label not in all_labels:
            all_labels.append(entry.schemaorg_label)
            schemaorg_label_col_map[entry.schemaorg_label] = [
                (entry.table_name, entry.column_index)
            ]
        else:
            schemaorg_label_col_map[entry.schemaorg_label].append(
                (entry.table_name, entry.column_index)
            )

        if entry.dbpedia_label not in all_labels:
            all_labels.append(entry.dbpedia_label)
            dbpedia_label_col_map[entry.dbpedia_label] = [
                (entry.table_name, entry.column_index)
            ]
        else:
            dbpedia_label_col_map[entry.dbpedia_label].append(
                (entry.table_name, entry.column_index)
            )

        equivalent_pairs.add((entry.schemaorg_label, entry.dbpedia_label))

    qgram_transformer = QGramTransformer(qgram_size=3)
    fasttext_transformer = FasttextTransformer(
        cache_dir=args.fasttext_model_dir
    )

    MRFEntry = make_dataclass(
        "MRFEntry",
        [
            ("label_1", str),
            ("label_2", str),
            ("label_1_processed", str),
            ("label_2_processed", str),
            ("label_1_table_path", str),
            ("label_2_table_path", str),
            ("label_1_col_idx", int),
            ("label_2_col_idx", int),
            ("name_qgram_similarity", float),
            ("name_jaccard_similarity", float),
            ("name_edit_distance", int),
            ("name_fasttext_similarity", float),
            ("name_word_count_ratio", float),
            ("name_char_count_ratio", float),
            ("value_jaccard_similarity", float),
            ("value_fasttext_similarity", float),
            ("relation_variable_name", str),
            ("relation_variable_label", int),
        ],
    )

    mrf_data = []

    for i, label_i in enumerate(all_labels):
        logger.info("\n" + "=" * 50)
        logger.info(f"Concept {i+1}: {label_i}")

        (
            label_i_name,
            label_i_col_values,
            label_i_table_path,
            label_i_col_idx,
            label_i_name_qgram_signature,
            label_i_name_fasttext_signature,
            label_i_value_fasttext_signature,
        ) = get_label_signatures(
            label_i,
            schemaorg_table_dir,
            dbpedia_table_dir,
            schemaorg_label_col_map,
            dbpedia_label_col_map,
            qgram_transformer,
            fasttext_transformer,
            logger,
        )

        if len(label_i_name_fasttext_signature) == 0:
            logger.info(
                f"Cannot compute name fasttext signature for label: {label_i}."
            )
            continue

        if len(label_i_value_fasttext_signature) == 0:
            logger.info(
                f"Cannot compute value fasttext signature for label: {label_i}."
            )
            continue

        for j in range(i + 1, len(all_labels)):
            label_j = all_labels[j]
            logger.info(f"Concept {j+1}: {label_j}")

            (
                label_j_name,
                label_j_col_values,
                label_j_table_path,
                label_j_col_idx,
                label_j_name_qgram_signature,
                label_j_name_fasttext_signature,
                label_j_value_fasttext_signature,
            ) = get_label_signatures(
                label_j,
                schemaorg_table_dir,
                dbpedia_table_dir,
                schemaorg_label_col_map,
                dbpedia_label_col_map,
                qgram_transformer,
                fasttext_transformer,
                logger,
            )

            if len(label_j_name_fasttext_signature) == 0:
                logger.info("Cannot compute name fasttext signature.")
                continue

            if len(label_j_value_fasttext_signature) == 0:
                logger.info("Cannot compute value fasttext signature.")
                continue

            name_qgram_sim = jaccard_index(
                label_i_name_qgram_signature, label_j_name_qgram_signature
            )
            name_jaccard_sim = jaccard_index(
                set(label_i_name.split()), set(label_j_name.split())
            )
            name_edit_dist = edit_distance(label_i_name, label_j_name)
            name_fasttext_sim = cosine_similarity(
                label_i_name_fasttext_signature, label_j_name_fasttext_signature
            )
            name_word_count_ratio = len(label_i_name.split()) / len(
                label_j_name.split()
            )
            name_char_count_ratio = len(label_i_name) / len(label_j_name)

            value_jaccard_sim = jaccard_index(
                set(label_i_col_values), set(label_j_col_values)
            )
            value_fasttext_sim = cosine_similarity(
                label_i_value_fasttext_signature,
                label_j_value_fasttext_signature,
            )

            relation_variable_name = f"R_{i+1}-{j+1}"

            if (label_i, label_j) in equivalent_pairs or (
                label_j,
                label_i,
            ) in equivalent_pairs:
                relation_variable_label = 1
            else:
                relation_variable_label = 0

            mrf_data.append(
                MRFEntry(
                    label_i,
                    label_j,
                    label_i_name,
                    label_j_name,
                    label_i_table_path,
                    label_j_table_path,
                    label_i_col_idx,
                    label_j_col_idx,
                    name_qgram_sim,
                    name_jaccard_sim,
                    name_edit_dist,
                    name_fasttext_sim,
                    name_word_count_ratio,
                    name_char_count_ratio,
                    value_jaccard_sim,
                    value_fasttext_sim,
                    relation_variable_name,
                    relation_variable_label,
                )
            )

    csv_output_filepath = os.path.join(
        args.output_dir, f"sotab_v2_{args.split}_vocabulary.csv"
    )
    mrf_df = pd.DataFrame(mrf_data)
    mrf_df.to_csv(csv_output_filepath, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source_data_dir",
        type=str,
        default="/ssd/congtj/openforge/sotab_v2",
        help="Path to the source data directory.",
    )

    parser.add_argument(
        "--split", type=str, default="test", help="Split of source data."
    )

    parser.add_argument(
        "--fasttext_model_dir",
        type=str,
        default="/ssd/congtj",
        help="Directory containing fasttext model weights.",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=12345,
        help="Random seed for sampling.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/ssd/congtj/openforge/sotab_v2/artifact",
        help="Directory to save outputs.",
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="/home/congtj/openforge/logs/sotab_v2",
        help="Directory to save logs.",
    )

    args = parser.parse_args()

    # Fix random seed
    random.seed(args.random_seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    logger = create_custom_logger(args.log_dir)
    logger.info(f"Running program: {__file__}")
    logger.info(f"{args}\n")

    equivalent_entries = find_equivalent_labels(
        args.source_data_dir, args.split, logger
    )
    synthesize_sotab_v2_mrf_data(equivalent_entries, args, logger)
