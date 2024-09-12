import os
import shutil

import pandas as pd
import pymongo


ROOT_DOWNLOAD = "/home/jjxing/ssd/new_project/data"
SELECTED_DATA_DIR = "/ssd/congtj/openforge/arts/artifact/verified_multi_relations_top_6_nodes/training_prop_0.5"  # "/ssd/congtj/openforge/arts/artifact/multi_relations_top_11_nodes/training_prop_0.5"  # "/ssd/congtj/openforge/arts/artifact/top_40_nodes/training_prop_0.5"  # noqa: E501
DESTINATION_DIR = "/ssd2/congtj/openforge/arts/verified_multi_relations_nyc_selected_tables"  # "/ssd/congtj/openforge/arts/multi_relation_nyc_selected_tables"  # "/ssd/congtj/openforge/arts/nyc_selected_tables" # noqa: E501
EXIST_TABLE_IDS = set()


def copy_tables_from_split(
    split_filepath: str, data_gov_csv_file_col: pymongo.collection.Collection
):
    df = pd.read_csv(split_filepath)
    table1_paths, table2_paths = [], []

    for row in df.itertuples():
        table1_id = row.concept_1_table_id
        table2_id = row.concept_2_table_id

        file1_metadata = data_gov_csv_file_col.find_one({"_id": table1_id})
        file2_metadata = data_gov_csv_file_col.find_one({"_id": table2_id})
        table1_local_path = file1_metadata["file_path"]
        table2_local_path = file2_metadata["file_path"]

        table1_paths.append(table1_local_path)
        table2_paths.append(table2_local_path)

        if table1_id not in EXIST_TABLE_IDS:
            EXIST_TABLE_IDS.add(table1_id)
            table1_path = os.path.join(ROOT_DOWNLOAD, table1_local_path[1:])

            try:
                shutil.copy2(table1_path, DESTINATION_DIR)
            except OSError as e:
                print(f"Error copying {table1_path}: {e}")

        if table2_id not in EXIST_TABLE_IDS:
            EXIST_TABLE_IDS.add(table2_id)
            table2_path = os.path.join(ROOT_DOWNLOAD, table2_local_path[1:])
            try:
                shutil.copy(table2_path, DESTINATION_DIR)
            except OSError as e:
                print(f"Error copying {table2_path}: {e}")

    df["concept_1_table_path"] = table1_paths
    df["concept_2_table_path"] = table2_paths

    output_filepath = split_filepath[:-4] + "_with_table_paths.csv"
    df.to_csv(output_filepath, index=False)


if __name__ == "__main__":
    client = pymongo.MongoClient("mongodb://localhost:27017/")

    data_gov_mongo = client["data_gov_mar_21_2023"]
    data_gov_csv_file_col = data_gov_mongo["csvfile"]

    if not os.path.exists(DESTINATION_DIR):
        os.makedirs(DESTINATION_DIR)

    for split in os.listdir(SELECTED_DATA_DIR):
        if split.endswith(".csv"):
            split_filepath = os.path.join(SELECTED_DATA_DIR, split)
            copy_tables_from_split(split_filepath, data_gov_csv_file_col)
