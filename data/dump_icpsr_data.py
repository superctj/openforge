import os

import pandas as pd
from pymongo import MongoClient


if __name__ == "__main__":
    client = MongoClient("mongodb://localhost:27017/")
    db_name = "icpsr"
    collection_prefix = "level3-data"
    db = client[db_name]

    data_dir = "/ssd/congtj/icpsr_data/level3_data/Data-Uploads-ACW2228-ACW2229/"

    for study in os.listdir(data_dir):
        study_dir = os.path.join(data_dir, study)
        if os.path.isdir(study_dir):
            collection_name = f"{collection_prefix}_{study}"
            collection = db[collection_name]
            if collection_name in db.list_collection_names():
                collection.drop()
            
            for file_name in os.listdir(study_dir):
                if file_name.endswith(".tsv"):
                    file_path = os.path.join(study_dir, file_name)
                    try:
                        table = pd.read_csv(file_path, delimiter="\t", on_bad_lines="skip")
                    except UnicodeDecodeError:
                        table = pd.read_csv(file_path, delimiter="\t", encoding ="ISO-8859-1", on_bad_lines="skip")
                    except pd.errors.ParserError:
                        table = pd.read_csv(file_path, delimiter="\t", engine="python", on_bad_lines="skip")
                    except Exception as e:
                        print(f"Error in reading table from {file_path}: {e}")

                    # Drop empty columns
                    table.dropna(axis="columns", how="all", inplace=True)
                    # Drop empty rows
                    table.dropna(axis="index", how="all", inplace=True)
            
                    try:
                        collection.insert_many(table.to_dict("records"))
                    except Exception as e:
                        # MongoDB can only handle up to 8-byte ints
                        print(f"Error in inserting documents to MongoDB: {e} when processing {file_path}")
                        continue
    client.close()
