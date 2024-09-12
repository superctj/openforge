import pymongo
import pandas as pd


ROOT_DOWNLOAD = "/home/jjxing/ssd/new_project/data"

# connect mongodb
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
column_semantics_evaluation = myclient["column_semantics_evaluation"]

# data.gov
data_gov_mongo = myclient["data_gov_mar_21_2023"]
data_gov_metadata_col = data_gov_mongo["metadataSimplified"]
data_gov_csv_file_col = data_gov_mongo["csvfile"]
data_gov_gpt_annotation_col = data_gov_mongo["GPTannotation"]
data_gov_denpendency_parse_col = data_gov_mongo["gpt_annotation_dp"]

# harward dataverse
dataverse_mongo = myclient["harward_dataverse"]
dataverse_metadata_col = dataverse_mongo["metadata"]
dataverse_datafile_metadata_col = dataverse_mongo["datafile"]
dataverse_gpt_annotation_col = dataverse_mongo["GPTannotation"]
dataverse_denpendency_parse_col = dataverse_mongo["gpt_annotation_dp"]

def convertTableIDToTableName(table_id: str):
    file_metadata = data_gov_csv_file_col.find_one({"_id": table_id})
    
    if file_metadata is None:
        raise Exception("No matched csvfile.")
    if file_metadata["loadable"] == False or file_metadata["file_path"] == '':
        raise FileNotFoundError
    
    return file_metadata['file_path']

def readCSVFileWithTableID(table_id: str, nrows=10, **kwargs):
    file_metadata = data_gov_csv_file_col.find_one({"_id": table_id})
    if file_metadata is None:
        raise Exception("No matched csvfile.")
    if file_metadata["loadable"] == False or file_metadata["file_path"] == '':
        raise FileNotFoundError
    df = pd.read_csv(ROOT_DOWNLOAD + file_metadata['file_path'], header=0, nrows=nrows, on_bad_lines="skip", **kwargs)
    return df