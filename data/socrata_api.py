import os

from pymongo import MongoClient
from sodapy import Socrata


if __name__ == "__main__":
    api_domain = "data.cityofnewyork.us" # "opendata.socrata.com"
    app_token = os.environ.get("SOCRATA_APP_TOKEN")
    client = Socrata(api_domain, app_token)
    limit = 1000
    offset = 0

    mongo_client = MongoClient("mongodb://localhost:27017/")
    db_name = "nyc_open_data"
    collection_name = "socrata_metadata"
    db = mongo_client[db_name]
    collection = db[collection_name]

    while True:
        try:
            datasets = client.datasets(limit=limit, offset=offset)
            collection.insert_many(datasets)
        except Exception as e:
            num_left = int(str(e).split(".")[-2].split()[-1])
            datasets = client.datasets(limit=num_left, offset=offset)
            collection.insert_many(datasets)
            break
             
        offset += limit
