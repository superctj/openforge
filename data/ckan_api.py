import os
import requests

import ckanapi
from pymongo import MongoClient


if __name__ == "__main__":
    # API_KEY = os.environ.get("DATA_GOV_API_KEY")
    # BASE_URL = "https://catalog.data.gov/api/3"

    # with ckanapi.RemoteCKAN(BASE_URL, apikey=API_KEY) as ckan_client:
    #     # publisher_name = "city-of-new-york"
    #     query = "organization:city-of-new-york"
    #     publisher_metadata = ckan_client.action.package_search(q=query, rows=10)
    #     # organization_show(id=publisher_name)
    #     print(publisher_metadata)
    
    base_url = "https://catalog.data.gov/api/3"
    api = "action/package_search?"
    params = {"q": "organization:city-of-new-york"}
    # query = "action/package_search?q=organization:city-of-new-york"

    response = requests.get(url=os.path.join(base_url, api), params=params)
    num_datasets = response.json()["result"]["count"]

    mongo_client = MongoClient("mongodb://localhost:27017/")
    db_name = "nyc_open_data"
    collection_name = "ckan_metadata"
    db = mongo_client[db_name]
    collection = db[collection_name]

    offset = 0
    limit = 1000
    while offset < num_datasets:
        params = {"q": "organization:city-of-new-york", "start": offset, "rows": limit}

        response = requests.get(url=os.path.join(base_url, api), params=params)
        results = response.json()["result"]["results"]
        collection.insert_many(results)

        offset += limit
