from ARTS.helpers.mongodb_helper import dataverse_metadata_col, dataverse_datafile_metadata_col
import requests
import urllib.parse
import json
from tenacity import retry, wait_random_exponential, stop_after_attempt

# q=*&type=dataset&sort=name&order=asc&start=100
# https://dataverse.harvard.edu/dataverse/harvard?q=&fq1=fileTypeGroupFacet%3A%22Data%22&fq0=dvObjectType%3A%28files%29&types=files&sort=dateSort&order=
# https://dataverse.harvard.edu/api/search?q=*&type=file&sort=date&order=desc&start=100&fq=fileTypeGroupFacet%3A%22Tabular+Data%22

SEARCH_API = "https://dataverse.harvard.edu/api/search?"
params = {
    "q": "*",
    "type": "file",
    "sort": "date",
    "order": "asc",
    "fq": "fileTypeGroupFacet:\"Tabular Data\"",
    "per_page": 1000,
    "start": 0
}

url = SEARCH_API + urllib.parse.urlencode(params)

@retry(wait=wait_random_exponential(min=5, max=20), stop=stop_after_attempt(6))
def request_metadata(url):
    rst = requests.get(url).json()
    return rst    

print("Processing ", url)
rst = request_metadata(url)
print(rst['data']['total_count'])

while len(rst['data']['items']) == 1000:
    dataverse_datafile_metadata_col.insert_many(rst['data']['items'])

    print("{} datasets done.".format(dataverse_datafile_metadata_col.count_documents({})))
    params["start"] += 1000
    url = SEARCH_API + urllib.parse.urlencode(params)
    print("Processing ", url)
    rst = request_metadata(url)

dataverse_datafile_metadata_col.insert_many(rst['data']['items'])
print("{} datasets done.".format(dataverse_datafile_metadata_col.count_documents({})))