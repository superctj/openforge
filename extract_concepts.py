import time

import nltk
import openai

from nltk.corpus import wordnet
from pymongo import MongoClient

from chatgpt_engine import get_subject


def pos_tagging(descr):
    words = nltk.word_tokenize(descr)
    words = [word.lower() for word in words if word.isalnum()]
    # words = [lemmatizer.lemmatize(word) for word in words]
    print(words)
    tags = nltk.pos_tag(words)
    return tags


def extract_nouns(tags):
    nouns = [word for word, pos in tags if pos.startswith("NN")]
    return nouns


def wordnet_multi_lookup(words):
    all_linked_concepts = {}
    for word in words:
        linked_concepts = wordnet_lookup(word)
        all_linked_concepts[word] = linked_concepts
    return all_linked_concepts


def wordnet_lookup(word):
    concepts = []
    synsets = wordnet.synsets(word, pos=wordnet.NOUN)
    for synset in synsets:
        hypernyms = synset.hypernyms()
        for hyper in hypernyms:
            concepts.append((synset.lemma_names()[0], hyper.lemma_names()[0]))
    return concepts


def process_metadata(cols_field_name, cols_descr):
    all_linked_concepts = []
    for field_name, descr in zip(cols_field_name, cols_descr):
        linked_concepts = [] # wordnet_lookup(field_name)
        if not linked_concepts:
            descr = descr.replace("\n", "").lower().strip()
            try:
                subject = get_subject(descr)
            except openai.error.APIError:
                time.sleep(0.1)
                subject = get_subject(descr)

            tags = pos_tagging(subject)
            keywords = extract_nouns(tags)
            linked_concepts = wordnet_multi_lookup(keywords)
        else:
            linked_concepts = {field_name: linked_concepts}
        
        all_linked_concepts.append(linked_concepts)
    return all_linked_concepts


if __name__ == "__main__":
    # col_descr = "Borough of arrest. B(Bronx), S(Staten Island), K(Brooklyn), M(Manhattan), Q(Queens)"
    # col_descr = "Latitude coordinate for Global Coordinate System, WGS 1984, decimal degrees (EPSG 4326)"
    # # col_descr = "Perpetrator's race description"

    # tags = pos_tagging(col_descr)
    # print(tags)
    # keywords = extract_nouns(tags)
    # print(keywords)
    # linked_concepts = wordnet_multi_lookup(keywords)
    # print(linked_concepts)

    mongo_client = MongoClient("mongodb://localhost:27017/")
    db_name = "nyc_open_data"
    collection_name = "socrata_metadata"
    target_collection_name = "socrata_annotations"
    db = mongo_client[db_name]
    collection = db[collection_name]
    target_collection = db[target_collection_name]

    cursor = collection.find()
    for doc in cursor:
        doc_id = doc["resource"]["id"]
        cols_field_name = doc["resource"]["columns_field_name"]
        cols_descr = doc["resource"]["columns_description"]
        linked_concepts = process_metadata(cols_field_name, cols_descr)
        target_collection.insert_one({"_id": doc_id, "linked_concepts": linked_concepts})
