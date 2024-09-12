import pickle
import requests

import nltk
from nltk.corpus import wordnet


WSD_URL = "http://nlp.uniroma1.it/amuse-wsd/api/model"
REQUEST_HEADERS = {
    "accept": "application/json",
    "Content-Type": "application/json"
}


def inspect_arts_taxonomy(filepath):
    with open(filepath, "rb") as f:
        taxonomy = pickle.load(f)
        print(type(taxonomy))


def word_sense_disambiguation(context, word):
    data = [{
        "text": context,
        "lang": "EN"
    }]

    response = requests.post(WSD_URL, headers=REQUEST_HEADERS, json=data)
    if response.status_code == 200:
        result = response.json()
        for token in result[0]["tokens"]:
            if token["text"] == word:
                return wordnet.synset(token["nltkSynset"])
    else:
        raise Exception(f"Error in WSD API call: {response.json()}")


def word_similarity(synset1, synset2, measure="path"):
    if measure == "path":
        return synset1.path_similarity(synset2)
    elif measure == "wup":
        return synset1.wup_similarity(synset2)
    

def is_hypernym(synset1, synset2):
    return synset1 in synset2.hypernyms()


if __name__ == "__main__":
    a = "total number of confirmed COVID-19 cases."
    b = "total count of COVID-19 cases."

    a_synset = word_sense_disambiguation(a, "number")
    b_synset = word_sense_disambiguation(b, "count")
    print(a_synset)
    print(b_synset)
    # print(word_similarity(a_synset, b_synset, "wup"))
    print(is_hypernym(a_synset, b_synset))
    print(is_hypernym(b_synset, a_synset))
