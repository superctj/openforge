import argparse
import os
import pickle

import numpy as np

from d3l.indexing.feature_extraction.schema.qgram_transformer import QGramTransformer
from d3l.indexing.feature_extraction.values.fasttext_embedding_transformer import FasttextTransformer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--sotab_benchmark", type=str, default="/ssd/congtj/openforge/sotab_v2/artifact/sotab_v2_cta_test_synthesized_microbenchmark.pkl", help="Path to the SOTAB benchmark data")

    args = parser.parse_args()

    with open(args.sotab_benchmark, "rb") as f:
        sotab_benchmark = pickle.load(f)
    
    X_sotab = np.array([x[0] for x in sotab_benchmark])
    y_sotab = np.array([x[1] for x in sotab_benchmark])

    thresholds = [thres/10 for thres in range(1, 10)]
    print(thresholds)
    name_heuristics = [0] * len(thresholds)
    value_heuristics = [0] * len(thresholds)

    for i in range(X_sotab.shape[0]):
        for j, thres in enumerate(thresholds):
            if X_sotab[i][0] > thres:
                name_heuristics[j] += 1
            if X_sotab[i][1] > thres:
                value_heuristics[j] += 1
    
    print(np.array(name_heuristics) / X_sotab.shape[0])
    print(np.array(value_heuristics) / X_sotab.shape[0])
