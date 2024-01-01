import numpy as np


CONSTANT = 1e-9


def jaccard_index(name_sig1: set, name_sig2: set):
    return len(name_sig1.intersection(name_sig2)) / len(name_sig1.union(name_sig2))


def cosine_similarity(value_sig1, value_sig2):
    return np.dot(value_sig1, value_sig2) / (np.linalg.norm(value_sig1) * np.linalg.norm(value_sig2) + CONSTANT)
