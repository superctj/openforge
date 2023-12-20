import random
import numpy as np

CONSTANT = 1e-9
VALUE_SIGNATURE_ATTEMPTS = 100


def compute_name_similarity(name_sig1, name_sig2):
    return len(name_sig1.intersection(name_sig2)) / len(name_sig1.union(name_sig2))


def compute_value_similarity(value_sig1, value_sig2):
    return np.dot(value_sig1, value_sig2) / (np.linalg.norm(value_sig1) * np.linalg.norm(value_sig2) + CONSTANT)
