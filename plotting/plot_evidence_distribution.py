import argparse
import pickle

import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arts_data", type=str, default="../data/arts_num-head-concepts-100.pkl", help="Path to the training and test data synthesized from ARTS.")
    args = parser.parse_args()

    with open(args.arts_data, "rb") as f:
        arts_data = pickle.load(f)
        X_arts = np.array([x[0] for x in arts_data])
        y_arts = np.array([x[1] for x in arts_data])
        print(X_arts.shape)
        print(y_arts.shape)
    
    idx_0 = np.where(y_arts == 0)
    idx_1 = np.where(y_arts == 1)
    
    fig, ax = plt.subplots()
    ax.scatter(X_arts[idx_0, 0], X_arts[idx_0, 1], c="b", label="negative")
    ax.scatter(X_arts[idx_1, 0], X_arts[idx_1, 1], c="r", label="positive")
    
    ax.set_xlabel("Name Similarity")
    ax.set_ylabel("Value Similarity")
    ax.legend(bbox_to_anchor=(0.3, 1.1), loc="upper left", ncol=2)
    ax.grid()

    fig.tight_layout()
    plt.savefig("./plots/evidence_distribution.png")
