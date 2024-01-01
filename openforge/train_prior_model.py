import argparse
import pickle
import random

import numpy as np

from sklearn import linear_model
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import extmath


class RidgeClassifierwithProba(linear_model.RidgeClassifier):
    def predict_proba(self, X):
        d = self.decision_function(X)
        d_2d = np.c_[-d, d]
        return extmath.softmax(d_2d)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arts_data", type=str, default="./data/arts_num-head-concepts-100.pkl", help="Path to the training and test data synthesized from ARTS.")

    parser.add_argument("--arts_test_size", type=float, default=0.25, help="Test size for the ARTS data.")
    
    parser.add_argument("--sotab_benchmark", type=str, default="/ssd/congtj/openforge/sotab_v2/artifact/sotab_v2_cta_test_synthesized_microbenchmark.pkl", help="Path to the SOTAB microbenchmark.")

    parser.add_argument("--random_seed", type=int, default=12345, help="Random seed.")

    parser.add_argument("--model_save_filepath", type=str, default="./models/rc_arts_num-head-concepts-100.pkl", help="Path to save the trained model")
    
    args = parser.parse_args()
    random.seed(args.random_seed)

    with open(args.arts_data, "rb") as f:
        arts_data = pickle.load(f)
        X_arts = np.array([x[0] for x in arts_data])
        y_arts = np.array([x[1] for x in arts_data])
        print(X_arts.shape)
        print(y_arts.shape)

    with open(args.sotab_benchmark, "rb") as f:
        sotab_benchmark = pickle.load(f)
        X_sotab = np.array([x[0] for x in sotab_benchmark])
        y_sotab = np.array([x[1] for x in sotab_benchmark])
        print(X_sotab.shape)
        print(y_sotab.shape)
        print("=" * 50)

    X_train, X_test, y_train, y_test = train_test_split(X_arts, y_arts, test_size=args.arts_test_size, random_state=args.random_seed, stratify=y_arts)

    # clf = RidgeClassifierwithProba(tol=1e-2, solver="sparse_cg")
    clf = GaussianProcessClassifier(kernel=1.0*RBF(1.0), random_state=args.random_seed)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_sotab_pred = clf.predict(X_sotab)
    print("ARTS training data shape: ", X_train.shape)
    print("ARTS test data shape: ", y_pred.shape)
    print("SOTAB test data shape: ", y_sotab_pred.shape)

    print(f"ARTS accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"SOTAB accuracy: {accuracy_score(y_sotab, y_sotab_pred):.2f}")
    
    confidence_scores = clf.predict_proba(X_test)
    print(confidence_scores.shape)
    print(confidence_scores[:5])
    print(confidence_scores[:5, 1])

    with open(args.model_save_filepath, "wb") as f:
        pickle.dump(clf, f)
