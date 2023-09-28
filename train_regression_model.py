import argparse
import pickle

import numpy as np

from sklearn import linear_model
from sklearn.metrics import accuracy_score, r2_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_filepath", type=str, default="./data/exp_tr_te_data.pkl", help="Path to the training and testing data")
    
    parser.add_argument("--model_save_filepath", type=str, default="./data/exp_model.pkl", help="Path to save the trained model")
    
    args = parser.parse_args()

    with open(args.data_filepath, "rb") as f:
        tr_te_data = pickle.load(f)

    num_train = int(len(tr_te_data) * 0.8)
    X_train = np.array([x[0] for x in tr_te_data[:num_train]])
    y_train = np.array([x[1] for x in tr_te_data[:num_train]])
    X_test = np.array([x[0] for x in tr_te_data[num_train:]])
    y_test = np.array([x[1] for x in tr_te_data[num_train:]])

    # clf = linear_model.LogisticRegression(random_state=0, max_iter=1000).fit(X_train, y_train)
    clf = linear_model.RidgeClassifier(tol=1e-2, solver="sparse_cg")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(X_train.shape)
    print(y_pred.shape)
    # y_pred = np.array([1 if pred > 0.5 else 0 for pred in y_pred])
    # print(f"Coefficient of determination: {r2_score(y_test, y_pred):.2f}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    # with open(args.model_save_filepath, "wb") as f:
    #     pickle.dump(clf, f)
