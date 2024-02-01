import argparse
import os
import pickle
import random

import numpy as np
import pandas as pd

from sklearn import linear_model
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import extmath

from openforge.utils.util import create_dir, get_proj_dir


class RidgeClassifierwithProba(linear_model.RidgeClassifier):
    def predict_proba(self, X):
        d = self.decision_function(X)
        d_2d = np.c_[-d, d]
        return extmath.softmax(d_2d)


def load_arts_data(data_filepath: str, args: argparse.Namespace):
    arts_df = pd.read_csv(data_filepath, delimiter=",", header=0)
    print("ARTS data shape: ", arts_df.shape)
    y_arts_df = arts_df.pop("relation_variable_label")

    X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(
        arts_df, y_arts_df,
        train_size=args.arts_train_prop,
        random_state=args.random_seed,
        stratify=y_arts_df
    )

    X_train = X_train_df[["name_similarity", "value_similarity"]].to_numpy()
    y_train = y_train_df.to_numpy()

    X_valid_df, X_test_df, y_valid_df, y_test_df = train_test_split(
        X_test_df, y_test_df,
        test_size=0.5,
        random_state=args.random_seed,
        stratify=y_test_df
    )

    X_valid = X_valid_df[["name_similarity", "value_similarity"]].to_numpy()
    y_valid = y_valid_df.to_numpy()

    X_test = X_test_df[["name_similarity", "value_similarity"]].to_numpy()
    y_test = y_test_df.to_numpy()

    X_test_df["relation_variable_label"] = y_test_df
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test, X_test_df


def load_sotab_data(data_filepath: str):
    sotab_df = pd.read_csv(data_filepath, delimiter=",", header=0)
    X_sotab, y_sotab = [], []

    for row in sotab_df.itertuples():
        X_sotab.append([row.name_similarity, row.value_similarity])
        y_sotab.append(row.relation_variable_label)
    
    X_sotab = np.array(X_sotab)
    y_sotab = np.array(y_sotab)

    print("\nX_sotab shape: ", X_sotab.shape)
    print("y_sotab shape: ", y_sotab.shape)

    return X_sotab, y_sotab, sotab_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default="train", help="Mode: train or test.")

    parser.add_argument("--arts_data", type=str, default="data/arts_mrf_synthesized_data_top-20-concepts/arts_mrf_data.csv", help="Path to the training and test data synthesized from ARTS.")

    parser.add_argument("--num_head_concepts", type=int, default=20, help="Number of head concepts considered in arts data.")

    parser.add_argument("--arts_train_prop", type=float, default=0.6, help="Training proportion of the ARTS data.")
    
    parser.add_argument("--sotab_benchmark", type=str, default="/ssd/congtj/openforge/sotab_v2/artifact/sotab_v2_test_mrf_data.csv", help="Path to the SOTAB benchmark.")

    parser.add_argument("--random_seed", type=int, default=12345, help="Random seed.")
    
    args = parser.parse_args()

    # fix random seed
    random.seed(args.random_seed)

    proj_dir = get_proj_dir(__file__, file_level=2)
    arts_data = os.path.join(proj_dir, args.arts_data)

    exp_dir = os.path.join(
        proj_dir, f"exps/arts_top-{args.num_head_concepts}-concepts"
    )
    if args.mode == "train":
        create_dir(exp_dir, force=True)
    
    model_save_filepath = os.path.join(exp_dir, "rc_model.pkl")

    X_train, y_train, X_valid, y_valid, X_test, y_test, test_df = load_arts_data(arts_data, args)

    if args.mode == "train":
        print("\nARTS training data shape: ", X_train.shape)
        print("Number of positive instances in training data: ", np.sum(y_train == 1))

        clf = RidgeClassifierwithProba(tol=1e-2, solver="sparse_cg")
        # clf = GaussianProcessClassifier(kernel=1.0*RBF(1.0), random_state=args.random_seed)
        clf.fit(X_train, y_train)
        
        y_valid_pred = clf.predict(X_valid)
        y_test_pred = clf.predict(X_test)

        print("\nARTS validation data shape: ", X_valid.shape)
        print("Number of positive instances in validation data: ", np.sum(y_valid == 1))

        print(f"ARTS validation accuracy: {accuracy_score(y_valid, y_valid_pred):.2f}")
        print(f"ARTS validation F1 score: {f1_score(y_valid, y_valid_pred):.2f}")

        print("\nARTS test data shape: ", X_test.shape)
        print("Number of positive instances in test data: ", np.sum(y_test == 1))

        print(f"ARTS test accuracy: {accuracy_score(y_test, y_test_pred):.2f}")
        print(f"ARTS test F1 score: {f1_score(y_test, y_test_pred):.2f}")

        test_confdc_scores = clf.predict_proba(X_test)
        print(test_confdc_scores.shape)
        print(test_confdc_scores[:5])

        with open(model_save_filepath, "wb") as f:
            pickle.dump(clf, f)

        test_df["positive_label_confidence_score"] = test_confdc_scores[:, 1]
        
        output_filepath = os.path.join(
            exp_dir, "arts_test_mrf_data_with_confidence_scores.csv"
        )
        test_df.to_csv(output_filepath, index=False)
    else:
        assert args.mode == "test"
        with open(model_save_filepath, "rb") as f:
            clf = pickle.load(f)

        # ARTS test data evaluation
        y_pred = clf.predict(X_test)
        print("ARTS test data shape: ", X_test.shape)
        print("Number of positive instances in test data: ", np.sum(y_test == 1))
        print(f"ARTS test accuracy: {accuracy_score(y_test, y_pred):.2f}")
        print(f"ARTS F1 score: {f1_score(y_test, y_pred):.2f}")

        arts_confdc_scores = clf.predict_proba(X_test)
        print("Confidence scores shape: ", arts_confdc_scores.shape)
        print(arts_confdc_scores[:5])
        
        # SOTAB benchmark evaluation
        X_sotab, y_sotab, sotab_df = load_sotab_data(args.sotab_benchmark)

        y_sotab_pred = clf.predict(X_sotab)
        print(f"\nSOTAB benchmark accuracy: {accuracy_score(y_sotab, y_sotab_pred):.2f}")
        print(f"SOTAB benchmark F1 score: {f1_score(y_sotab, y_sotab_pred):.2f}")

        sotab_confdc_scores = clf.predict_proba(X_sotab)
        print("SOTAB confidence scores shape: ", sotab_confdc_scores.shape)
        print(sotab_confdc_scores[:5])

        sotab_df["positive_label_confidence_score"] = sotab_confdc_scores[:, 1]
        
        output_filepath = os.path.join(
            exp_dir, "sotab_v2_test_mrf_data_with_confidence_scores.csv"
        )
        sotab_df.to_csv(output_filepath, index=False)
