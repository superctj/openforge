import argparse
import os
import pickle
import random

import numpy as np
import pandas as pd

from sklearn import linear_model
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import extmath

from openforge.utils.util import create_dir, get_proj_dir


class RidgeClassifierwithProba(linear_model.RidgeClassifier):
    def predict_proba(self, X):
        d = self.decision_function(X)
        d_2d = np.c_[-d, d]
        return extmath.softmax(d_2d)


def prepare_sotab_data_for_inference(data_filepath: str, random_seed):
    if data_filepath.endswith(".pkl"):
        with open(data_filepath, "rb") as f:
            sotab_benchmark = pickle.load(f)
            X_sotab = np.array([x[0] for x in sotab_benchmark])
            y_sotab = np.array([x[1] for x in sotab_benchmark])
    else:
        assert data_filepath.endswith(".csv")

        sotab_df = pd.read_csv(data_filepath, delimiter=",", header=0)

        X_sotab = []
        y_sotab = []

        pos_sotab_df = sotab_df[sotab_df.relation_variable_label == 1]
        for row in pos_sotab_df.itertuples():
            assert row.relation_variable_label == 1
            X_sotab.append([row.name_similarity, row.value_similarity])
            y_sotab.append(row.relation_variable_label)
        
        # Make a balanced test set as the accuracy will be extremely high for the original imbalanced test set
        neg_sotab_df = sotab_df[sotab_df.relation_variable_label == 0]
        sampled_neg_sotab_df = neg_sotab_df.sample(
            n=len(X_sotab),
            replace=False,
            random_state=random_seed
        )

        for row in sampled_neg_sotab_df.itertuples():
            assert row.relation_variable_label == 0
            X_sotab.append([row.name_similarity, row.value_similarity])
            y_sotab.append(row.relation_variable_label)
        
        X_sotab = np.array(X_sotab)
        y_sotab = np.array(y_sotab)
    
    print("\nX_sotab shape: ", X_sotab.shape)
    print("y_sotab shape: ", y_sotab.shape)
    balanced_sotab_df = pd.concat([pos_sotab_df, sampled_neg_sotab_df])

    return X_sotab, y_sotab, balanced_sotab_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default="test", help="Mode: train or test.")

    parser.add_argument("--arts_data", type=str, default="data/arts_top-100-concepts_evidence/arts_evidence.pkl", help="Path to the training and test data synthesized from ARTS.")

    parser.add_argument("--num_head_concepts", type=int, default=100, help="Number of head concepts considered in arts data.")

    parser.add_argument("--num_pairs", type=int, default=0, help="Number of concept pairs to consider (0 for considering all pairs).")

    parser.add_argument("--arts_test_size", type=float, default=0.4, help="Test size for the ARTS data.")
    
    parser.add_argument("--sotab_benchmark", type=str, default="/ssd/congtj/openforge/sotab_v2/artifact/sotab_v2_test_mrf_data.csv", help="Path to the SOTAB benchmark.")

    parser.add_argument("--random_seed", type=int, default=12345, help="Random seed.")

    # parser.add_argument("--model_save_filepath", type=str, default="", help="Path to save the trained model")
    
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

    with open(arts_data, "rb") as f:
        arts_data = pickle.load(f)
        print("Number of arts data points: ", len(arts_data))
        
        if args.num_pairs > 0:
            arts_data = arts_data[:args.num_pairs]

        X_arts = np.array([x[0] for x in arts_data])
        y_arts = np.array([x[1] for x in arts_data])
        print("X_arts shape: ", X_arts.shape)
        print("y_arts shape: ", y_arts.shape)

    X_train, X_test, y_train, y_test = train_test_split(X_arts, y_arts,
        test_size=args.arts_test_size, random_state=args.random_seed,
        stratify=y_arts)

    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test,
        test_size=0.5, random_state=args.random_seed, stratify=y_test)

    if args.mode == "train":
        clf = RidgeClassifierwithProba(tol=1e-2, solver="sparse_cg")
        # clf = GaussianProcessClassifier(kernel=1.0*RBF(1.0), random_state=args.random_seed)
        clf.fit(X_train, y_train)
        
        y_valid_pred = clf.predict(X_valid)
        y_test_pred = clf.predict(X_test)
        print("\nARTS training data shape: ", X_train.shape)
        print("Number of positive instances in training data: ", np.sum(y_train == 1))

        print("\nARTS validation data shape: ", X_valid.shape)
        print("Number of positive instances in validation data: ", np.sum(y_valid == 1))

        print("\nARTS test data shape: ", X_test.shape)
        print("Number of positive instances in test data: ", np.sum(y_test == 1))
        
        print(f"\nARTS validation accuracy: {accuracy_score(y_valid, y_valid_pred):.2f}")
        print(f"ARTS test accuracy: {accuracy_score(y_test, y_test_pred):.2f}")

        test_confdc_scores = clf.predict_proba(X_test)
        print(test_confdc_scores.shape)
        print(test_confdc_scores[:5])

        with open(model_save_filepath, "wb") as f:
            pickle.dump(clf, f)

        arts_confdc_scores_save_filepath = os.path.join(
            exp_dir, "arts_confidence_scores.pkl"
        )
        with open(arts_confdc_scores_save_filepath, "wb") as f:
            pickle.dump(test_confdc_scores, f)
    else:
        assert args.mode == "test"
        with open(model_save_filepath, "rb") as f:
            clf = pickle.load(f)

        y_pred = clf.predict(X_test)
        print(f"\nARTS test accuracy: {accuracy_score(y_test, y_pred):.2f}")

        arts_confdc_scores = clf.predict_proba(X_test)
        print("Confidence scores shape: ", arts_confdc_scores.shape)
        print(arts_confdc_scores[:5])
        print(arts_confdc_scores[:5, 1])
            
        X_sotab, y_sotab, balanced_sotab_df = prepare_sotab_data_for_inference(args.sotab_benchmark, args.random_seed)

        y_sotab_pred = clf.predict(X_sotab)
        print(f"\nSOTAB benchmark accuracy: {accuracy_score(y_sotab, y_sotab_pred):.2f}")

        sotab_confdc_scores = clf.predict_proba(X_sotab)
        print("SOTAB confidence scores shape: ", sotab_confdc_scores.shape)
        print(sotab_confdc_scores[:5])
        print(sotab_confdc_scores[:5, 1])

        balanced_sotab_df["positive_label_confidence_score"] = sotab_confdc_scores[:, 1]
        
        output_filepath = os.path.join(
            exp_dir, "sotab_v2_test_mrf_data_with_confidence_scores.csv"
        )
        balanced_sotab_df.to_csv(output_filepath, index=False)
