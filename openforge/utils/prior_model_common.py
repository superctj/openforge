import logging
import os

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def load_openforge_sotab_split(
    split_filepath: str, logger: logging.Logger
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load a split of an OpenForge-SOTAB dataset.

    Args:
        split_filepath: The file path to the split.
        logger: The logging instance.

    Returns:
        X: Features.
        y: Labels.
        df: The split as a DataFrame.
    """

    df = pd.read_csv(split_filepath, delimiter=",", header=0)
    X, y = [], []

    for row in df.itertuples():
        X.append(
            [
                row.name_qgram_similarity,
                row.name_jaccard_similarity,
                row.name_edit_distance,
                row.name_fasttext_similarity,
                row.name_word_count_ratio,
                row.name_char_count_ratio,
                row.value_jaccard_similarity,
                row.value_fasttext_similarity,
            ]
        )
        y.append(row.relation_variable_label)

    X = np.array(X)
    y = np.array(y)

    logger.info(f"X shape: {X.shape}")
    logger.info(f"y shape: {y.shape}")
    logger.info(f"Number of positive instances: {np.sum(y == 1)}\n")

    return X, y, df


def load_openforge_sotab_benchmark(
    data_dir: str, logger: logging.Logger
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Load training, validation, and test splits of the OpenForge-SOTAB-X
    benchmark.

    Args:
        data_dir: The directory containing the benchmark.
        logger: The logging instance.

    Returns:
        The training, validation, and test data.
    """

    train_filepath = os.path.join(data_dir, "training.csv")
    valid_filepath = os.path.join(data_dir, "validation.csv")
    test_filepath = os.path.join(data_dir, "test.csv")

    logger.info("Loading training split...")
    X_train, y_train, train_df = load_openforge_sotab_split(
        train_filepath, logger
    )

    logger.info("Loading validation split...")
    X_valid, y_valid, valid_df = load_openforge_sotab_split(
        valid_filepath, logger
    )

    logger.info("Loading test split...")
    X_test, y_test, test_df = load_openforge_sotab_split(test_filepath, logger)

    return (
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test,
        train_df,
        valid_df,
        test_df,
    )


def evaluate_prior_model_predictions(
    y: np.ndarray, y_pred: np.ndarray
) -> tuple[float, float]:
    """Evaluate predictions of a prior model.

    Args:
        y: The true labels.
        y_pred: The predicted labels.

    Returns:
        f1: The F1 score.
        accuracy: The accuracy.
    """

    f1 = f1_score(y, y_pred)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = precision_score(y, y_pred)

    return f1, accuracy, precision, recall


def log_exp_metrics(
    split: str, y: np.ndarray, y_pred: np.ndarray, logger: logging.Logger
):
    """Log experiment metrics of a dataset split.

    Args:
        y: The true labels.
        y_pred: The predicted labels.
        logger: The logging instance.
    """

    f1, accuracy, precision, recall = evaluate_prior_model_predictions(
        y, y_pred
    )

    logger.info(f"Split: {split}")
    logger.info(f"  Accuracy: {accuracy:.2f}")
    logger.info(f"  F1 score: {f1:.2f}")
    logger.info(f"  Precision: {precision:.2f}")
    logger.info(f"  Recall: {recall:.2f}\n")


def log_exp_records(
    y: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    split: str,
    logger: logging.Logger,
    num_records: int = 5,
):
    """Log experiment records.

    Args:
        y: The true labels.
        y_pred: The predicted labels.
        y_proba: The predicted probabilities.
        split: The dataset split.
        logger: The logging instance.
        num_records: The number of records to log.
    """

    logger.info(f"Split: {split}")
    logger.info(f"First {num_records} labels: {y[:num_records]}")
    logger.info(f"First {num_records} predictions: {y_pred[:num_records]}")
    logger.info(
        f"First {num_records} prediction probabilities:\n"
        f"{y_proba[:num_records]}"
    )
