import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from openforge.utils.custom_logging import get_logger


PRIOR_CONSTANT = 1e-9


def convert_var_name_to_var_id(var_name):
    var_id = tuple(int(elem) for elem in var_name.split("_")[1].split("-"))

    return var_id


def evaluate_inference_results(
    prior_data: pd.DataFrame, results: dict, log_predictions: bool = False
):
    if log_predictions:
        logger = get_logger()

    y_true, y_prior, y_pred = [], [], []

    for row in prior_data.itertuples():
        if log_predictions:
            logger.info("-" * 80)

        if hasattr(row, "relation_variable_label"):
            var_label = int(row.relation_variable_label)
        elif hasattr(row, "label"):
            var_label = int(row.label)
        else:
            raise AttributeError(
                "No ground truth attribute found in prior data."
            )

        var_name = row.relation_variable_name
        pred = int(results[var_name])

        y_true.append(var_label)
        y_pred.append(pred)

        if hasattr(row, "positive_label_prediction_probability"):
            if row.positive_label_prediction_probability >= 0.5:
                prior_pred = 1
                log_msg = (
                    f"Prior for variable {var_name}: ({prior_pred}, "
                    f"{row.positive_label_prediction_probability:.2f})"
                )
            else:
                prior_pred = 0
                log_msg = (
                    f"Prior for variable {var_name}: ({prior_pred}, "
                    f"{1 - row.positive_label_prediction_probability:.2f})"
                )
        elif hasattr(row, "confidence_score"):
            prior_pred = row.prediction
            confdc_score = row.confidence_score
            log_msg = (
                f"Prior for variable {var_name}: ({prior_pred}, "
                f"{confdc_score:.2f})"
            )
        else:
            raise AttributeError("No confidence score found in prior data.")

        y_prior.append(prior_pred)

        if log_predictions:
            logger.info(log_msg)
            logger.info(f"Posterior for variable {var_name}: {pred}")
            logger.info(f"True label for variable {var_name}: {var_label}")

            if prior_pred != var_label:
                logger.info("Prior prediction is incorrect.")
            if pred != var_label:
                logger.info("Posterior prediction is incorrect.")

    if log_predictions:
        logger.info("-" * 80)
        logger.info(f"Number of test instances: {len(prior_data)}")
        logger.info(
            f"Prior test accuracy: {accuracy_score(y_true, y_prior):.2f}"
        )
        logger.info(f"Prior F1 score: {f1_score(y_true, y_prior):.2f}")
        logger.info(f"Prior precision: {precision_score(y_true, y_prior):.2f}")
        logger.info(f"Prior recall: {recall_score(y_true, y_prior):.2f}")

    mrf_accuracy = accuracy_score(y_true, y_pred)
    mrf_f1_score = f1_score(y_true, y_pred)
    mrf_precision = precision_score(y_true, y_pred)
    mrf_recall = recall_score(y_true, y_pred)

    if log_predictions:
        logger.info(f"MRF test accuracy: {mrf_accuracy:.2f}")
        logger.info(f"MRF F1 score: {mrf_f1_score:.2f}")
        logger.info(f"MRF precision: {mrf_precision:.2f}")
        logger.info(f"MRF recall: {mrf_recall:.2f}")

    return mrf_f1_score, mrf_accuracy, mrf_precision, mrf_recall


def evaluate_multi_class_inference_results(
    prior_data: pd.DataFrame, results: dict, log_predictions: bool = False
):
    if log_predictions:
        logger = get_logger()

    y_true, y_prior, y_pred = [], [], []

    for row in prior_data.itertuples():
        if log_predictions:
            logger.info("-" * 80)

        var_name = row.relation_variable_name
        var_label = int(row.relation_variable_label)
        pred = int(results[var_name])

        y_true.append(row.relation_variable_label)
        y_pred.append(pred)

        prior_proba = np.array(
            [
                row.class_0_prediction_probability,
                row.class_1_prediction_probability,
                row.class_2_prediction_probability,
                # row.class_3_prediction_probability,
            ]
        )
        ml_pred = prior_proba.argmax()
        y_prior.append(ml_pred)

        if log_predictions:
            logger.info(f"Variable {var_name}:")
            logger.info(
                f"Prior prediction: ({ml_pred}, {prior_proba[ml_pred]:.2f})"
            )
            logger.info(f"Posterior prediction: {pred}")
            logger.info(f"True label: {var_label}")

            if ml_pred != var_label:
                logger.info("Prior prediction is incorrect.")
            if pred != row.relation_variable_label:
                logger.info("Posterior prediction is incorrect.")

    average_type = "macro"

    if log_predictions:
        logger.info("-" * 80)
        logger.info(f"Number of test instances: {len(prior_data)}")
        logger.info(
            f"Prior test accuracy: {accuracy_score(y_true, y_prior):.2f}"
        )

        prior_f1_score = f1_score(y_true, y_prior, average=average_type)
        prior_precision = precision_score(y_true, y_prior, average=average_type)
        prior_recall = recall_score(y_true, y_prior, average=average_type)

        logger.info(f"Prior F1 score: {prior_f1_score:.2f}")
        logger.info(f"Prior precision: {prior_precision:.2f}")
        logger.info(f"Prior recall: {prior_recall:.2f}")

    mrf_accuracy = accuracy_score(y_true, y_pred)
    mrf_f1_score = f1_score(y_true, y_pred, average=average_type)
    mrf_precision = precision_score(y_true, y_pred, average=average_type)
    mrf_recall = recall_score(y_true, y_pred, average=average_type)

    if log_predictions:
        logger.info(f"MRF test accuracy: {mrf_accuracy:.2f}")
        logger.info(f"MRF F1 score: {mrf_f1_score:.2f}")
        logger.info(f"MRF precision: {mrf_precision:.2f}")
        logger.info(f"MRF recall: {mrf_recall:.2f}")

    return mrf_f1_score, mrf_accuracy, mrf_precision, mrf_recall


def evaluate_fixed_prior_multi_class_inference_results(
    prior_data: pd.DataFrame, results: dict, log_predictions: bool = False
):
    if log_predictions:
        logger = get_logger()

    y_true, y_pred = [], []

    for row in prior_data.itertuples():
        if log_predictions:
            logger.info("-" * 80)

        var_name = row.relation_variable_name
        var_label = int(row.relation_variable_label)
        pred = int(results[var_name])

        y_true.append(row.relation_variable_label)
        y_pred.append(pred)

        if log_predictions:
            logger.info(f"Variable {var_name}:")
            logger.info(f"Posterior prediction: {pred}")
            logger.info(f"True label: {var_label}")

    average_type = "macro"

    if log_predictions:
        logger.info("-" * 80)
        logger.info(f"Number of test instances: {len(prior_data)}")

    mrf_accuracy = accuracy_score(y_true, y_pred)
    mrf_f1_score = f1_score(y_true, y_pred, average=average_type)
    mrf_precision = precision_score(y_true, y_pred, average=average_type)
    mrf_recall = recall_score(y_true, y_pred, average=average_type)

    if log_predictions:
        logger.info(f"MRF test accuracy: {mrf_accuracy:.2f}")
        logger.info(f"MRF F1 score: {mrf_f1_score:.2f}")
        logger.info(f"MRF precision: {mrf_precision:.2f}")
        logger.info(f"MRF recall: {mrf_recall:.2f}")

    return mrf_f1_score, mrf_accuracy, mrf_precision, mrf_recall
