import pandas as pd

from sklearn.metrics import accuracy_score, f1_score

from openforge.utils.custom_logging import get_logger


def convert_var_name_to_var_id(var_name):
    var_id = tuple(int(elem) for elem in var_name.split("_")[1].split("-"))

    return var_id


def evaluate_inference_results(prior_data: pd.DataFrame, results: dict):
    logger = get_logger()
    y_true, y_prior, y_pred = [], [], []

    for row in prior_data.itertuples():
        logger.info("-" * 80)

        var_name = row.relation_variable_name
        var_label = int(row.relation_variable_label)
        pred = int(results[var_name])

        y_true.append(row.relation_variable_label)
        y_pred.append(pred)

        if row.positive_label_confidence_score >= 0.5:
            ml_pred = 1
            log_msg = (
                f"Prior for variable {var_name}: ({ml_pred}, "
                f"{row.positive_label_confidence_score:.2f})"
            )
        else:
            ml_pred = 0
            log_msg = (
                f"Prior for variable {var_name}: ({ml_pred}, "
                f"{1 - row.positive_label_confidence_score:.2f})"
            )

        y_prior.append(ml_pred)
        logger.info(log_msg)
        logger.info(f"Posterior for variable {var_name}: {pred}")
        logger.info(f"True label for variable {var_name}: {var_label}")

        if ml_pred != var_label:
            logger.info("Prior prediction is incorrect.")
        if pred != row.relation_variable_label:
            logger.info("Posterior prediction is incorrect.")

    logger.info("-" * 80)
    logger.info(f"Number of test instances: {len(prior_data)}")

    logger.info(f"Prior test accuracy: {accuracy_score(y_true, y_prior):.2f}")
    logger.info(f"Prior F1 score: {f1_score(y_true, y_prior):.2f}")

    mrf_accuracy = accuracy_score(y_true, y_pred)
    mrf_f1_score = f1_score(y_true, y_pred)
    logger.info(f"MRF test accuracy: {mrf_accuracy:.2f}")
    logger.info(f"MRF F1 score: {mrf_f1_score:.2f}")

    return mrf_f1_score, mrf_accuracy
