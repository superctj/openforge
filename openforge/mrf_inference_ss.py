import argparse
import os
import time

from itertools import combinations

import pandas as pd
import pyAgrum as gum

from openforge.utils.custom_logging import create_custom_logger
from sklearn.metrics import accuracy_score, f1_score

# BINARY_TABLE = [0.5, 0.5, 0.5, 0.5]
TERNARY_TABLE = [0.7, 1, 1, 0, 1, 0, 0, 1]
DBPEDIA_PREFIX = "https:"


def convert_var_name_to_var_id(var_name):
    var_id = tuple(int(elem) for elem in var_name.split("_")[1].split("-"))

    return var_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prior_data",
        type=str,
        default="/home/congtj/openforge/exps/arts-context_top-40-nodes/sotab_v2_test_mrf_data_with_ml_prior.csv",  # noqa: E501
        help="Path to prior data.",
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="/home/congtj/openforge/logs/sotab_mrf_synthesized_data/pyagrum_ss",  # noqa: E501
        help="Directory to save logs.",
    )

    parser.add_argument(
        "--boost",
        type=str,
        default="prior_knowledge",
        help="Plug in hard evidence for boosting MRF inference. Options are 'none', 'confident_ml_prior', and 'prior_knowledge'.",  # noqa: E501
    )

    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    logger = create_custom_logger(args.log_dir)
    logger.info(args)

    prior_df = pd.read_csv(args.prior_data)

    mrf = gum.MarkovRandomField()
    unaryid_var_map = {}  # map from unary clique id to corresponding variable
    num_concepts = 1  # Count the first concept

    # Add variables and unary factors
    for row in prior_df.itertuples():
        var_name = row.relation_variable_name

        if var_name.startswith("R_1-"):
            num_concepts += 1

        var_id = convert_var_name_to_var_id(row.relation_variable_name)
        var_descr = (
            f"Relation variable over pair of concepts {var_id[0]} and "
            f"{var_id[1]}"
        )
        var = gum.IntegerVariable(var_name, var_descr, [0, 1])

        mrf.add(var)
        unaryid_var_map[var_id] = var

        confdc_score = row.positive_label_confidence_score
        prior = [1 - confdc_score, confdc_score]

        unary_factor = gum.Potential().add(var).fillWith(prior)
        mrf.addFactor(unary_factor)

    assert len(mrf.names()) == len(unaryid_var_map)
    logger.info(f"Number of MRF variables / unary factors: {len(mrf.names())}")
    logger.info(f"MRF variables:\n{mrf.names()}")
    logger.info(f"MRF unary factors:\n{mrf.factors()}")

    # Add ternary factors
    start = time.time()
    ternary_combos = combinations(range(1, num_concepts + 1), 3)
    end = time.time()
    logger.info(f"Time to generate ternary combos: {end-start:.2f} seconds")

    start = time.time()
    for combo in ternary_combos:
        var1 = unaryid_var_map[(combo[0], combo[1])]
        var2 = unaryid_var_map[(combo[0], combo[2])]
        var3 = unaryid_var_map[(combo[1], combo[2])]

        ternary_factor = (
            gum.Potential()
            .add(var1)
            .add(var2)
            .add(var3)
            .fillWith(TERNARY_TABLE)
        )
        mrf.addFactor(ternary_factor)

    end = time.time()
    logger.info(f"Time to add ternary factors: {end-start:.2f} seconds")

    logger.info(f"Number of MRF factors: {len(mrf.factors())}")
    logger.info(f"MRF factors:\n{mrf.factors()}")

    # Inference
    ss = gum.ShaferShenoyMRFInference(mrf)

    num_test_instances = 0
    test_instance_ids = set()

    if args.boost == "confident_ml_prior":
        for i, row in enumerate(prior_df.itertuples()):
            var_name = row.relation_variable_name
            pos_confdc_score = row.positive_label_confidence_score

            if pos_confdc_score >= 0.85:
                ss.addEvidence(var_name, 1)
            elif pos_confdc_score <= 0.15:
                ss.addEvidence(var_name, 0)
            else:
                num_test_instances += 1
                test_instance_ids.add(i)
    elif args.boost == "prior_knowledge":
        for i, row in enumerate(prior_df.itertuples()):
            concept1_from_dbpedia = row.label_1.startswith(DBPEDIA_PREFIX)
            concept2_from_dbpedia = row.label_2.startswith(DBPEDIA_PREFIX)

            if (concept1_from_dbpedia and concept2_from_dbpedia) or (
                not concept1_from_dbpedia and not concept2_from_dbpedia
            ):
                var_name = row.relation_variable_name
                ss.addEvidence(var_name, 0)
            else:
                num_test_instances += 1
                test_instance_ids.add(i)
    else:
        assert args.boost == "none"
        num_test_instances = len(prior_df)

    start_time = time.time()
    ss.makeInference()
    end_time = time.time()
    logger.info(f"Inference time: {end_time - start_time:.1f} seconds")

    y_true, y_pred, y_prior = [], []

    if args.boosting == "none":
        for row in prior_df.itertuples():
            logger.info("-" * 80)

            var_name = row.relation_variable_name
            posterior = ss.posterior(var_name)
            pred = posterior.argmax()[0][0][var_name]
            prob = posterior.argmax()[1]

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
            logger.info(
                f"Posterior for variable {var_name}: ({pred}, {prob:.2f})"
            )
            logger.info(
                f"True label for variable {var_name}: "
                f"{row.relation_variable_label}"
            )

            if ml_pred != row.relation_variable_label:
                logger.info("Prior prediction is incorrect.")
            if pred != row.relation_variable_label:
                logger.info("Posterior prediction is incorrect.")
    else:
        for i, row in enumerate(prior_df.itertuples()):
            logger.info("-" * 80)

            var_name = row.relation_variable_name

            if i in test_instance_ids:
                posterior = ss.posterior(var_name)
                pred = posterior.argmax()[0][0][var_name]
                prob = posterior.argmax()[1]

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
                logger.info(
                    f"Posterior for variable {var_name}: ({pred}, {prob:.2f})"
                )
                logger.info(
                    f"True label for variable {var_name}: "
                    f"{row.relation_variable_label}"
                )

                if ml_pred != row.relation_variable_label:
                    logger.info("Prior prediction is incorrect.")
                if pred != row.relation_variable_label:
                    logger.info("Posterior prediction is incorrect.")

    logger.info("-" * 80)
    logger.info(f"Number of test instances: {num_test_instances}")

    logger.info(f"Prior test accuracy: {accuracy_score(y_true, y_prior):.2f}")
    logger.info(f"Prior F1 score: {f1_score(y_true, y_prior):.2f}")

    logger.info(f"MRF test accuracy: {accuracy_score(y_true, y_pred):.2f}")
    logger.info(f"MRF F1 score: {f1_score(y_true, y_pred):.2f}")
