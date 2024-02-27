import argparse
import os
import time

from itertools import combinations

import pandas as pd

from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import Mplp
from pgmpy.models import MarkovNetwork

from openforge.utils.custom_logging import create_custom_logger
from sklearn.metrics import accuracy_score, f1_score

# ARTS - 40
TERNARY_TABLE = [
    0.6937840237588209,
    0.9999718175956511,
    0.9999718175956511,
    0,
    0.9999718175956511,
    0,
    0,
    0.999976048166078,
]
# TERNARY_TABLE = [
#     0.7,
#     1,
#     1,
#     0,
#     1,
#     0,
#     0,
#     1,
# ]

# ARTS-20
# TERNARY_TABLE = [
#     0.6848856139369088,
#     0.9213272477855319,
#     0.9213272477855319,
#     0,
#     0.9213272477855319,
#     0,
#     0,
#     0.9986992365040954,
# ]


def convert_var_name_to_var_id(var_name):
    var_id = tuple(int(elem) for elem in var_name.split("_")[1].split("-"))

    return var_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prior_data",
        type=str,
        default="/home/congtj/openforge/exps/arts-context_top-40-nodes/arts_mrf_data_test_with_ml_prior.csv",  # noqa: E501
        help="Path to prior data.",
    )

    parser.add_argument(
        "--num_concepts",
        type=int,
        default=19,
        help="Number of concepts in the vocabulary to be cleaned.",
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="/home/congtj/openforge/logs/arts-context_top-40-nodes/pgmpy_mplp",  # noqa: E501
        help="Directory to save logs.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    logger = create_custom_logger(args.log_dir)
    logger.info(args)

    prior_df = pd.read_csv(args.prior_data)
    mrf = MarkovNetwork()

    # add variables and unary factors
    for row in prior_df.itertuples():
        var_name = row.relation_variable_name
        mrf.add_node(var_name)

        confdc_score = row.positive_label_confidence_score
        prior = [1 - confdc_score, confdc_score]
        unary_factor = DiscreteFactor([var_name], cardinality=[2], values=prior)

        mrf.add_factors(unary_factor)

    num_nodes = len(mrf.nodes())
    num_unary_factors = len(mrf.get_factors())
    logger.info(f"Number of MRF variables / unary factors: {num_nodes}")
    logger.info(f"MRF nodes:\n{mrf.nodes()}")
    assert num_nodes == num_unary_factors
    assert mrf.check_model()

    start = time.time()
    ternary_combos = combinations(range(1, args.num_concepts + 1), 3)
    end = time.time()
    logger.info(
        f"Time taken to generate ternary combos: {end-start:.2f} seconds"
    )

    start = time.time()
    for combo in ternary_combos:
        var1 = f"R_{combo[0]}-{combo[1]}"
        var2 = f"R_{combo[0]}-{combo[2]}"
        var3 = f"R_{combo[1]}-{combo[2]}"

        mrf.add_edges_from([(var1, var2), (var1, var3), (var2, var3)])
        ternary_factor = DiscreteFactor(
            [var1, var2, var3], cardinality=[2, 2, 2], values=TERNARY_TABLE
        )
        ternary_factor.normalize()

        mrf.add_factors(ternary_factor)

    end = time.time()
    logger.info(f"Time taken to add ternary factors: {end-start:.2f} seconds")

    logger.info(f"Number of MRF edges: {len(mrf.edges())}")
    logger.info(f"MRF edges:\n{mrf.edges()}")

    mrf_factors = mrf.get_factors()
    logger.info(f"Number of MRF factors: {len(mrf_factors)}")
    logger.info(f"MRF factors:\n{mrf_factors}")

    # Inference
    mplp = Mplp(mrf)

    start_time = time.time()
    results = mplp.map_query(tighten_triplet=False)
    end_time = time.time()
    logger.info(f"Inference time: {end_time - start_time:.1f} seconds")

    y_true, y_pred = [], []
    for row in prior_df.itertuples():
        logger.info("-" * 80)

        var_name = row.relation_variable_name
        pred = results[var_name]

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

        logger.info(log_msg)
        logger.info(f"Posterior for variable {var_name}: {pred}")
        logger.info(
            f"True label for variable {var_name}: {row.relation_variable_label}"
        )

        if ml_pred != row.relation_variable_label:
            logger.info("Prior prediction is incorrect.")
        if pred != row.relation_variable_label:
            logger.info("Posterior prediction is incorrect.")

    logger.info("-" * 80)
    logger.info(f"Number of test instances: {len(prior_df)}")
    logger.info(f"Test accuracy: {accuracy_score(y_true, y_pred):.2f}")
    logger.info(f"F1 score: {f1_score(y_true, y_pred):.2f}")
