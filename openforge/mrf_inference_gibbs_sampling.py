import argparse
import os
import time

from itertools import combinations

import pandas as pd

from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.models import MarkovNetwork
from pgmpy.sampling import GibbsSampling

from openforge.utils.custom_logging import create_custom_logger


TERNARY_TABLE = [1, 1, 1, 0, 1, 0, 0, 1]


def convert_var_name_to_var_id(var_name):
    var_id = tuple(int(elem) for elem in var_name.split("_")[1].split("-"))

    return var_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prior_data",
        type=str,
        default="/home/congtj/openforge/exps/arts-context_top-30-nodes/\
arts_mrf_data_test_with_ml_prior.csv",
        help="Path to prior data.",
    )

    parser.add_argument(
        "--num_concepts",
        type=int,
        default=16,
        help="Number of concepts in the vocabulary to be cleaned.",
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="/home/congtj/openforge/logs/arts-context_top-30-nodes/\
pgmpy_gibbs_sampling",
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
    gibbs = GibbsSampling(mrf)
    start_time = time.time()
    samples = gibbs.generate_sample(size=1)
    end_time = time.time()
    logger.info(f"Gibbs sampling time: {end_time - start_time:.1f} seconds")
