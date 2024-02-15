import argparse
import os
import time

import pandas as pd

from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.models import MarkovNetwork
from pgmpy.sampling import GibbsSampling

from openforge.utils.custom_logging import get_custom_logger
from sklearn.metrics import accuracy_score, f1_score


TERNARY_TABLE = [1, 1, 1, 0, 1, 0, 0, 1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prior_data",
        type=str,
        default="/home/congtj/openforge/exps/arts-context_top-20-nodes/arts_mrf_data_test_with_ml_prior.csv", # "/home/congtj/openforge/exps/arts_mrf_synthesized_data_top-30-concepts/arts_test_mrf_data_with_confidence_scores.csv"
        help="Path to prior data."
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="/home/congtj/openforge/logs/arts-context_top-20-nodes/pgmpy",
        help="Directory to save logs."
    )

    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    logger = get_custom_logger(args.log_dir)
    logger.info(args)

    prior_df = pd.read_csv(args.prior_data)

    mrf = MarkovNetwork()
    unary_cliques = [] # list of unary clique ids (tuple of two integers)

    # add variables and unary factors
    for row in prior_df.itertuples():
        var_name = row.relation_variable_name
        mrf.add_node(var_name)
        
        unary_id = tuple(
            int(elem) for elem in var_name.split("_")[1].split("-")
        )
        unary_cliques.append(unary_id)

        confdc_score = row.positive_label_confidence_score
        prior = [1 - confdc_score, confdc_score]
        unary_factor = DiscreteFactor(
            [var_name],
            cardinality=[2],
            values=prior
        )

        mrf.add_factors(unary_factor)

    num_nodes = len(mrf.nodes())
    num_unary_factors = len(mrf.get_factors())
    logger.info(f"Number of MRF variables / unary factors: {num_nodes}")
    assert num_nodes == num_unary_factors

    # Add ternary factors
    ternary_cliques = set() # set of ternary clique ids (sorted tuples)

    for i, unary_id1 in enumerate(unary_cliques):
        for unary_id2 in unary_cliques[i+1:]:
            intersect = set(unary_id1).intersection(set(unary_id2))
            
            if len(intersect) == 1:
                ternary_id = list(set(unary_id1).union(set(unary_id2)))
                ternary_id.sort()

                if tuple(ternary_id) in ternary_cliques:
                    continue
            
                ternary_cliques.add(tuple(ternary_id))
                ternary_factor = DiscreteFactor(
                    [f"R_{ternary_id[0]}-{ternary_id[1]}",
                     f"R_{ternary_id[0]}-{ternary_id[2]}",
                     f"R_{ternary_id[1]}-{ternary_id[2]}"],
                    cardinality=[2, 2, 2],
                    values=TERNARY_TABLE
                )

                mrf.add_factors(ternary_factor)

    mrf_factors = mrf.get_factors()
    logger.info(f"Number of MRF factors: {len(mrf_factors)}")
    logger.info(f"MRF factors:\n{mrf_factors}")


    