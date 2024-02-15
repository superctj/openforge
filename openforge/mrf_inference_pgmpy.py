import argparse
import os
import time

from itertools import combinations

import pandas as pd

from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import Mplp
from pgmpy.models import MarkovNetwork
from pgmpy.sampling import GibbsSampling

from openforge.utils.custom_logging import get_custom_logger
from sklearn.metrics import accuracy_score, f1_score


TERNARY_TABLE = [1, 1, 1, 0, 1, 0, 0, 1]


def convert_var_name_to_var_id(var_name):
    var_id = tuple(
        int(elem) for elem in var_name.split("_")[1].split("-")
    )

    return var_id


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

        unary_id = convert_var_name_to_var_id(var_name)
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
    logger.info(f"MRF nodes:\n{mrf.nodes()}")
    assert num_nodes == num_unary_factors
    assert mrf.check_model()

    # # Add edges and ternary factors
    # ternary_cliques = set()

    # for edge in combinations(mrf.nodes(), 2):
    #     mrf.add_edge(*edge)

    #     var1_id = convert_var_name_to_var_id(edge[0])
    #     var2_id = convert_var_name_to_var_id(edge[1])

    #     intersect = set(var1_id).intersection(set(var2_id))
    #     if len(intersect) == 1:
    #         ternary_id = list(set(var1_id).union(set(var2_id)))
    #         ternary_id.sort()

    #         if tuple(ternary_id) in ternary_cliques:
    #             continue

    #         ternary_cliques.add(tuple(ternary_id))
    #         var1 = f"R_{ternary_id[0]}-{ternary_id[1]}"
    #         var2 = f"R_{ternary_id[0]}-{ternary_id[2]}"
    #         var3 = f"R_{ternary_id[1]}-{ternary_id[2]}"

    #         ternary_factor = DiscreteFactor(
    #             [var1, var2, var3],
    #             cardinality=[2, 2, 2],
    #             values=TERNARY_TABLE
    #         )
    #         ternary_factor.normalize()

    #         mrf.add_factors(ternary_factor)
    
    # Add ternary factors
    ternary_cliques = set() # set of ternary clique ids (sorted tuples)

    for i, unary_id1 in enumerate(unary_cliques):
        # logger.info(f"{i}-th unary clique: {unary_id1}")
        for unary_id2 in unary_cliques[i+1:]:
            intersect = set(unary_id1).intersection(set(unary_id2))
            
            if len(intersect) == 1:
                ternary_id = list(set(unary_id1).union(set(unary_id2)))
                ternary_id.sort()

                if tuple(ternary_id) in ternary_cliques:
                    continue
            
                logger.info(f"Ternary id: {ternary_id}")
                ternary_cliques.add(tuple(ternary_id))

                var1 = f"R_{ternary_id[0]}-{ternary_id[1]}"
                var2 = f"R_{ternary_id[0]}-{ternary_id[2]}"
                var3 = f"R_{ternary_id[1]}-{ternary_id[2]}"

                mrf.add_edges_from([(var1, var2), (var1, var3), (var2, var3)])
                ternary_factor = DiscreteFactor(
                    [var1, var2, var3],
                    cardinality=[2, 2, 2],
                    values=TERNARY_TABLE
                )
                ternary_factor.normalize()

                mrf.add_factors(ternary_factor)
                assert mrf.check_model()
    
    logger.info(f"Number of MRF edges: {len(mrf.edges())}")
    logger.info(f"MRF edges:\n{mrf.edges()}")

    mrf_factors = mrf.get_factors()
    logger.info(f"Number of MRF factors: {len(mrf_factors)}")
    logger.info(f"MRF factors:\n{mrf_factors}")

    # Inference
    mplp = Mplp(mrf)
    logger.info(f"mplp objective: {mplp.objective}")

    start_time = time.time()
    results = mplp.map_query(tighten_triplet=False)
    end_time = time.time()
    logger.info(f"Inference time: {end_time - start_time:.1f} seconds")
    logger.info(f"Results:\n{results}")

    y_true, y_pred = [], []
    for row in prior_df.itertuples():
        logger.info("-"*80)

        var_name = row.relation_variable_name
        pred = results[var_name]

        y_true.append(row.relation_variable_label)
        y_pred.append(pred)

        if row.positive_label_confidence_score >= 0.5:
            ml_pred = 1
            logger.info(f"Prior for variable {var_name}: ({ml_pred}, {row.positive_label_confidence_score:.2f})")
        else:
            ml_pred = 0
            logger.info(f"Prior for variable {var_name}: ({ml_pred}, {1 - row.positive_label_confidence_score:.2f})")
        
        logger.info(f"Posterior for variable {var_name}: {pred}")
        logger.info(f"True label for variable {var_name}: {row.relation_variable_label}")

        if ml_pred != row.relation_variable_label:
            logger.info(f"Prior prediction is incorrect.")
        if pred != row.relation_variable_label:
            logger.info(f"Posterior prediction is incorrect.")

    logger.info("-"*80)
    logger.info(f"Number of test instances: {len(prior_df)}")
    logger.info(f"Test accuracy: {accuracy_score(y_true, y_pred):.2f}")
    logger.info(f"F1 score: {f1_score(y_true, y_pred):.2f}")
