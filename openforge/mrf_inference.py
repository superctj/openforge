import argparse
import os
import time

import pandas as pd
import pyAgrum as gum

from openforge.utils.custom_logging import get_custom_logger
from sklearn.metrics import f1_score


BINARY_TABLE = [0.5, 0.5, 0.5, 0.5]
TERNARY_TABLE = [1, 1, 1, 0, 1, 0, 0, 1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sotab_data",
        type=str,
        default="/home/congtj/openforge/exps/arts_top-100-concepts/sotab_v2_test_mrf_data_with_confidence_scores.csv",
        help="Path to the synthesized SOTAB benchmark."
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="/home/congtj/openforge/logs/sotab_synthesized_data/mrf_inference",
        help="Directory to store logs."
    )

    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    logger = get_custom_logger(args.log_dir)
    logger.info(args)

    sotab_df = pd.read_csv(args.sotab_data)

    mrf = gum.MarkovRandomField()
    variables = []

    binary_pairs = []
    pair_varidx_map = {}
    unary_table = []

    # Add variables
    for row in sotab_df.itertuples():
        var_name = row.relation_variable_name
        idx1, idx2 = var_name.split("_")[1].split("-")        
        
        var = gum.IntegerVariable(
            var_name,
            f"Relation variable between concepts {idx1} and {idx2}",
            [0, 1]
        )
        mrf.add(var)
        variables.append(var)
        
        key = (int(idx1), int(idx2))
        pair_varidx_map[key] = len(variables) - 1
        binary_pairs.append(key)

        confdc_score = row.positive_label_confidence_score
        unary_table.append([1 - confdc_score, confdc_score])

    # Add unary factors
    for i, var in enumerate(variables):
        p = gum.Potential().add(var).fillWith(unary_table[i])
        mrf.addFactor(p)
    
    # Add binary and ternary factors
    all_ternary_indices = set()

    for i, pair1 in enumerate(binary_pairs):
        for pair2 in binary_pairs[i+1:]:
            intersect = set(pair1).intersection(set(pair2))
            if len(intersect) == 1:
                var1 = variables[pair_varidx_map[pair1]]
                var2 = variables[pair_varidx_map[pair2]]
                
                p = gum.Potential().add(var1).add(var2).fillWith(BINARY_TABLE)
                mrf.addFactor(p)

                ternary_indices = list(set(pair1).union(set(pair2)))
                ternary_indices.sort()

                if tuple(ternary_indices) in all_ternary_indices:
                    continue
                
                ternary_factor = []
                for i in range(len(ternary_indices)):
                    for j in range(i+1, len(ternary_indices)):
                        pair = (ternary_indices[i], ternary_indices[j])

                        if pair in pair_varidx_map:
                            ternary_factor.append(pair)

                if len(ternary_factor) == 3:
                    var1 = variables[pair_varidx_map[ternary_factor[0]]]
                    var2 = variables[pair_varidx_map[ternary_factor[1]]]
                    var3 = variables[pair_varidx_map[ternary_factor[2]]]
                    
                    p = gum.Potential().add(var1).add(var2).add(var3).fillWith(TERNARY_TABLE)
                    mrf.addFactor(p)

                    all_ternary_indices.add(tuple(ternary_indices))

    logger.info(f"Number of MRF variables: {len(mrf.names())}")
    logger.info(f"MRF variables:\n{mrf.names()}")
    logger.info(f"Number of MRF factors: {len(mrf.factors())}")
    logger.info(f"MRF factors:\n{mrf.factors()}")

    ss = gum.ShaferShenoyMRFInference(mrf)

    start_time = time.time()
    ss.makeInference()
    end_time = time.time()
    logger.info(f"Inference time: {end_time - start_time} seconds")

    y_true, y_pred = [], []
    for row in sotab_df.itertuples():
        var_name = row.relation_variable_name
        posterior = ss.posterior(var_name)
        pred = posterior.argmax()[0][0][var_name]
        prob = posterior.argmax()[1]

        y_true.append(row.relation_variable_label)
        y_pred.append(pred)

        logger.info(f"Posterior for variable {var_name}: ({pred}, {prob:.2f})")

    logger.info(f"Number of test instances: {len(sotab_df)}")
    logger.info(f"F1 score: {f1_score(y_true, y_pred):.2f}")
