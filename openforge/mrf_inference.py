import argparse
import os
import time

import pandas as pd
import pyAgrum as gum

from openforge.utils.custom_logging import get_custom_logger
from sklearn.metrics import accuracy_score, f1_score


# BINARY_TABLE = [0.5, 0.5, 0.5, 0.5]
TERNARY_TABLE = [1, 1, 1, 0, 1, 0, 0, 1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mrf_data",
        type=str,
        default="/home/congtj/openforge/exps/arts_top-100-concepts/sotab_v2_test_mrf_data_with_confidence_scores.csv", # "/home/congtj/openforge/exps/arts_mrf_synthesized_data_top-30-concepts/arts_test_mrf_data_with_confidence_scores.csv"
        help="Path to synthesized MRF data."
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="/home/congtj/openforge/logs/sotab_mrf_synthesized_data/mrf_inference", # "/home/congtj/openforge/logs/arts_mrf_synthesized_data_top-30-concepts/mrf_inference"
        help="Directory to store logs."
    )

    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    logger = get_custom_logger(args.log_dir)
    logger.info(args)

    mrf_df = pd.read_csv(args.mrf_data)

    mrf = gum.MarkovRandomField()
    variables = []

    unary_cliques = [] # list of unary clique ids (tuple of two integers)
    unaryid_varidx_map = {} # map from unary clique id to corresponding variable index in 'variables'

    # Add variables and unary factors
    for row in mrf_df.itertuples():
        var_name = row.relation_variable_name
        unary_id = tuple(
            int(elem) for elem in var_name.split("_")[1].split("-")
        )
        
        var = gum.IntegerVariable(
            var_name,
            f"Relation variable over pair of concepts {unary_id[0]} and {unary_id[1]}",
            [0, 1]
        )
        mrf.add(var)
        variables.append(var)

        unaryid_varidx_map[unary_id] = len(variables) - 1
        unary_cliques.append(unary_id)

        confdc_score = row.positive_label_confidence_score
        prior = [1 - confdc_score, confdc_score]

        unary_factor = gum.Potential().add(var).fillWith(prior)
        mrf.addFactor(unary_factor)
    
    assert len(mrf.names()) == len(variables)
    assert len(mrf.names()) == len(unary_cliques)
    assert len(mrf.names()) == len(unaryid_varidx_map)
    logger.info(f"Number of MRF variables / unary factors: {len(mrf.names())}")
    logger.info(f"MRF variables:\n{mrf.names()}")
    logger.info(f"MRF unary factors:\n{mrf.factors()}")
    
    # Add binary and ternary factors
    ternary_cliques = set() # set of ternary clique ids (sorted tuples)

    for i, unary_id1 in enumerate(unary_cliques):
        for unary_id2 in unary_cliques[i+1:]:
            intersect = set(unary_id1).intersection(set(unary_id2))
            
            if len(intersect) == 1: # variables involving a common concept
                # var1 = variables[unaryid_varidx_map[unary_id1]]
                # var2 = variables[unaryid_varidx_map[unary_id2]]
                
                # binary_factor = gum.Potential().add(var1).add(var2).fillWith(BINARY_TABLE)
                # mrf.addFactor(binary_factor)

                ternary_ids = list(set(unary_id1).union(set(unary_id2)))
                ternary_ids.sort()

                if tuple(ternary_ids) in ternary_cliques:
                    continue
                
                pairs_in_ternary_clique = []
                for j in range(len(ternary_ids)):
                    for k in range(j+1, len(ternary_ids)):
                        pair = (ternary_ids[j], ternary_ids[k])

                        if pair in unaryid_varidx_map:
                            pairs_in_ternary_clique.append(pair)

                if len(pairs_in_ternary_clique) == 3:
                    var1 = variables[unaryid_varidx_map[pairs_in_ternary_clique[0]]]
                    var2 = variables[unaryid_varidx_map[pairs_in_ternary_clique[1]]]
                    var3 = variables[unaryid_varidx_map[pairs_in_ternary_clique[2]]]
                    
                    ternary_factor = gum.Potential().add(var1).add(var2).add(var3).fillWith(TERNARY_TABLE)
                    mrf.addFactor(ternary_factor)

                    ternary_cliques.add(tuple(ternary_ids))

    logger.info(f"Number of MRF factors: {len(mrf.factors())}")
    logger.info(f"MRF factors:\n{mrf.factors()}")

    ss = gum.ShaferShenoyMRFInference(mrf)

    for row in mrf_df.itertuples():
        var_name = row.relation_variable_name
        pos_confdc_score = row.positive_label_confidence_score

        if pos_confdc_score >= 0.85:
            ss.addEvidence(var_name, 1)
        elif pos_confdc_score <= 0.15:
            ss.addEvidence(var_name, 0)

    start_time = time.time()
    ss.makeInference()
    end_time = time.time()
    logger.info(f"Inference time: {end_time - start_time:.1f} seconds")

    y_true, y_pred = [], []
    for row in mrf_df.itertuples():
        logger.info("-"*80)

        var_name = row.relation_variable_name
        posterior = ss.posterior(var_name)
        pred = posterior.argmax()[0][0][var_name]
        prob = posterior.argmax()[1]

        y_true.append(row.relation_variable_label)
        y_pred.append(pred)

        if row.positive_label_confidence_score >= 0.5:
            ml_pred = 1
            logger.info(f"Prior for variable {var_name}: ({ml_pred}, {row.positive_label_confidence_score:.2f})")
        else:
            ml_pred = 0
            logger.info(f"Prior for variable {var_name}: ({ml_pred}, {1 - row.positive_label_confidence_score:.2f})")
        
        logger.info(f"Posterior for variable {var_name}: ({pred}, {prob:.2f})")
        logger.info(f"True label for variable {var_name}: {row.relation_variable_label}")

        if ml_pred != row.relation_variable_label:
            logger.info(f"Prior prediction is incorrect.")
        if pred != row.relation_variable_label:
            logger.info(f"Posterior prediction is incorrect.")

    logger.info("-"*80)
    logger.info(f"Number of test instances: {len(mrf_df)}")
    logger.info(f"Test accuracy: {accuracy_score(y_true, y_pred):.2f}")
    logger.info(f"F1 score: {f1_score(y_true, y_pred):.2f}")
