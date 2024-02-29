import argparse
import os
import time

from itertools import combinations

import numpy as np
import pandas as pd

from pgmax import fgraph, fgroup, infer, vgroup

from sklearn.metrics import accuracy_score, f1_score

from openforge.utils.custom_logging import create_custom_logger

TERNARY_TABLE = [
    0.6937840237588209,
    0.9999718175956511,
    0.9999718175956511,
    1e-9,
    0.9999718175956511,
    1e-9,
    1e-9,
    0.999976048166078,
]

log_ternary_table = np.log(np.array(TERNARY_TABLE))

unary_factor_config = np.array([[0], [1]])
ternary_factor_config = np.array(
    [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ]
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prior_data",
        type=str,
        default="/home/congtj/openforge/exps/arts-context_top-20-nodes/arts_mrf_data_test_with_ml_prior.csv",  # noqa: E501
        help="Path to prior data.",
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="/home/congtj/openforge/logs/arts-context_top-20-nodes/pgmax_lbp",  # noqa: E501
        help="Directory to save logs.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    logger = create_custom_logger(args.log_dir)
    logger.info(args)
    logger.info(f"Ternary table: {TERNARY_TABLE}")

    prior_df = pd.read_csv(args.prior_data)

    var_names = prior_df["relation_variable_name"].tolist()
    variables = vgroup.VarDict(num_states=2, variable_names=var_names)

    fg = fgraph.FactorGraph(variables)

    num_concepts = 1  # Count the first concept
    variables_for_unary_factors = []
    log_potentials = []

    # add unary factors
    for row in prior_df.itertuples():
        var_name = row.relation_variable_name
        if var_name.startswith("R_1-"):
            num_concepts += 1

        var = variables.__getitem__(var_name)
        variables_for_unary_factors.append([var])

        confdc_score = row.positive_label_confidence_score
        prior = np.log(np.array([1 - confdc_score, confdc_score]))
        log_potentials.append(prior)

    unary_factor_group = fgroup.EnumFactorGroup(
        variables_for_factors=variables_for_unary_factors,
        factor_configs=unary_factor_config,
        log_potentials=np.array(log_potentials),
    )
    fg.add_factors(unary_factor_group)

    # add ternary factors
    start = time.time()
    ternary_combos = combinations(range(1, num_concepts + 1), 3)
    end = time.time()
    logger.info(f"Time to generate ternary combos: {end-start:.2f} seconds")

    variables_for_ternary_factors = []
    start = time.time()

    for combo in ternary_combos:
        var1 = variables.__getitem__(f"R_{combo[0]}-{combo[1]}")
        var2 = variables.__getitem__(f"R_{combo[0]}-{combo[2]}")
        var3 = variables.__getitem__(f"R_{combo[1]}-{combo[2]}")
        variables_for_ternary_factors.append([var1, var2, var3])

    ternary_factor_group = fgroup.EnumFactorGroup(
        variables_for_factors=variables_for_ternary_factors,
        factor_configs=ternary_factor_config,
        log_potentials=log_ternary_table,
    )
    fg.add_factors(ternary_factor_group)

    end = time.time()
    logger.info(f"Time to add ternary factors: {end-start:.2f} seconds")

    # Inference
    lbp = infer.build_inferer(fg.bp_state, backend="bp")
    lbp_arrays = lbp.init()

    start_time = time.time()
    lbp_arrays, _ = lbp.run_with_diffs(lbp_arrays, num_iters=200, temperature=0)
    end_time = time.time()
    logger.info(f"Inference time: {end_time - start_time:.1f} seconds")

    beliefs = lbp.get_beliefs(lbp_arrays)
    map_states = infer.decode_map_states(beliefs)
    results = list(map_states.values())[0]

    y_true, y_pred = [], []
    for row in prior_df.itertuples():
        logger.info("-" * 80)

        var_name = row.relation_variable_name
        pred = float(results[var_name])

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
