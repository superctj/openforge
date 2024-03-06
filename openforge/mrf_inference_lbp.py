import argparse
import os
import time

from itertools import combinations

import numpy as np
import pandas as pd

from pgmax import fgraph, fgroup, infer, vgroup

from sklearn.metrics import accuracy_score, f1_score

from openforge.utils.custom_logging import create_custom_logger

# TERNARY_TABLE = [
#     0.4459630461679717,
#     0.5260795123585992,
#     0.5260795123585992,
#     1e-9,
#     0.5260795123585992,
#     1e-9,
#     1e-9,
#     0.43414671350935646,
# ]  # BO-GP

# TERNARY_TABLE = [
#     0.6576313720060961,
#     0.8430622522793525,
#     0.8430622522793525,
#     1e-9,
#     0.8430622522793525,
#     1e-9,
#     1e-9,
#     0.5040575110778069,
# ]  # BO-RF
# log_ternary_table = np.log(np.array(TERNARY_TABLE))

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
        default="/home/congtj/openforge/exps/arts-context_top-40-nodes/arts_mrf_data_test_with_ml_prior.csv",  # noqa: E501
        help="Path to prior data.",
    )

    parser.add_argument(
        "--ternary_alpha",
        type=float,
        required=True,
        help="Ternary factor value for assignment [0, 0, 0].",
    )

    parser.add_argument(
        "--ternary_beta",
        type=float,
        required=True,
        help="Ternary factor value for assignments [0, 0, 1], [0, 1, 0] and [1, 0, 0].",  # noqa: E501
    )

    parser.add_argument(
        "--ternary_gamma",
        type=float,
        required=True,
        help="Ternary factor value for assignment [1, 1, 1].",
    )

    parser.add_argument(
        "--num_iters",
        type=int,
        default=575,
        help="Number of iterations for loopy belief propagation.",
    )

    parser.add_argument(
        "--damping",
        type=float,
        default=0.8996097484570096,
        help="Dampling for loopy belief propagation.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.06096136506950575,
        help="Temperature for loopy belief propagation.",
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="/home/congtj/openforge/logs/arts-context_top-40-nodes/pgmax_lbp",  # noqa: E501
        help="Directory to save logs.",
    )

    args = parser.parse_args()

    ternary_table = [
        args.ternary_alpha,  # 0, 0, 0
        args.ternary_beta,  # 0, 0, 1
        args.ternary_beta,  # 0, 1, 0
        1e-9,  # 0, 1, 1
        args.ternary_beta,  # 1, 0, 0
        1e-9,  # 1, 0, 1
        1e-9,  # 1, 1, 0
        args.ternary_gamma,  # 1, 1, 1
    ]
    log_ternary_table = np.log(np.array(ternary_table))

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    logger = create_custom_logger(args.log_dir)
    logger.info(args)
    logger.info(f"Ternary table: {ternary_table}")

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
    lbp_arrays, _ = lbp.run_with_diffs(
        lbp_arrays,
        num_iters=args.num_iters,
        damping=args.damping,
        temperature=args.temperature,
    )
    beliefs = lbp.get_beliefs(lbp_arrays)
    decoded_states = infer.decode_map_states(beliefs)
    results = list(decoded_states.values())[0]

    end_time = time.time()
    logger.info(f"Inference time: {end_time - start_time:.1f} seconds")

    y_true, y_pred = [], []
    for row in prior_df.itertuples():
        logger.info("-" * 80)

        var_name = row.relation_variable_name
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
