import argparse
import os
import random
import time

from itertools import combinations

import numpy as np

from pgmax import fgraph, fgroup, infer, vgroup

from openforge.utils.custom_logging import create_custom_logger
from openforge.utils.util import fix_global_random_state

TERNARY_TABLE = [
    0.4459630461679717,
    0.5260795123585992,
    0.5260795123585992,
    1e-9,
    0.5260795123585992,
    1e-9,
    1e-9,
    0.43414671350935646,
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
        "--num_concepts",
        type=int,
        required=True,
        help="Number of concepts in the synthesized MRF.",
    )

    parser.add_argument(
        "--num_iters",
        type=int,
        default=200,
        help="Number of iterations for loopy belief propagation.",
    )

    parser.add_argument(
        "--damping",
        type=float,
        default=0.5,
        help="Dampling for loopy belief propagation.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0,
        help="Temperature for loopy belief propagation.",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=12345,
        help="Random seed.",
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="/home/congtj/openforge/logs/synthesized_mrf/pgmax_lbp",  # noqa: E501
        help="Directory to save logs.",
    )

    args = parser.parse_args()

    fix_global_random_state(args.random_seed)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    logger = create_custom_logger(args.log_dir)
    logger.info(args)
    logger.info(f"Ternary table: {TERNARY_TABLE}")

    # Synthesize MRF variables
    start = time.time()
    binary_combos = combinations(range(1, args.num_concepts + 1), 2)
    variable_names = [f"R_{combo[0]}-{combo[1]}" for combo in binary_combos]

    end = time.time()
    logger.info(f"Time to synthesize variable names: {end-start:.2f} seconds")
    logger.info(f"Number of variables: {len(variable_names)}")

    # Add MRF variables
    start = time.time()
    variables = vgroup.VarDict(num_states=2, variable_names=variable_names)
    fg = fgraph.FactorGraph(variables)

    end = time.time()
    logger.info(
        f"Time to create and add MRF variables: {end-start:.2f} seconds"
    )

    # Add unary factors
    start = time.time()
    variables_for_unary_factors = []
    log_potentials = []

    for var_name in variable_names:
        var = variables.__getitem__(var_name)
        variables_for_unary_factors.append([var])

        confdc_score = random.random()
        prior = np.log(np.array([1 - confdc_score, confdc_score]))
        log_potentials.append(prior)

    unary_factor_group = fgroup.EnumFactorGroup(
        variables_for_factors=variables_for_unary_factors,
        factor_configs=unary_factor_config,
        log_potentials=np.array(log_potentials),
    )
    fg.add_factors(unary_factor_group)

    end = time.time()
    logger.info(f"Time to add unary factors: {end-start:.2f} seconds")

    # Add ternary factors
    start = time.time()
    ternary_combos = combinations(range(1, args.num_concepts + 1), 3)
    end = time.time()
    logger.info(f"Time to generate ternary combos: {end-start:.2f} seconds")

    start = time.time()
    variables_for_ternary_factors = []

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

    all_factors = fg.factors
    num_factors = sum(
        [len(all_factors[factor_type]) for factor_type in all_factors]
    )
    _num_factors = len(variables_for_ternary_factors) + len(
        variables_for_unary_factors
    )
    assert num_factors == _num_factors
    logger.info(f"Number of factors: {num_factors}")

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
