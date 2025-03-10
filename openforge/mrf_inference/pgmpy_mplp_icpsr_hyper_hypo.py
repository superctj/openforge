import argparse
import os
import time

from itertools import combinations, product

import numpy as np
import pandas as pd

from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import Mplp
from pgmpy.models import MarkovNetwork

from openforge.hp_optimization.hp_space import HyperparameterSpace
from openforge.hp_optimization.tuning import TuningEngine
from openforge.utils.custom_logging import create_custom_logger, get_logger
from openforge.utils.mrf_common import (
    PRIOR_CONSTANT,
    evaluate_multi_class_inference_results,
)
from openforge.utils.util import fix_global_random_state, parse_config


class MRFWrapper:
    def __init__(self, prior_filepath: str):
        if prior_filepath.endswith(".csv"):
            self.prior_data = pd.read_csv(prior_filepath)
        elif prior_filepath.endswith(".json"):
            self.prior_data = pd.read_json(prior_filepath)
        self.num_concepts = self._count_num_concepts()
        self.logger = get_logger()

    def _count_num_concepts(self):
        num_concepts = 1  # Count the first concept

        for row in self.prior_data.itertuples():
            if row.relation_variable_name.startswith("R_1-"):
                num_concepts += 1
            else:
                break

        return num_concepts

    def create_mrf(self, mrf_hp_config: dict) -> MarkovNetwork:
        ternary_table = [
            1,  # (0, 0, 0)
            1,  # (0, 0, 1)
            1,  # (0, 0, 2)
            1,  # (0, 1, 0)
            1,  # (0, 1, 1)
            1,  # (0, 1, 2)
            1,  # (0, 2, 0)
            1,  # (0, 2, 1)
            1,  # (0, 2, 2)
            1,  # (1, 0, 0)
            1,  # (1, 0, 1)
            1,  # (1, 0, 2)
            1e-5,  # (1, 1, 0)
            1,  # (1, 1, 1)
            1e-5,  # (1, 1, 2)
            1,  # (1, 2, 0)
            1,  # (1, 2, 1)
            1,  # (1, 2, 2)
            1,  # (2, 0, 0)
            1,  # (2, 0, 1)
            1,  # (2, 0, 2)
            1,  # (2, 1, 0)
            1,  # (2, 1, 1)
            1,  # (2, 1, 2)
            1e-5,  # (2, 2, 0)
            1e-5,  # (2, 2, 1)
            1,  # (2, 2, 2)
        ]
        ternary_table = np.array(ternary_table) * mrf_hp_config["alpha"]

        mrf = MarkovNetwork()

        # add variables and unary factors
        start = time.time()
        for row in self.prior_data.itertuples():
            var_name = row.relation_variable_name
            mrf.add_node(var_name)

            prior = [
                row.class_0_prediction_probability,
                row.class_1_prediction_probability,
                row.class_2_prediction_probability,
            ]
            unary_factor = DiscreteFactor(
                [var_name], cardinality=[3], values=prior
            )
            mrf.add_factors(unary_factor)

        end = time.time()
        self.logger.info(
            f"Time to add MRF variables and unary factors: {end-start:.2f} "
            "seconds"
        )
        num_nodes = len(mrf.nodes())
        num_unary_factors = len(mrf.get_factors())

        assert num_nodes == num_unary_factors
        assert mrf.check_model()

        start = time.time()
        ternary_combos = combinations(range(1, self.num_concepts + 1), 3)
        end = time.time()
        self.logger.info(
            f"Time to generate ternary combos: {end-start:.2f} seconds"
        )

        start = time.time()
        for combo in ternary_combos:
            var1 = f"R_{combo[0]}-{combo[1]}"
            var2 = f"R_{combo[1]}-{combo[2]}"
            var3 = f"R_{combo[0]}-{combo[2]}"
            mrf.add_edges_from([(var1, var2), (var1, var3), (var2, var3)])

            ternary_factor = DiscreteFactor(
                [var1, var2, var3], cardinality=[3, 3, 3], values=ternary_table
            )
            ternary_factor.normalize()
            mrf.add_factors(ternary_factor)

        end = time.time()
        self.logger.info(
            f"Time to add ternary factors: {end-start:.2f} seconds"
        )
        self.logger.info(f"Number of MRF edges: {len(mrf.edges())}")
        self.logger.info(f"Number of MRF factors: {len(mrf.get_factors())}")

        return mrf

    # MPLP inference
    # add `mrf_hp_config`` for consistent API
    def run_inference(self, mrf, mrf_hp_config: dict) -> dict:
        mplp = Mplp(mrf)

        start_time = time.time()
        results = mplp.map_query(tighten_triplet=False)
        end_time = time.time()
        self.logger.info(f"Inference time: {end_time - start_time:.1f} seconds")

        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the experiment configuration file.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="hp_tuning",
        help="Mode: hp_tuning or inference.",
    )

    args = parser.parse_args()

    # Parse experiment configuration
    config = parse_config(args.config_path)

    # Set global random state
    fix_global_random_state(config.getint("hp_optimization", "random_seed"))

    # Create logger
    output_dir = config.get("results", "output_dir")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = create_custom_logger(output_dir)
    logger.info(f"Running program: {__file__}\n")
    logger.info(f"{args}\n")

    printable_config = {section: dict(config[section]) for section in config}
    logger.info(f"Experiment configuration:\n{printable_config}\n")

    if args.mode == "hp_tuning":
        # Create MRF hyperparameter space
        hp_space = HyperparameterSpace(
            config.get("hp_optimization", "hp_spec_filepath"),
            config.getint("hp_optimization", "random_seed"),
        ).create_hp_space()

        # Create MRF wrapper
        mrf_wrapper = MRFWrapper(config.get("mrf", "validation_filepath"))

        # Hyperparameter tuning
        tuning_engine = TuningEngine(config, mrf_wrapper, hp_space)
        best_hp_config = tuning_engine.run()
    else:
        assert args.mode == "inference", (
            f"Invalid mode: {args.mode}. Mode must either be hp_tuning or "
            "inference."
        )

        best_hp_config = {
            "alpha": 0.11716047224357475,
        }

    test_mrf_wrapper = MRFWrapper(config.get("mrf", "test_filepath"))

    test_mrf = test_mrf_wrapper.create_mrf(dict(best_hp_config))
    results = test_mrf_wrapper.run_inference(test_mrf, dict(best_hp_config))

    evaluate_multi_class_inference_results(
        test_mrf_wrapper.prior_data, results, log_predictions=True
    )
