import argparse
import os
import time

from itertools import combinations, product

import numpy as np
import pandas as pd

from pgmax import fgraph, fgroup, infer, vgroup

from openforge.hp_optimization.hp_space import HyperparameterSpace
from openforge.hp_optimization.tuning import TuningEngine
from openforge.utils.custom_logging import create_custom_logger, get_logger
from openforge.utils.mrf_common import evaluate_inference_results
from openforge.utils.util import fix_global_random_state, parse_config


UNARY_FACTOR_CONFIG = np.array([[0], [1], [2], [3]])
TERNARY_FACTOR_CONFIG = np.array(
    [config for config in product(range(4), repeat=3)]
)


class MRFWrapper:
    def __init__(self, prior_filepath: str, **kwargs):
        self.prior_data = pd.read_csv(prior_filepath)
        self.tune_lbp_hp = kwargs.get("tune_lbp_hp", False)

        if not self.tune_lbp_hp:
            self.num_iters = kwargs.get("num_iters", 200)
            self.damping = kwargs.get("damping", 0.5)
            self.temperature = kwargs.get("temperature", 0)

        self.num_concepts = self._count_num_concepts()
        self.hard_evidence = {}
        self.logger = get_logger()

    def _count_num_concepts(self):
        num_concepts = 1  # Count the first concept

        for row in self.prior_data.itertuples():
            if row.relation_variable_name.startswith("R_1-"):
                num_concepts += 1
            else:
                break

        return num_concepts

    def create_mrf(self, mrf_hp_config: dict) -> fgraph:
        ternary_table = [
            mrf_hp_config["theta_1"],  # (0, 0, 0)
            mrf_hp_config["theta_2"],  # (0, 0, 1)
            mrf_hp_config["theta_3"],  # (0, 0, 2)
            mrf_hp_config["theta_4"],  # (0, 0, 3)
            mrf_hp_config["theta_5"],  # (0, 1, 0)
            1e-9,  # (0, 1, 1)
            1e-9,  # (0, 1, 2)
            1e-9,  # (0, 1, 3)
            mrf_hp_config["theta_6"],  # (0, 2, 0)
            1e-9,  # (0, 2, 1)
            1e-9,  # (0, 2, 2)
            1e-9,  # (0, 2, 3)
            mrf_hp_config["theta_7"],  # (0, 3, 0)
            1e-9,  # (0, 3, 1)
            1e-9,  # (0, 3, 2)
            mrf_hp_config["theta_8"],  # (0, 3, 3)
            mrf_hp_config["theta_9"],  # (1, 0, 0)
            1e-9,  # (1, 0, 1)
            1e-9,  # (1, 0, 2)
            1e-9,  # (1, 0, 3)
            1e-9,  # (1, 1, 0)
            mrf_hp_config["theta_10"],  # (1, 1, 1)
            1e-9,  # (1, 1, 2)
            1e-9,  # (1, 1, 3)
            1e-9,  # (1, 2, 0)
            1e-9,  # (1, 2, 1)
            mrf_hp_config["theta_11"],  # (1, 2, 2)
            1e-9,  # (1, 2, 3)
            1e-9,  # (1, 3, 0)
            1e-9,  # (1, 3, 1)
            1e-9,  # (1, 3, 2)
            mrf_hp_config["theta_12"],  # (1, 3, 3)
            mrf_hp_config["theta_13"],  # (2, 0, 0)
            1e-9,  # (2, 0, 1)
            mrf_hp_config["theta_14"],  # (2, 0, 2)
            1e-9,  # (2, 0, 3)
            1e-9,  # (2, 1, 0)
            1e-9,  # (2, 1, 1)
            mrf_hp_config["theta_15"],  # (2, 1, 2)
            1e-9,  # (2, 1, 3)
            1e-9,  # (2, 2, 0)
            1e-9,  # (2, 2, 1)
            mrf_hp_config["theta_16"],  # (2, 2, 2)
            1e-9,  # (2, 2, 3)
            1e-9,  # (2, 3, 0)
            1e-9,  # (2, 3, 1)
            mrf_hp_config["theta_17"],  # (2, 3, 2)
            mrf_hp_config["theta_18"],  # (2, 3, 3)
            mrf_hp_config["theta_19"],  # (3, 0, 0)
            1e-9,  # (3, 0, 1)
            1e-9,  # (3, 0, 2)
            1e-9,  # (3, 0, 3)
            1e-9,  # (3, 1, 0)
            1e-9,  # (3, 1, 1)
            1e-9,  # (3, 1, 2)
            mrf_hp_config["theta_20"],  # (3, 1, 3)
            mrf_hp_config["theta_21"],  # (3, 2, 0)
            mrf_hp_config["theta_22"],  # (3, 2, 1)
            mrf_hp_config["theta_23"],  # (3, 2, 2)
            mrf_hp_config["theta_24"],  # (3, 2, 3)
            1e-9,  # (3, 3, 0)
            1e-9,  # (3, 3, 1)
            1e-9,  # (3, 3, 2)
            mrf_hp_config["theta_25"],  # (3, 3, 3)
        ]
        log_ternary_table = np.log(np.array(ternary_table))

        var_names = self.prior_data["relation_variable_name"].tolist()
        variables = vgroup.VarDict(num_states=4, variable_names=var_names)

        fg = fgraph.FactorGraph(variables)

        variables_for_unary_factors = []
        log_potentials = []

        # add unary factors
        for row in self.prior_data.itertuples():
            var_name = row.relation_variable_name
            var = variables.__getitem__(var_name)
            variables_for_unary_factors.append([var])

            pred_proba = [
                row.class_0_prediction_probability,
                row.class_1_prediction_probability,
                row.class_2_prediction_probability,
                row.class_3_prediction_probability,
            ]

            prior = np.log(np.array(pred_proba))
            log_potentials.append(prior)

        unary_factor_group = fgroup.EnumFactorGroup(
            variables_for_factors=variables_for_unary_factors,
            factor_configs=UNARY_FACTOR_CONFIG,
            log_potentials=np.array(log_potentials),
        )
        fg.add_factors(unary_factor_group)

        # add ternary factors
        start = time.time()
        ternary_combos = combinations(range(1, self.num_concepts + 1), 3)
        end = time.time()
        self.logger.info(
            f"Time to generate ternary combos: {end-start:.2f} seconds"
        )

        variables_for_ternary_factors = []
        start = time.time()

        for combo in ternary_combos:
            var1 = variables.__getitem__(f"R_{combo[0]}-{combo[1]}")
            var2 = variables.__getitem__(f"R_{combo[0]}-{combo[2]}")
            var3 = variables.__getitem__(f"R_{combo[1]}-{combo[2]}")
            variables_for_ternary_factors.append([var1, var2, var3])

        ternary_factor_group = fgroup.EnumFactorGroup(
            variables_for_factors=variables_for_ternary_factors,
            factor_configs=TERNARY_FACTOR_CONFIG,
            log_potentials=log_ternary_table,
        )
        fg.add_factors(ternary_factor_group)

        end = time.time()
        self.logger.info(
            f"Time to add ternary factors: {end-start:.2f} seconds"
        )

        return fg

    # LBP inference
    def run_inference(self, fg, mrf_hp_config: dict) -> dict:
        lbp = infer.build_inferer(fg.bp_state, backend="bp")

        if self.hard_evidence:
            lbp_arrays = lbp.init(evidence_updates=self.hard_evidence)
        else:
            lbp_arrays = lbp.init()

        start_time = time.time()

        if self.tune_lbp_hp:
            lbp_arrays, _ = lbp.run_with_diffs(
                lbp_arrays,
                num_iters=mrf_hp_config["num_iters"],
                damping=mrf_hp_config["damping"],
                temperature=mrf_hp_config["temperature"],
            )
        else:
            lbp_arrays, _ = lbp.run_with_diffs(
                lbp_arrays,
                num_iters=self.num_iters,
                damping=self.damping,
                temperature=self.temperature,
            )

        beliefs = lbp.get_beliefs(lbp_arrays)
        decoded_states = infer.decode_map_states(beliefs)
        results = list(decoded_states.values())[0]

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
        if config.getboolean("mrf_lbp", "tune_lbp_hp"):
            mrf_wrapper = MRFWrapper(
                config.get("mrf_lbp", "validation_filepath"),
                tune_lbp_hp=True,
            )
        else:
            mrf_wrapper = MRFWrapper(
                config.get("mrf_lbp", "validation_filepath"),
                tune_lbp_hp=False,
                num_iters=config.getint("mrf_lbp", "num_iters"),
                damping=config.getfloat("mrf_lbp", "damping"),
                temperature=config.getfloat("mrf_lbp", "temperature"),
            )

        # Hyperparameter tuning
        tuning_engine = TuningEngine(
            config, mrf_wrapper, hp_space, multi_class=True
        )
        best_hp_config = tuning_engine.run()
    else:
        assert args.mode == "inference", (
            f"Invalid mode: {args.mode}. Mode must either be hp_tuning or "
            "inference."
        )

        best_hp_config = {
            "damping": 0.7858213721344507,
            "num_iters": 921,
            "temperature": 0.9910111614267442,
            "alpha": 0.5768611571269355,
            "beta": 0.5855811670070769,
            "gamma": 0.016516056183874597,
            "delta": 0.588351726474382,
        }

    test_mrf_wrapper = MRFWrapper(
        config.get("mrf_lbp", "test_filepath"), tune_lbp_hp=True
    )

    test_mrf = test_mrf_wrapper.create_mrf(dict(best_hp_config))
    results = test_mrf_wrapper.run_inference(test_mrf, dict(best_hp_config))

    evaluate_inference_results(
        test_mrf_wrapper.prior_data, results, log_predictions=True
    )
