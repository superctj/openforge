import argparse
import os
import time

from itertools import combinations

import numpy as np
import pandas as pd

from pgmax import fgraph, fgroup, infer, vgroup

from openforge.hp_optimization.hp_space import HyperparameterSpace
from openforge.hp_optimization.tuning import TuningEngine
from openforge.utils.custom_logging import create_custom_logger, get_logger
from openforge.utils.mrf_common import (
    PRIOR_CONSTANT,
    evaluate_inference_results,
)
from openforge.utils.util import fix_global_random_state, parse_config


UNARY_FACTOR_CONFIG = np.array([[0], [1]])
TERNARY_FACTOR_CONFIG = np.array(
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


class MRFWrapper:
    def __init__(self, prior_filepath: str, **kwargs):
        if prior_filepath.endswith(".csv"):
            self.prior_data = pd.read_csv(prior_filepath)
        elif prior_filepath.endswith(".json"):
            self.prior_data = pd.read_json(prior_filepath)
        else:
            raise ValueError("Invalid file format. Must be .csv or .json.")

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
            if row.random_variable_name.startswith("R_1-"):
                num_concepts += 1
            else:
                break

        return num_concepts

    def create_mrf(self, mrf_hp_config: dict) -> fgraph:
        ternary_table = [
            mrf_hp_config["alpha"],  # 0, 0, 0
            mrf_hp_config["beta"],  # 0, 0, 1
            mrf_hp_config["gamma"],  # 0, 1, 0
            1e-9,  # 0, 1, 1
            mrf_hp_config["delta"],  # 1, 0, 0
            1e-9,  # 1, 0, 1
            1e-9,  # 1, 1, 0
            1e-9,  # 1, 1, 1
        ]
        log_ternary_table = np.log(np.array(ternary_table))

        start = time.time()
        var_names = self.prior_data["random_variable_name"].tolist()
        variables = vgroup.VarDict(num_states=2, variable_names=var_names)

        fg = fgraph.FactorGraph(variables)
        end = time.time()
        logger.info(
            f"Time to create and add MRF variables: {end-start:.2f} seconds"
        )

        variables_for_unary_factors = []
        log_potentials = []

        # add unary factors
        start = time.time()
        for row in self.prior_data.itertuples():
            var_name = row.random_variable_name
            var = variables.__getitem__(var_name)
            variables_for_unary_factors.append([var])

            if hasattr(row, "positive_label_confidence_score"):
                pred_proba = row.positive_label_prediction_probability

                # Get around the warning of dividing by zero encountered in log
                if pred_proba == 1:
                    pred_proba = 1 - PRIOR_CONSTANT
                elif pred_proba == 0:
                    pred_proba = PRIOR_CONSTANT

                prior = np.log(np.array([1 - pred_proba, pred_proba]))
            elif hasattr(row, "confidence_score"):
                pred_proba = row.confidence_score

                # Get around the warning of dividing by zero encountered in log
                if pred_proba == 1:
                    pred_proba = 1 - PRIOR_CONSTANT
                elif pred_proba == 0:
                    pred_proba = PRIOR_CONSTANT

                if row.prediction == 1:
                    prior = np.log(np.array([1 - pred_proba, pred_proba]))
                else:
                    prior = np.log(np.array([pred_proba, 1 - pred_proba]))
            else:
                raise AttributeError("No confidence score found in prior data.")

            log_potentials.append(prior)

        unary_factor_group = fgroup.EnumFactorGroup(
            variables_for_factors=variables_for_unary_factors,
            factor_configs=UNARY_FACTOR_CONFIG,
            log_potentials=np.array(log_potentials),
        )
        fg.add_factors(unary_factor_group)

        end = time.time()
        logger.info(f"Time to add unary factors: {end-start:.2f} seconds")

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
            var2 = variables.__getitem__(f"R_{combo[1]}-{combo[2]}")
            var3 = variables.__getitem__(f"R_{combo[0]}-{combo[2]}")
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

    # Create logger
    output_dir = config.get("io", "output_dir")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = create_custom_logger(output_dir)
    logger.info(f"Running program: {__file__}\n")
    logger.info(f"{args}\n")

    printable_config = {section: dict(config[section]) for section in config}
    logger.info(f"Experiment configuration:\n{printable_config}\n")

    if args.mode == "hp_tuning":
        # Set global random state
        fix_global_random_state(config.getint("hp_optimization", "random_seed"))

        # Create MRF hyperparameter space
        hp_space = HyperparameterSpace(
            config.get("hp_optimization", "hp_spec_filepath"),
            config.getint("hp_optimization", "random_seed"),
        ).create_hp_space()

        # Create MRF wrapper
        mrf_wrapper = MRFWrapper(
            config.get("io", "validation_filepath"),
            tune_lbp_hp=config.getboolean("hp_optimization", "tune_lbp_hp"),
        )

        # Hyperparameter tuning
        tuning_engine = TuningEngine(config, mrf_wrapper, hp_space)
        best_hp_config = tuning_engine.run()

        test_mrf_wrapper = MRFWrapper(
            config.get("io", "test_filepath"),
            tune_lbp_hp=config.getboolean("hp_optimization", "tune_lbp_hp"),
        )
    else:
        assert args.mode == "inference", (
            f"Invalid mode: {args.mode}. Mode must either be hp_tuning or "
            "inference."
        )

        best_hp_config = {
            "damping": 0.16642048711955618,  # noqa: E501 0.7858213721344507,
            "num_iters": 165,  # 921,
            "temperature": 0.9705734290236616,  # noqa: E501 0.9910111614267442,
            "alpha": 0.7822110970515735,  # noqa: E501 0.5768611571269355,
            "beta": 0.07215988266477134,  # noqa: E501 0.5855811670070769,
            "gamma": 0.8463037695085346,  # noqa: E501 0.016516056183874597,
            "delta": 0.8006681327874328,  # noqa: E501 0.588351726474382,
        }
        # best_hp_config = {
        #     "alpha": 0.8071540338456273,
        #     "beta": 0.04738649782097863,
        #     "damping": 0.05602489087262887,
        #     "delta": 0.8468977818900162,
        #     "gamma": 0.7935151701483814,
        #     "num_iters": 217,
        #     "temperature": 0.9724354141004133,
        # }
        logger.info(f"Best hyperparameters:\n{best_hp_config}")

        if "damping" not in best_hp_config:
            test_mrf_wrapper = MRFWrapper(
                config.get("io", "test_filepath"), tune_lbp_hp=False
            )
        else:
            test_mrf_wrapper = MRFWrapper(
                config.get("io", "test_filepath"), tune_lbp_hp=True
            )

    test_mrf = test_mrf_wrapper.create_mrf(dict(best_hp_config))
    results = test_mrf_wrapper.run_inference(test_mrf, dict(best_hp_config))

    evaluate_inference_results(
        test_mrf_wrapper.prior_data, results, log_predictions=True
    )
