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
from openforge.utils.mrf_common import (
    PRIOR_CONSTANT,
    evaluate_multi_class_inference_results,
)
from openforge.utils.util import fix_global_random_state, parse_config


UNARY_FACTOR_CONFIG = np.array([[0], [1], [2]])
TERNARY_FACTOR_CONFIG = np.array(
    [config for config in product(range(3), repeat=3)]
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
            mrf_hp_config["theta_1"],  # 0, 0, 0
            mrf_hp_config["theta_2"],  # 0, 0, 1
            mrf_hp_config["theta_3"],  # 0, 0, 2
            mrf_hp_config["theta_4"],  # 0, 1, 0
            1e-9,  # 0, 1, 1 (invalid)
            1e-9,  # 0, 1, 2 (invalid)
            mrf_hp_config["theta_5"],  # 0, 2, 0
            1e-9,  # 0, 2, 1 (invalid)
            1e-9,  # 0, 2, 2 (invalid)
            mrf_hp_config["theta_6"],  # 1, 0, 0
            1e-9,  # 1, 0, 1 (invalid)
            1e-9,  # 1, 0, 2 (invalid)
            1e-9,  # 1, 1, 0 (invalid)
            mrf_hp_config["theta_7"],  # 1, 1, 1
            1e-9,  # 1, 1, 2 (invalid)
            1e-9,  # 1, 2, 0 (invalid)
            1e-9,  # 1, 2, 1 (invalid)
            mrf_hp_config["theta_8"],  # 1, 2, 2
            mrf_hp_config["theta_9"],  # 2, 0, 0
            1e-9,  # 2, 0, 1 (invalid)
            mrf_hp_config["theta_10"],  # 2, 0, 2
            1e-9,  # 2, 1, 0 (invalid)
            1e-9,  # 2, 1, 1 (invalid)
            mrf_hp_config["theta_11"],  # 2, 1, 2
            1e-9,  # 2, 2, 0
            1e-9,  # 2, 2, 1
            mrf_hp_config["theta_12"],  # 2, 2, 2
        ]
        log_ternary_table = np.log(np.array(ternary_table))

        var_names = self.prior_data["relation_variable_name"].tolist()
        variables = vgroup.VarDict(num_states=3, variable_names=var_names)

        fg = fgraph.FactorGraph(variables)

        variables_for_unary_factors = []
        log_potentials = []

        # add unary factors
        for row in self.prior_data.itertuples():
            var_name = row.relation_variable_name
            try:
                var = variables.__getitem__(var_name)
            except ValueError or KeyError:
                continue

            variables_for_unary_factors.append([var])

            if row.class_0_prediction_probability == 1:
                pred_proba = [
                    row.class_0_prediction_probability - PRIOR_CONSTANT,
                    row.class_1_prediction_probability + PRIOR_CONSTANT / 2,
                    row.class_2_prediction_probability + PRIOR_CONSTANT / 2,
                ]
            elif row.class_1_prediction_probability == 1:
                pred_proba = [
                    row.class_0_prediction_probability + PRIOR_CONSTANT / 2,
                    row.class_1_prediction_probability - PRIOR_CONSTANT,
                    row.class_2_prediction_probability + PRIOR_CONSTANT / 2,
                ]
            elif row.class_2_prediction_probability == 1:
                pred_proba = [
                    row.class_0_prediction_probability + PRIOR_CONSTANT / 2,
                    row.class_1_prediction_probability + PRIOR_CONSTANT / 2,
                    row.class_2_prediction_probability - PRIOR_CONSTANT,
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
            try:
                var1 = variables.__getitem__(f"R_{combo[0]}-{combo[1]}")
                var2 = variables.__getitem__(f"R_{combo[1]}-{combo[2]}")
                var3 = variables.__getitem__(f"R_{combo[0]}-{combo[2]}")
            except ValueError or KeyError:
                continue

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
            "damping": 0.7996067179474019,
            "num_iters": 299,
            "temperature": 0.8646208996742156,
            "theta_1": 0.8281892761927666,
            "theta_10": 0.7160829149621573,
            "theta_11": 0.06467029145598348,
            "theta_12": 0.9604915621479585,
            "theta_2": 0.8476117082576713,
            "theta_3": 0.68919886965042,
            "theta_4": 0.9981987186655246,
            "theta_5": 0.7484245704418557,
            "theta_6": 0.0905392183519112,
            "theta_7": 0.9925108194099971,
            "theta_8": 0.6098967885182909,
            "theta_9": 0.6847057422479415
        }

    test_mrf_wrapper = MRFWrapper(
        config.get("mrf_lbp", "test_filepath"), tune_lbp_hp=True
    )

    test_mrf = test_mrf_wrapper.create_mrf(dict(best_hp_config))
    results = test_mrf_wrapper.run_inference(test_mrf, dict(best_hp_config))

    evaluate_multi_class_inference_results(
        test_mrf_wrapper.prior_data,
        results,
        log_predictions=True,
    )
