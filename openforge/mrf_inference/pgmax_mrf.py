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
from openforge.utils.mrf_common import evaluate_inference_results
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
        self.prior_data = pd.read_csv(prior_filepath)
        self.tune_lbp_hp = kwargs.get("tune_lbp_hp", False)

        if not self.tune_lbp_hp:
            self.num_iters = kwargs.get("num_iters", 200)
            self.damping = kwargs.get("damping", 0.5)
            self.temperature = kwargs.get("temperature", 0)

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

    def create_mrf(self, mrf_hp_config: dict) -> fgraph:
        ternary_table = [
            mrf_hp_config["ternary_alpha"],  # 0, 0, 0
            mrf_hp_config["ternary_beta"],  # 0, 0, 1
            mrf_hp_config["ternary_beta"],  # 0, 1, 0
            1e-9,  # 0, 1, 1
            mrf_hp_config["ternary_beta"],  # 1, 0, 0
            1e-9,  # 1, 0, 1
            1e-9,  # 1, 1, 0
            mrf_hp_config["ternary_gamma"],  # 1, 1, 1
        ]
        log_ternary_table = np.log(np.array(ternary_table))

        var_names = self.prior_data["relation_variable_name"].tolist()
        variables = vgroup.VarDict(num_states=2, variable_names=var_names)

        fg = fgraph.FactorGraph(variables)

        variables_for_unary_factors = []
        log_potentials = []

        # add unary factors
        for row in self.prior_data.itertuples():
            var_name = row.relation_variable_name
            var = variables.__getitem__(var_name)
            variables_for_unary_factors.append([var])

            confdc_score = row.positive_label_confidence_score
            prior = np.log(np.array([1 - confdc_score, confdc_score]))
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

    args = parser.parse_args()

    # Parse experiment configuration
    config = parse_config(args.config_path)

    # Set global random state
    fix_global_random_state(config.getint("hp_optimization", "random_seed"))

    # Create MRF hyperparameter space
    hp_space = HyperparameterSpace(
        config.get("hp_optimization", "hp_spec_filepath"),
        config.getint("hp_optimization", "random_seed"),
    ).create_hp_space()

    # Create logger
    output_dir = config.get("results", "output_dir")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = create_custom_logger(output_dir)
    printable_config = {section: dict(config[section]) for section in config}
    logger.info(f"Experiment configuration:\n{printable_config}\n")

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
    tuning_engine = TuningEngine(config, mrf_wrapper, hp_space)
    best_hp_config = tuning_engine.run()

    test_mrf_wrapper = MRFWrapper(
        config.get("mrf_lbp", "test_filepath"), tune_lbp_hp=True
    )

    test_mrf = test_mrf_wrapper.create_mrf(dict(best_hp_config))
    results = test_mrf_wrapper.run_inference(test_mrf, dict(best_hp_config))

    f1_score, accuracy = evaluate_inference_results(
        test_mrf_wrapper.prior_data, results
    )

    logger.info("Split: test")
    logger.info(f"  F1 score: {f1_score:.2f}")
    logger.info(f"  Accuracy: {accuracy:.2f}")
