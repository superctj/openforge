import argparse
import multiprocessing
import os

multiprocessing.set_start_method("spawn", force=True)
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
import time

from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from pgmax import fgraph, fgroup, infer, vgroup

from openforge.hp_optimization.hp_space import HyperparameterSpace
from openforge.hp_optimization.tuning import (
    SparseDatasetTuningEngine as TuningEngine,
)
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
    def __init__(
        self,
        prior_dir: str,
        ground_truth_filepath: str,
        batch_size: int,
        **kwargs,
    ):
        self.prior_dir = prior_dir
        if ground_truth_filepath.endswith(".json"):
            self.ground_truth = pd.read_json(ground_truth_filepath)
        elif ground_truth_filepath.endswith(".csv"):
            self.ground_truth = pd.read_csv(ground_truth_filepath)
        else:
            raise ValueError(
                f"Invalid ground truth file format: {ground_truth_filepath}"
            )
        self.ground_truth.rename(
            columns={
                "prior_prediction": "prediction",
                "prior_confidence_score": "confidence_score",
            },
            inplace=True,
        )
        self.batch_size = batch_size
        self.tune_lbp_hp = kwargs.get("tune_lbp_hp", False)

        if not self.tune_lbp_hp:
            self.num_iters = kwargs.get("num_iters", 200)
            self.damping = kwargs.get("damping", 0.5)
            self.temperature = kwargs.get("temperature", 0)

        self.logger = get_logger()

    def create_mrf(self, prior_df: pd.DataFrame, mrf_hp_config: dict) -> fgraph:
        # start = time.time()
        var_identifiers = []
        ternary_combos = []
        l_id = None

        for i, row in prior_df.iterrows():
            var_identifiers.append((row["l_id"], row["r_id"]))

            if i == 0:
                l_id = row["l_id"]
                assert l_id.startswith("l_"), f"Invalid l_id: {l_id}"

            if row["l_id"].startswith("r_") and row["r_id"].startswith("r_"):
                ternary_combos.append((l_id, row["l_id"], row["r_id"]))

        variables = vgroup.VarDict(num_states=2, variable_names=var_identifiers)
        fg = fgraph.FactorGraph(variables)

        # end = time.time()
        # self.logger.info(
        #     f"Time to create and add MRF variables: {end-start:.2f} seconds"
        # )

        # add unary factors
        # start = time.time()
        variables_for_unary_factors = []
        log_unary_potentials = []

        for _, row in prior_df.iterrows():
            var = variables.__getitem__((row["l_id"], row["r_id"]))
            variables_for_unary_factors.append([var])
            pred_proba = row["confidence_score"]

            # Get around the warning of dividing by zero encountered in log
            if pred_proba == 1:
                pred_proba -= PRIOR_CONSTANT

            if row["prediction"] == 1:
                prior = np.log(np.array([1 - pred_proba, pred_proba]))
            else:
                prior = np.log(np.array([pred_proba, 1 - pred_proba]))

            log_unary_potentials.append(prior)

        unary_factor_group = fgroup.EnumFactorGroup(
            variables_for_factors=variables_for_unary_factors,
            factor_configs=UNARY_FACTOR_CONFIG,
            log_potentials=np.array(log_unary_potentials),
        )
        fg.add_factors(unary_factor_group)

        # end = time.time()
        # self.logger.info(f"Time to add unary factors: {end-start:.2f} seconds") # noqa: E501

        # start = time.time()
        # add ternary factors
        ternary_table = [
            mrf_hp_config["alpha"],  # 0, 0, 0
            mrf_hp_config["beta"],  # 0, 0, 1
            mrf_hp_config["gamma"],  # 0, 1, 0
            1e-9,  # 0, 1, 1
            mrf_hp_config["delta"],  # 1, 0, 0
            1e-9,  # 1, 0, 1
            1e-9,  # 1, 1, 0
            1e-9,  # mrf_hp_config["epsilon"],  # 1, 1, 1
        ]
        log_ternary_table = np.log(np.array(ternary_table))
        variables_for_ternary_factors = []

        for combo in ternary_combos:
            var1 = variables.__getitem__((combo[0], combo[1]))
            var2 = variables.__getitem__((combo[0], combo[2]))
            var3 = variables.__getitem__((combo[1], combo[2]))
            variables_for_ternary_factors.append((var1, var2, var3))

        ternary_factor_group = fgroup.EnumFactorGroup(
            variables_for_factors=variables_for_ternary_factors,
            factor_configs=TERNARY_FACTOR_CONFIG,
            log_potentials=log_ternary_table,
        )
        fg.add_factors(ternary_factor_group)

        # end = time.time()
        # self.logger.info(
        #     f"Time to add ternary factors: {end-start:.2f} seconds"
        # )

        return fg

    def run_single_inference(
        self, prior_filename: str, mrf_hp_config: dict
    ) -> dict:
        prior_filepath = os.path.join(self.prior_dir, prior_filename)
        prior_df = pd.read_json(prior_filepath)
        # first_row = prior_df.iloc[0]

        # Only run inference if the test instance is predicted as positive
        # if first_row["prediction"] == 0:
        #     return None, -1

        fg = self.create_mrf(prior_df, mrf_hp_config)
        lbp = infer.build_inferer(fg.bp_state, backend="bp")
        lbp_arrays = lbp.init()

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

        target_row = prior_df.iloc[0]
        target_id = (target_row["l_id"], target_row["r_id"])

        return target_id, results[target_id]

    # LBP inference
    def run_inference(self, mrf_hp_config: dict) -> dict:
        start_time = time.time()
        all_results = {}

        filenames = [
            f for f in os.listdir(self.prior_dir) if f.endswith(".json")
        ]

        with ProcessPoolExecutor(max_workers=self.batch_size) as executor:
            futures = {
                executor.submit(self.run_single_inference, f, mrf_hp_config): f
                for f in filenames
            }

            for future in as_completed(futures):
                target_id, result = future.result()
                # logger.info(f"target_id, result: {target_id}, {result}")

                if target_id is not None:
                    all_results[target_id] = int(result)

        # logger.info(f"All results: {all_results}")
        end_time = time.time()
        self.logger.info(f"Inference time: {end_time - start_time:.1f} seconds")

        return all_results


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

    mrf_input_dir = config.get("io", "mrf_input_dir")
    ground_truth_dir = config.get("io", "ground_truth_dir")
    batch_size = config.getint("io", "batch_size")
    random_seed = config.getint("hp_optimization", "random_seed")

    valid_mrf_input_dir = os.path.join(mrf_input_dir, "validation")
    test_mrf_input_dir = os.path.join(mrf_input_dir, "test")

    valid_ground_truth_filepath = os.path.join(
        ground_truth_dir, "validation.csv"
    )
    if not os.path.exists(valid_ground_truth_filepath):
        valid_ground_truth_filepath = os.path.join(
            ground_truth_dir, "validation.json"
        )

    test_ground_truth_filepath = os.path.join(ground_truth_dir, "test.csv")
    if not os.path.exists(test_ground_truth_filepath):
        test_ground_truth_filepath = os.path.join(ground_truth_dir, "test.json")

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
            valid_mrf_input_dir,
            valid_ground_truth_filepath,
            batch_size,
            tune_lbp_hp=config.getboolean("hp_optimization", "tune_lbp_hp"),
        )

        # Hyperparameter tuning
        tuning_engine = TuningEngine(config, mrf_wrapper, hp_space)
        best_hp_config = tuning_engine.run()
    else:
        assert args.mode == "inference", (
            f"Invalid mode: {args.mode}. Mode must either be hp_tuning or "
            "inference."
        )

        best_hp_config = {
            "alpha": 0.3163755552654104,
            "beta": 0.5332322301219835,
            # "damping": 0.02170897897556423,
            "delta": 0.7214512689066233,
            "epsilon": 0.31628815459566506,
            "gamma": 0.09126672146562628,
            #     'num_iters': 444,
            #     'temperature': 0.3338919742857718,
        }
        logger.info(f"Best hyperparameters:\n{best_hp_config}")

    test_mrf_wrapper = MRFWrapper(
        test_mrf_input_dir,
        test_ground_truth_filepath,
        batch_size,
        tune_lbp_hp=config.getboolean("hp_optimization", "tune_lbp_hp"),
    )
    results = test_mrf_wrapper.run_inference(dict(best_hp_config))

    evaluate_inference_results(
        test_mrf_wrapper.ground_truth, results, log_predictions=True
    )
