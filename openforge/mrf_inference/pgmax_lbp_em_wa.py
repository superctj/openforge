import argparse
import os
import time

# from collections import Counter
# from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd

# from sklearn.cluster import KMeans
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


def profile_connected_components(df: pd.DataFrame):
    G = nx.Graph()

    for _, row in df.iterrows():
        lid = row["ltable_id"]
        rid = row["rtable_id"]
        label = row["label"]

        G.add_node(lid)
        G.add_node(rid)
        G.add_edge(lid, rid, label=label)

    connected_components = list(nx.connected_components(G))

    # Sort the components by size in descending order
    cc_sorted = sorted(connected_components, key=len, reverse=True)

    # Iterate through each component, get the subgraph, and compute number of edges and cycles # noqa: E501
    for i, component in enumerate(cc_sorted):
        subgraph = G.subgraph(component)
        num_edges = subgraph.number_of_edges()
        cycles = nx.cycle_basis(subgraph)
        num_cycles = len(cycles)

        print(
            f"Component {i + 1} (size {len(component)}, edges {num_edges}, cycles {num_cycles})"  # noqa: E501
        )
        print(f"Cycles: {cycles}")
        print(f"Component: {component}\n")


def create_nodes(
    df: pd.DataFrame, feature_vectors: np.ndarray, prior_df: pd.DataFrame
) -> pd.DataFrame:
    nodes = []

    for i, row in df.iterrows():
        lid = row["ltable_id"]
        rid = row["rtable_id"]
        label = row["label"]
        feature = feature_vectors[i]
        prediction = prior_df.loc[i, "prediction"]
        confidence_score = prior_df.loc[i, "confidence_score"]

        nodes.append(
            Node(lid, rid, label, feature, prediction, confidence_score)
        )

    return nodes


class Node:
    def __init__(
        self,
        lid: int,
        rid: int,
        label: int,
        feature: np.ndarray,
        prediction: int,
        confidence_score: float,
    ):
        self.id = (lid, rid)
        self.label = label
        self.feature = feature
        self.prediction = prediction
        self.confidence_score = confidence_score
        self.cluster_id = -1

    def intersect(self, other):
        return self.id[0] == other.id[0] or self.id[1] == other.id[1]


class MRFWrapper:
    def __init__(
        self,
        prior_filepath: str,
        **kwargs,
    ):
        self.prior_data = pd.read_csv(prior_filepath)
        self.tune_lbp_hp = kwargs.get("tune_lbp_hp", False)

        if not self.tune_lbp_hp:
            self.num_iters = kwargs.get("num_iters", 200)
            self.damping = kwargs.get("damping", 0.5)
            self.temperature = kwargs.get("temperature", 0)

        self.fg = None  # Factor graph corresponding to MRF
        self.variables = None  # Random variables in MRF
        self.extrapolated_variables = None  # Extrapolated variables in MRF
        self.logger = get_logger()

    def get_connected_components(self):
        G = nx.Graph()

        for _, row in self.prior_data.iterrows():
            lid = row["l_id"]
            rid = row["r_id"]

            G.add_node(lid)
            G.add_node(rid)
            G.add_edge(lid, rid)

        # Sort the components by size in descending order
        connected_components = list(nx.connected_components(G))
        cc_sorted = sorted(connected_components, key=len, reverse=True)

        return cc_sorted, G

    def create_mrf_variables_and_unary_factors(self, subgraph):
        start = time.time()

        edges = list(subgraph.edges())
        var_identifiers = []

        for e in edges:
            if self.prior_data.loc[
                (self.prior_data["l_id"] == e[0])
                & (self.prior_data["r_id"] == e[1])
            ].empty:
                var_identifiers.append((e[1], e[0]))
            else:
                var_identifiers.append((e[0], e[1]))

        variables = vgroup.VarDict(num_states=2, variable_names=var_identifiers)

        uniq_var_identifiers = set(var_identifiers)
        extrapolated_var_identifiers = set()
        cycles = nx.cycle_basis(subgraph)

        for c in cycles:
            for i in range(len(c) - 2):
                if (c[i], c[i + 2]) not in uniq_var_identifiers:
                    extrapolated_var_identifiers.add((c[i], c[i + 2]))

        extrapolated_variables = vgroup.VarDict(
            num_states=2, variable_names=list(extrapolated_var_identifiers)
        )
        fg = fgraph.FactorGraph([variables, extrapolated_variables])

        end = time.time()
        self.logger.info(
            f"Time to create and add MRF variables: {end-start:.2f} seconds"
        )

        variables_for_unary_factors = []
        log_potentials = []

        # add unary factors
        start = time.time()

        for var_id in var_identifiers:
            var = variables.__getitem__(var_id)
            variables_for_unary_factors.append([var])

            row = self.prior_data.loc[
                (self.prior_data["l_id"] == var_id[0])
                & (self.prior_data["r_id"] == var_id[1])
            ].iloc[0]
            pred_proba = row["confidence_score"]

            # Get around the warning of dividing by zero encountered in log
            if pred_proba == 1:
                pred_proba = 1 - PRIOR_CONSTANT
            elif pred_proba == 0:
                pred_proba = PRIOR_CONSTANT

            if row.prediction == 1:
                prior = np.log(np.array([1 - pred_proba, pred_proba]))
            else:
                prior = np.log(np.array([pred_proba, 1 - pred_proba]))

            log_potentials.append(prior)

        for var_id in extrapolated_var_identifiers:
            var = extrapolated_variables.__getitem__(var_id)
            variables_for_unary_factors.append([var])

            # Extrapolated variables are from the same set and assumed to
            # belong to the negative class
            prior = np.log(np.array([1 - PRIOR_CONSTANT, PRIOR_CONSTANT]))
            log_potentials.append(prior)

        unary_factor_group = fgroup.EnumFactorGroup(
            variables_for_factors=variables_for_unary_factors,
            factor_configs=UNARY_FACTOR_CONFIG,
            log_potentials=np.array(log_potentials),
        )
        fg.add_factors(unary_factor_group)

        end = time.time()
        self.logger.info(f"Time to add unary factors: {end-start:.2f} seconds")

        self.fg = fg
        self.variables = variables
        self.extrapolated_variables = extrapolated_variables

    def create_ternary_factors(self, subgraph, mrf_hp_config: dict):
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
        variables_for_ternary_factors = set()
        cycles = nx.cycle_basis(subgraph)

        for c in cycles:
            for i in range(len(c) - 2):
                try:
                    var1 = self.variables.__getitem__((c[i], c[i + 1]))
                except ValueError:
                    var1 = self.variables.__getitem__((c[i + 1], c[i]))

                try:
                    var2 = self.variables.__getitem__((c[i], c[i + 2]))
                except ValueError:
                    try:
                        var2 = self.variables.__getitem__((c[i + 2], c[i]))
                    except ValueError:
                        var2 = self.extrapolated_variables.__getitem__(
                            (c[i], c[i + 2])
                        )

                try:
                    var3 = self.variables.__getitem__((c[i + 1], c[i + 2]))
                except ValueError:
                    var3 = self.variables.__getitem__((c[i + 2], c[i + 1]))

                variables_for_ternary_factors.add((var1, var2, var3))

        ternary_factor_group = fgroup.EnumFactorGroup(
            variables_for_factors=list(variables_for_ternary_factors),
            factor_configs=TERNARY_FACTOR_CONFIG,
            log_potentials=log_ternary_table,
        )
        self.fg.add_factors(ternary_factor_group)

        end = time.time()
        self.logger.info(
            f"Time to add ternary factors: {end-start:.2f} seconds"
        )

    def create_mrf(self, mrf_hp_config: dict) -> fgraph:
        # Get connected components of the raw data graph
        cc_sorted, G = self.get_connected_components()

        # Create MRF for the largest connected component
        subgraph = G.subgraph(cc_sorted[0])
        self.create_mrf_variables_and_unary_factors(subgraph)
        self.create_ternary_factors(subgraph, mrf_hp_config)

        return self.fg

    # LBP inference
    def run_inference(self, fg, mrf_hp_config: dict) -> dict:
        start_time = time.time()

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

    prior_dir = config.get("io", "prior_dir")
    valid_prior_filepath = os.path.join(prior_dir, "validation.csv")
    test_prior_filepath = os.path.join(prior_dir, "test.csv")
    random_seed = config.getint("hp_optimization", "random_seed")

    if args.mode == "hp_tuning":
        # Set global random state
        fix_global_random_state(config.getint("hp_optimization", "random_seed"))

        # Create MRF hyperparameter space
        hp_space = HyperparameterSpace(
            config.get("hp_optimization", "hp_spec_filepath"),
            config.getint("hp_optimization", "random_seed"),
        ).create_hp_space()

        # Create MRF wrapper
        if config.getboolean("hp_optimization", "tune_lbp_hp"):
            mrf_wrapper = MRFWrapper(
                valid_prior_filepath,
                tune_lbp_hp=True,
            )
        else:
            mrf_wrapper = MRFWrapper(
                valid_prior_filepath,
                tune_lbp_hp=False,
                num_iters=config.getint("hp_optimization", "num_iters"),
                damping=config.getfloat("hp_optimization", "damping"),
                temperature=config.getfloat("hp_optimization", "temperature"),
            )

        # Hyperparameter tuning
        tuning_engine = TuningEngine(config, mrf_wrapper, hp_space)
        best_hp_config = tuning_engine.run()

        test_mrf_wrapper = MRFWrapper(test_prior_filepath, tune_lbp_hp=True)
    else:
        assert args.mode == "inference", (
            f"Invalid mode: {args.mode}. Mode must either be hp_tuning or "
            "inference."
        )

        best_hp_config = {
            # "damping": 0.7858213721344507,
            # "num_iters": 921,
            # "temperature": 0.9910111614267442,
            "alpha": 0.9,  # 0.5768611571269355,
            "beta": 0.7,  # 0.5855811670070769,
            "gamma": 0.7,  # 0.016516056183874597,
            "delta": 0.7,  # 0.588351726474382,
        }
        logger.info(f"Best hyperparameters:\n{best_hp_config}")

        test_mrf_wrapper = MRFWrapper(test_prior_filepath, tune_lbp_hp=False)

    test_mrf = test_mrf_wrapper.create_mrf(dict(best_hp_config))
    results = test_mrf_wrapper.run_inference(test_mrf, dict(best_hp_config))

    evaluate_inference_results(
        test_mrf_wrapper.prior_data, results, log_predictions=True
    )
