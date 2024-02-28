import time

from itertools import combinations

import pandas as pd
import pyAgrum as gum

from openforge.utils.custom_logging import get_logger
from openforge.utils.mrf_common import convert_var_name_to_var_id


class MRFWrapper:
    def __init__(self, prior_filepath: str):
        self.prior_data = pd.read_csv(prior_filepath)
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

    def create_mrf(self, mrf_hp_config: dict) -> gum.MarkovRandomField:
        ternary_table = [
            mrf_hp_config["ternary_alpha"],  # 0, 0, 0
            mrf_hp_config["ternary_beta"],  # 0, 0, 1
            mrf_hp_config["ternary_beta"],  # 0, 1, 0
            0,  # 0, 1, 1
            mrf_hp_config["ternary_beta"],  # 1, 0, 0
            0,  # 1, 0, 1
            0,  # 1, 1, 0
            mrf_hp_config["ternary_gamma"],  # 1, 1, 1
        ]

        mrf = gum.MarkovRandomField()
        # Map from unary clique id to corresponding variable
        unaryid_var_map = {}

        # Add variables and unary factors
        for row in self.prior_data.itertuples():
            var_name = row.relation_variable_name
            var_id = convert_var_name_to_var_id(var_name)
            var_descr = (
                f"Relation variable over pair of concepts {var_id[0]} and "
                f"{var_id[1]}"
            )
            var = gum.IntegerVariable(var_name, var_descr, [0, 1])

            mrf.add(var)
            unaryid_var_map[var_id] = var

            confdc_score = row.positive_label_confidence_score
            prior = [1 - confdc_score, confdc_score]

            unary_factor = gum.Potential().add(var).fillWith(prior)
            mrf.addFactor(unary_factor)

        assert len(mrf.names()) == len(unaryid_var_map)
        self.logger.info(
            f"Number of MRF variables / unary factors: {len(mrf.names())}"
        )

        # Add ternary factors
        start = time.time()
        ternary_combos = combinations(range(1, self.num_concepts + 1), 3)
        end = time.time()
        self.logger.info(
            f"Time to generate ternary combos: {end-start:.2f} seconds"
        )

        start = time.time()
        for combo in ternary_combos:
            var1 = unaryid_var_map[(combo[0], combo[1])]
            var2 = unaryid_var_map[(combo[0], combo[2])]
            var3 = unaryid_var_map[(combo[1], combo[2])]

            ternary_factor = (
                gum.Potential()
                .add(var1)
                .add(var2)
                .add(var3)
                .fillWith(ternary_table)
            )
            mrf.addFactor(ternary_factor)

        end = time.time()
        self.logger.info(
            f"Time to add ternary factors: {end-start:.2f} seconds"
        )
        self.logger.info(f"Number of MRF edges: {len(mrf.edges())}")
        self.logger.info(f"Number of MRF factors: {len(mrf.get_factors())}")

        return mrf

    # Shafer-Shenoy inference
    # https://pyagrum.readthedocs.io/en/1.12.1/MRFInference.html
    def run_ss_inference(self, mrf) -> dict:
        ss = gum.ShaferShenoyMRFInference(mrf)

        start_time = time.time()
        ss.makeInference()
        end_time = time.time()
        self.logger.info(f"Inference time: {end_time - start_time:.1f} seconds")

        results = {}

        for row in self.prior_df.itertuples():
            var_name = row.relation_variable_name
            posterior = ss.posterior(var_name)
            pred = posterior.argmax()[0][0][var_name]
            # prob = posterior.argmax()[1]
            results[var_name] = pred

        return results
