import time

from itertools import combinations

import pandas as pd

from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import Mplp
from pgmpy.models import MarkovNetwork

from openforge.utils.custom_logging import get_logger


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

    def create_mrf(self, mrf_hp_config: dict) -> MarkovNetwork:
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

        mrf = MarkovNetwork()

        # add variables and unary factors
        for row in self.prior_data.itertuples():
            var_name = row.relation_variable_name
            mrf.add_node(var_name)

            confdc_score = row.positive_label_confidence_score
            prior = [1 - confdc_score, confdc_score]
            unary_factor = DiscreteFactor(
                [var_name], cardinality=[2], values=prior
            )
            mrf.add_factors(unary_factor)

        num_nodes = len(mrf.nodes())
        num_unary_factors = len(mrf.get_factors())
        self.logger.info(
            f"Number of MRF variables / unary factors: {num_nodes}"
        )

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
            var2 = f"R_{combo[0]}-{combo[2]}"
            var3 = f"R_{combo[1]}-{combo[2]}"
            mrf.add_edges_from([(var1, var2), (var1, var3), (var2, var3)])

            ternary_factor = DiscreteFactor(
                [var1, var2, var3], cardinality=[2, 2, 2], values=ternary_table
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

    def run_mplp_inference(self, mrf) -> dict:
        mplp = Mplp(mrf)

        start_time = time.time()
        results = mplp.map_query(tighten_triplet=False)
        end_time = time.time()
        self.logger.info(f"Inference time: {end_time - start_time:.1f} seconds")

        return results
