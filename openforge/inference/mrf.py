import time

from itertools import combinations

import pandas as pd

from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import Mplp
from pgmpy.models import MarkovNetwork
from sklearn.metrics import accuracy_score, f1_score


def convert_var_name_to_var_id(var_name):
    var_id = tuple(int(elem) for elem in var_name.split("_")[1].split("-"))

    return var_id


class MRFWrapper:
    def __init__(self, prior_filepath: str, num_concepts: int, logger):
        self.prior_data = pd.read_csv(prior_filepath)
        self.num_concepts = num_concepts
        self.logger = logger

    def create_mrf(self, ternary_table):
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
        # self.logger.info(f"MRF nodes:\n{mrf.nodes()}")
        assert num_nodes == num_unary_factors
        assert mrf.check_model()

        start = time.time()
        ternary_combos = combinations(range(1, self.num_concepts + 1), 3)
        end = time.time()
        self.logger.info(
            f"Time taken to generate ternary combos: {end-start:.2f} seconds"
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
        # self.logger.info(f"MRF edges:\n{mrf.edges()}")

        mrf_factors = mrf.get_factors()
        self.logger.info(f"Number of MRF factors: {len(mrf_factors)}")
        # self.logger.info(f"MRF factors:\n{mrf_factors}")

        return mrf

    def run_mplp_inference(self, mrf) -> dict:
        mplp = Mplp(mrf)

        start_time = time.time()
        results = mplp.map_query(tighten_triplet=False)
        end_time = time.time()
        self.logger.info(f"Inference time: {end_time - start_time:.1f} seconds")

        return results

    def evaluate_results(self, results: dict):
        y_true, y_prior, y_pred = [], [], []

        for row in self.prior_data.itertuples():
            self.logger.info("-" * 80)

            var_name = row.relation_variable_name
            var_label = row.relation_variable_label
            pred = results[var_name]

            y_true.append(row.relation_variable_label)
            y_pred.append(pred)

            if row.positive_label_confidence_score >= 0.5:
                ml_pred = 1
                log_msg = (
                    f"Prior for variable {var_name}: ({ml_pred}, "
                    f"{row.positive_label_confidence_score:.2f})"
                )
            else:
                ml_pred = 0
                log_msg = (
                    f"Prior for variable {var_name}: ({ml_pred}, "
                    f"{1 - row.positive_label_confidence_score:.2f})"
                )

            y_prior.append(ml_pred)
            self.logger.info(log_msg)
            self.logger.info(f"Posterior for variable {var_name}: {pred}")
            self.logger.info(f"True label for variable {var_name}: {var_label}")

            if ml_pred != var_label:
                self.logger.info("Prior prediction is incorrect.")
            if pred != row.relation_variable_label:
                self.logger.info("Posterior prediction is incorrect.")

        self.logger.info("-" * 80)
        self.logger.info(f"Number of test instances: {len(self.prior_data)}")

        self.logger.info(
            f"Prior test accuracy: {accuracy_score(y_true, y_prior):.2f}"
        )
        self.logger.info(f"Prior F1 score: {f1_score(y_true, y_prior):.2f}")

        self.logger.info(
            f"MRF test accuracy: {accuracy_score(y_true, y_pred):.2f}"
        )
        self.logger.info(f"MRF F1 score: {f1_score(y_true, y_pred):.2f}")
