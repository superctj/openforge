import json

import ConfigSpace.hyperparameters as CSH

from ConfigSpace import ConfigurationSpace


class MRFHyperparameterSpace:
    def __init__(self, hp_spec_filepath: str, random_seed: int):
        self.random_seed = random_seed
        self.hp_spec = self.read_hp_spec(hp_spec_filepath)

    def read_hp_spec(self, hp_spec_filepath: str) -> list[dict]:
        """
        Read a JSON file specifying the hyperparameters of the MRF model.

        Args:
            hp_spec_filepath (str): Path to the JSON file.

        Returns:
            hp_spec (list[dict]): List of hyperparameter specifications.
        """

        with open(hp_spec_filepath, "r") as f:
            hp_spec = json.load(f)

        return hp_spec

    def create_hp_space(self) -> ConfigurationSpace:
        """
        Create hyperparameter space for the MRF model.

        Returns:
            hp_space (ConfigurationSpace): ConfigurationSpace object.
        """

        all_hps = []

        for hp_spec in self.hp_spec:
            hp_name, hp_type = hp_spec["name"], hp_spec["type"]

            if hp_type == "float":
                hp = CSH.UniformFloatHyperparameter(
                    name=hp_name,
                    lower=float(hp_spec["min_val"]),
                    upper=float(hp_spec["max_val"]),
                    default_value=float(hp_spec["default_val"]),
                )
            elif hp_type == "int":
                hp = CSH.UniformIntegerHyperparameter(
                    name=hp_name,
                    lower=int(hp_spec["min_val"]),
                    upper=int(hp_spec["max_val"]),
                    default_value=int(hp_spec["default_val"]),
                )
            else:
                raise ValueError(f"Unsupported hyperparameter type: {hp_type}.")

            all_hps.append(hp)

        hp_space = ConfigurationSpace(seed=self.random_seed)
        hp_space.add_hyperparameters(all_hps)

        return hp_space
