import argparse

from openforge.hyperparameter_optimization.hp_space import (
    MRFHyperparameterSpace,
)
from openforge.hyperparameter_optimization.tuning import TuningEngine
from openforge.inference.mrf import MRFWrapper
from openforge.utils.util import fix_global_random_state, parse_config


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
    hp_space = MRFHyperparameterSpace(
        config.get("hp_optimization", "hp_spec_filepath"),
        config.getint("hp_optimization", "random_seed"),
    )

    # Create MRF wrapper
    mrf_wrapper = MRFWrapper(config)

    # Hyperparameter tuning
    tuning_engine = TuningEngine(config, mrf_wrapper, hp_space)
