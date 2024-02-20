import argparse

from openforge.hyperparameter_optimization.hp_space import (
    MRFHyperparameterSpace,
)
from openforge.hyperparameter_optimization.tuning import TuningEngine
from openforge.inference.mrf import MRFWrapper
from openforge.utils.custom_logging import get_custom_logger
from openforge.utils.util import (
    create_dir,
    fix_global_random_state,
    parse_config,
)


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
    ).create_hp_space()

    # Create logger
    log_dir = config.get("mrf", "log_dir")
    create_dir(log_dir, force=False)
    logger = get_custom_logger(log_dir)

    # Create MRF wrapper
    mrf_wrapper = MRFWrapper(
        config.get("mrf", "prior_filepath"),
        config.getint("mrf", "num_concepts"),
        logger,
    )

    # Hyperparameter tuning
    tuning_engine = TuningEngine(config, mrf_wrapper, hp_space, logger)
    tuning_engine.run()
