import argparse
import os

from openforge.hp_optimization.hp_space import MRFHyperparameterSpace
from openforge.hp_optimization.tuning import TuningEngine
from openforge.inference.pgmpy_mrf import MRFWrapper
from openforge.utils.custom_logging import create_custom_logger
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
    ).create_hp_space()

    # Create logger
    log_dir = config.get("results", "log_dir")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = create_custom_logger(log_dir)
    printable_config = {section: dict(config[section]) for section in config}
    logger.info(f"Experiment configuration:\n{printable_config}\n")

    # Create MRF wrapper
    mrf_wrapper = MRFWrapper(config.get("mrf", "prior_filepath"))

    # Hyperparameter tuning
    tuning_engine = TuningEngine(config, mrf_wrapper, hp_space)
    tuning_engine.run()
