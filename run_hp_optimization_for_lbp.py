import argparse
import os

from openforge.hp_optimization.hp_space import HyperparameterSpace
from openforge.hp_optimization.tuning import TuningEngine
from openforge.inference.pgmax_mrf import MRFWrapper
from openforge.utils.custom_logging import create_custom_logger
from openforge.utils.util import fix_global_random_state, parse_config

os.environ["XLA_FLAGS"] = "--xla_dump_to=/home/congtj/openforge"


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
            config.get("mrf_lbp", "prior_filepath"),
            tune_lbp_hp=True,
        )
    else:
        mrf_wrapper = MRFWrapper(
            config.get("mrf_lbp", "prior_filepath"),
            tune_lbp_hp=False,
            num_iters=config.getint("mrf_lbp", "num_iters"),
            damping=config.getfloat("mrf_lbp", "damping"),
            temperature=config.getfloat("mrf_lbp", "temperature"),
        )

    # Hyperparameter tuning
    tuning_engine = TuningEngine(config, mrf_wrapper, hp_space)
    tuning_engine.run()
