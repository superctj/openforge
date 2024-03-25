from configparser import ConfigParser, NoOptionError
from functools import partial

from ConfigSpace import ConfigurationSpace
from smac import BlackBoxFacade as BBFacade
from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario


def get_bo_optimizer(
    exp_config: ConfigParser,
    hp_space: ConfigurationSpace,
    target_function: object,
):
    try:
        output_dir = exp_config.get("results", "log_dir")
    except NoOptionError:
        output_dir = exp_config.get("results", "output_dir")

    scenario = Scenario(
        configspace=hp_space,
        output_directory=output_dir,
        deterministic=True,
        objectives="cost",  # minimize the objective
        n_trials=exp_config.getint("hp_optimization", "n_trials"),
        seed=exp_config.getint("hp_optimization", "random_seed"),
    )

    target_function = partial(
        target_function,
        seed=exp_config.getint("hp_optimization", "random_seed"),
    )

    if exp_config.get("hp_optimization", "optimizer") == "bo-gp":
        optimizer = BBFacade(scenario=scenario, target_function=target_function)
    elif exp_config.get("hp_optimization", "optimizer") == "bo-rf":
        optimizer = HPOFacade(
            scenario=scenario, target_function=target_function
        )
    else:
        raise ValueError(f"Optimizer {optimizer} is not supported.")

    return optimizer
