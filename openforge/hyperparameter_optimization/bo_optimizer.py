from configparser import ConfigParser

from ConfigSpace import ConfigurationSpace
from smac import BlackBoxFacade as BBFacade
from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario


def get_bo_optimizer(
    exp_config: ConfigParser, mrf_hp_space: ConfigurationSpace, target_function
):
    scenario = Scenario(
        configspace=mrf_hp_space,
        output_directory=exp_config["mrf"]["log_dir"],
        deterministic=True,
        objectives="cost",  # minimize the objective
        n_trials=exp_config.getint("hp_optimization", "n_trials"),
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
