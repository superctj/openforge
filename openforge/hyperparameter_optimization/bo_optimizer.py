from functools import partial

from ConfigSpace import ConfigurationSpace
from smac import BlackBoxFacade as BBFacade
from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario


def get_bo_optimizer(config, mrf_hp_space: ConfigurationSpace, target_function):
    scenario = Scenario(
        configspace=mrf_hp_space,
        output_directory=config["results"]["save_path"],
        deterministic=True,
        objectives="cost",  # minimize the objective
        n_trials=100,
        seed=int(config["knob_space"]["random_seed"]),
    )

    target_function = partial(
        target_function, seed=int(config["knob_space"]["random_seed"])
    )

    if config["config_optimizer"]["optimizer"] == "bo-gp":
        optimizer = BBFacade(scenario=scenario, target_function=target_function)
    elif config["config_optimizer"]["optimizer"] == "bo-rf":
        optimizer = HPOFacade(
            scenario=scenario, target_function=target_function
        )
    else:
        raise ValueError(f"Optimizer {optimizer} not supported.")

    return optimizer
