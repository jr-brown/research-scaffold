"""
Minimal example of a main file for running experiments from configs.
"""

# Standard library
import sys
import logging
import argparse

from os import environ
from typing import Optional
from collections.abc import Callable

# For testing multi device semantics
# TODO: Make this a proper parsed argument
# environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

# Pre-import configuration
config_path: Optional[str] = None
meta_config_path: Optional[str] = None
sweep_config_path: Optional[str] = None

# Import structure note:
# Args and kwargs need to be parsed and interpreted before importing any other library,
# since some things might need to happen before any other imports to properly take effect
# E.g. setting log levels, cuda things


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run experiments/analyses/tests from configs"
    )
    parser.add_argument("-c", "--config_path", default=None, help="Config file to run")
    parser.add_argument(
        "-m",
        "--meta_config_path",
        default=None,
        help="Meta config file that can specify many experiments, each possibly composing configs",
    )
    parser.add_argument(
        "-s",
        "--sweep_config_path",
        default=None,
        help="Sweep config file for wandb hyperparameter search",
    )
    parser.add_argument(
        "-l", "--loglevel", default="info", help="Provide logging level"
    )
    parser.add_argument(
        "-e",
        "--externalLoglevel",
        default="warning",
        help="Provide logging level for external libraries",
    )
    parser.add_argument(
        "-f",
        "--force_cpu",
        default=False,
        action="store_true",
        help="Force CPU usage instead of GPU/TPU",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.loglevel.upper(),
        format="%(asctime)s %(module)-15s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # TODO: Extend this to all external libraries
    external_logger_names = ["absl", "jax"]

    for name in external_logger_names:
        logging.getLogger(name).setLevel(args.externalLoglevel.upper())

    if args.force_cpu:
        environ["CUDA_VISIBLE_DEVICES"] = ""

    config_path = args.config_path
    meta_config_path = args.meta_config_path
    sweep_config_path = args.sweep_config_path


# Third-party
from tqdm.contrib.logging import logging_redirect_tqdm

# Local
from functions.jason_examples import example_simple_config, example_log_levels, example_sweep_function
from functions.lennie_examples import example_multi_arg_config
from research_scaffold import execute_experiments

log = logging.getLogger(__name__)


function_map: dict[str, Callable] = {
    f.__name__: f for f in [example_simple_config, example_log_levels, example_multi_arg_config, example_sweep_function]
}


if __name__ == "__main__":
    with logging_redirect_tqdm():
        log.info("##### Program Start #####")
        execute_experiments(
            function_map=function_map,
            config_path=config_path,
            meta_config_path=meta_config_path,
            sweep_config_path=sweep_config_path,
        )
        log.info("##### Program End #####")

    sys.exit()  # Helps close potentially hanging threads that may get produced by some libraries
