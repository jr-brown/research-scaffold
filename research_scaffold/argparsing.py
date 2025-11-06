import logging
import argparse

from os import environ


def get_base_argparser() -> argparse.ArgumentParser:
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
    return parser


def process_base_args(args) -> tuple[str | None, str | None, str | None]:
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

    return args.config_path, args.meta_config_path, args.sweep_config_path

