"""
Minimal example of a main file for running experiments from configs.
"""

import sys
import logging
import argparse
from tqdm.contrib.logging import logging_redirect_tqdm

# Parse arguments first (before other imports) to set up logging/device settings
parser = argparse.ArgumentParser(description="Run experiments from configs")
parser.add_argument("-c", "--config_path", help="Single config file")
parser.add_argument("-m", "--meta_config_path", help="Meta config with multiple experiments")
parser.add_argument("-s", "--sweep_config_path", help="Wandb sweep config")
parser.add_argument("-l", "--loglevel", default="info", help="Logging level")
args = parser.parse_args()

# Configure logging
logging.basicConfig(
    level=args.loglevel.upper(),
    format="%(asctime)s %(module)-15s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# Import local modules
from functions.examples import (
    example_simple,
    example_with_logging,
    example_composition,
    example_sweep,
    example_sleep,
)

from research_scaffold import execute_experiments

log = logging.getLogger(__name__)

# Map function names to actual functions (used by configs)
function_map = {
    "example_simple": example_simple,
    "example_with_logging": example_with_logging,
    "example_composition": example_composition,
    "example_sweep": example_sweep,
    "example_sleep": example_sleep,
}

if __name__ == "__main__":
    with logging_redirect_tqdm():
        log.info("##### Program Start #####")
        execute_experiments(
            function_map=function_map,
            config_path=args.config_path,
            meta_config_path=args.meta_config_path,
            sweep_config_path=args.sweep_config_path,
        )
        log.info("##### Program End #####")
    
    sys.exit()
