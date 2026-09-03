"""
Jason R Brown's research scaffold
Refactored by Lennie Wells, October 2024
"""

# Keep this before any other imports / code
from beartype.claw import beartype_this_package
beartype_this_package()

from .config_tools import execute_experiments, build_configs, dry_run_report
from .argparsing import get_base_argparser, process_base_args
from .util import get_logger

from importlib.metadata import version
__version__ = version("research-scaffold")
