import logging
import random
from typing import Optional

log = logging.getLogger(__name__)


def test_multi_arg_config(
    root_arg: str,
    arg1: int,
    arg2: bool,
    patch_arg: str,
    expt_input_format: str,
    bonus_arg: Optional[str] = None,
    default_arg: str = "default argument",
    rng_seed: int = 42,
    printq: bool = False,
) -> None:

    print("Testing simple config")

    random.seed(rng_seed)
    runif = random.random()

    arg_description_string = f""""
        root_arg: {root_arg}
        arg1: {arg1}
        arg2: {arg2}
        patch_arg: {patch_arg}
        expt_input_format: {expt_input_format}
        bonus_arg: {bonus_arg}
        default_arg: {default_arg}
        rng: {runif}
        """
    if printq:
        print(arg_description_string)
