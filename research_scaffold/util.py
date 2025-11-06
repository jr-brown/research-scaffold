"""
Generic Utility Functions
"""

from re import search, subn
from copy import deepcopy
from typing import Optional, TypeVar
from collections.abc import Callable
from datetime import datetime

try:
    from accelerate import PartialState
    from accelerate.logging import get_logger as _get_logger
    accelerate_partial_state = PartialState()
    is_main_process = accelerate_partial_state.is_main_process

except ModuleNotFoundError:
    from logging import getLogger as _get_logger
    is_main_process = True  # Do we want this to cover JAX parallelism as well?


def get_logger(*args, **kwargs):
    return _get_logger(*args, **kwargs)


def nones_to_empty_lists(*args: Optional[list]) -> list[list]:
    """Accepts any number of arguments and returns list of arguments.
    Arguments should each be list or None, and Nones are replaced by empty lists."""
    return [[] if l is None else l for l in args]


def nones_to_empty_dicts(*args: Optional[dict]) -> list[dict]:
    """Accepts any number of arguments and returns list of arguments.
    Arguments should each be dict or None, and Nones are replaced by empty dicts."""
    return [{} if d is None else d for d in args]


def get_time_stamp(include_seconds: bool = False) -> str:
    """Returns a string with the current date and time.
    This is formatted as 'YYYY-MM-DD_HH-MM' or 'YYYY-MM-DD_HH-MM-SS'"""
    date = str(datetime.now().date())
    time = str(datetime.now().time())

    if include_seconds:
        time = "-".join(time.split(":")).split(".", maxsplit=1)[0]
    else:
        time = "-".join(time.split(":")[:2])

    return f"{date}_{time}"


# generic TypeVar to show consistency of input and output types for type hinting
A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")


def check_name_sub_general(
    _x: A, new_name: str, count: int = 0, run_name_dummy: str = "RUN_NAME",
) -> tuple[A, int]:
    """Counts occurrences of run_name_dummy and substitutes these name.
    If _x is a dict or list, this is done recursively."""

    def inner(x):
        if isinstance(x, dict):
            return {k: inner(v) for k, v in x.items()}

        elif isinstance(x, list):
            return [inner(v) for v in x]

        elif isinstance(x, str) and search(run_name_dummy, x):
            revised_string, extra_count = subn(run_name_dummy, new_name, x)
            nonlocal count
            count += extra_count
            return revised_string

        else:
            return x

    return inner(_x), count  # type: ignore


def recursive_dict_update(
    base: dict,
    target: dict,
    assert_type_match=True,
    type_match_ignore_nones=True,
    extend_lists=False,
    compositions: Optional[dict] = None,
) -> dict:
    """
    Return new dictionary from recursively updating base dictionary with target dictionary.
    Options:
    - assert_type_match: If True, raises ValueError if types do not match for a key.
    - type_match_ignore_nones: If True, ignores type mismatches if either type is None.
    - extend_lists: If True, appends lists instead of replacing them.
    - compositions: Dict of functions to combine base and target values instead of replacing them.
    """
    compositions = compositions if compositions is not None else {}
    new = deepcopy(base)

    for k, v in target.items():

        if assert_type_match and (k in base.keys()):
            t1, t2 = type(base[k]), type(v)
            if t1 != t2 and (not type_match_ignore_nones or (t1 is None or t2 is None)):
                raise ValueError(f"Types do not match for key {k}, {t1} vs {t2}")

        if k not in base.keys():
            new[k] = v

        elif k in compositions.keys():
            new[k] = compositions[k](base[k], v)

        elif isinstance(v, dict) and isinstance(base[k], dict):
            new[k] = recursive_dict_update(
                base[k],
                v,
                assert_type_match=assert_type_match,
                type_match_ignore_nones=type_match_ignore_nones,
                extend_lists=extend_lists,
            )

        elif isinstance(v, list) and isinstance(base[k], list) and extend_lists:
            new[k] += v

        else:
            new[k] = v

    return new


def dmap(f: Callable[[B], C], d: dict[A, B]) -> dict[A, C]:
    return {k: f(v) for k, v in d.items()}


def key_list_get(_dict: dict, keys: list):
    if len(keys) == 0:
        raise Exception("Empty keys")

    elif len(keys) == 1:
        return _dict[keys[0]]

    else:
        k = keys[0]
        next_keys = keys[1:]
        return key_list_get(_dict[k], next_keys)


def merge_dicts(*args: dict) -> dict:
    _dict = dict()
    for arg in args:
        _dict.update(arg)
    return _dict
