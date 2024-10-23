"""
Provides utility functions for file input/output.
"""

import yaml
import json
import pickle
import logging

from os import path, remove
from time import sleep
from random import uniform
from typing import Callable, Union, TypeVar
from pathlib import Path
from functools import partial


log = logging.getLogger(__name__)


A = TypeVar("A")
PathStr = Union[str, Path]


def wait_for_user(txt: str = "\nPress enter to continue... "):
    """Prints txt and waits for user to press enter."""
    input(txt)


# Key: (suffix, treat as binary, load func, save func)
load_save_methods: dict[str, tuple[str, bool, Callable, Callable]] = {
    "pickle": (".pickle", True, pickle.load, pickle.dump),
    "json": (
        ".json",
        False,
        json.load,
        partial(json.dump, ensure_ascii=False, indent=4),
    ),
    "yaml": (".yaml", False, yaml.full_load, yaml.dump),
}


def _load_from_path_preprocess(file_path: PathStr, suffix: str):
    """Checks that the file_path is a file, and has correct suffix."""
    file_path = Path(file_path)

    if file_path.suffix == "":
        file_path = file_path.with_suffix(suffix)
    elif file_path.suffix != suffix:
        raise ValueError(f"{file_path=} has {file_path.suffix=}, expected {suffix=}")

    if not file_path.is_file():
        raise ValueError(f"{file_path} not a file")

    return file_path


def load(method: str, file_path: PathStr):
    """Loads data from file_path looking up method from `load_save_methods`"""
    suffix, use_binary, load_func, _ = load_save_methods[method]
    file_path = _load_from_path_preprocess(file_path, suffix)
    open_type = "rb" if use_binary else "r"

    with open(file_path, open_type) as f:
        data = load_func(f)

    return data


def _write_to_path_preprocess(file_path: PathStr, suffix: str, overwrite: bool = False):
    """Checks correct suffix, avoids unintended overwrites and creates parent directories."""
    file_path = Path(file_path)
    if file_path.suffix != suffix:
        file_path = file_path.with_suffix(suffix)
    if file_path.exists() and not overwrite:
        raise FileExistsError
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path


def save(method: str, data, file_path: PathStr, overwrite: bool = False) -> Path:
    """Saves data to file_path using method from `load_save_methods`"""
    suffix, use_binary, _, save_func = load_save_methods[method]
    file_path = _write_to_path_preprocess(file_path, suffix, overwrite=overwrite)
    open_type = "wb" if use_binary else "w"
    with open(file_path, open_type) as f:
        save_func(data, f)
    return file_path


def transform(
    method: str, transforming_function: Callable, file_path: PathStr, default_data=None
):
    """Loads data from file_path, applies transform, and saves back to file_path.
    If default_data is provided and file_path does not exist, default_data is used."""
    suffix, _, _, _ = load_save_methods[method]
    file_path = _write_to_path_preprocess(file_path, suffix, overwrite=True)
    lock_path = f"{file_path}.lock"

    # Wait until no-one else has lock on file
    while path.exists(lock_path):
        log.info(f"{file_path} in use, waiting 1-5s...")
        sleep(uniform(1, 5))

    # Create lock on file, and then make sure nearly all execution paths will end up removing lock
    with open(lock_path, "w", encoding="utf-8") as f:
        # Lennie allowed copilot to add utf-8 to satisfy pylint; need to check no problems
        f.write("")

    using_default_data = False

    try:
        if file_path.exists():
            data = load(method, file_path)
        elif default_data is not None:
            data = default_data
            using_default_data = True
        else:
            raise ValueError(f"{file_path=} does not exist and {default_data=}")

        new_data = transforming_function(data)

    except Exception as load_process_exception:
        log.error("Encountered error when loading or processing data, removing lock")
        remove(lock_path)
        raise load_process_exception

    try:
        save(method, new_data, file_path, overwrite=True)
        remove(lock_path)

    except (
        OSError,
        IOError,
        pickle.PicklingError,
        yaml.YAMLError,
        json.JSONDecodeError,
    ) as save_exception:
        try:
            if using_default_data:
                log.error(
                    "Error saving transformed data, original data was supplied default, removing file and lock"
                )
                remove(file_path)
                remove(lock_path)

            else:
                log.error(
                    "Error saving transformed data, resaving old data and removing lock"
                )
                save(method, data, file_path, overwrite=True)
                remove(lock_path)

        except Exception as action_required_exception:
            log.critical(
                "Error handling failed! Lock has not been removed, manual intervention required!"
            )
            raise action_required_exception

        else:
            raise save_exception
