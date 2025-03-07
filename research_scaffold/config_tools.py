"""
Tools for loading and executing experiments from config files.
"""

# Standard Library
import logging
import subprocess

from os import path, makedirs
from pprint import pformat
from typing import Any, Callable, Optional, Union
from functools import partial
from dataclasses import dataclass

# Third Party
try:
    import jax
    has_jax = True
except ModuleNotFoundError:
    has_jax = False

import wandb

# Local
from .util import (
    is_main_process,
    get_logger,
    nones_to_empty_lists,
    nones_to_empty_dicts,
    get_time_stamp,
    recursive_dict_update,
    check_name_sub_general,
)
from .file_io import load


log = get_logger(__name__)


### Type definitions
StringKeyDict = dict[str, Any]
FunctionMap = dict[str, Callable]
ConfigPathOrMultiple = Union[str, list[str]]
ConfigPathAxes = list[ConfigPathOrMultiple]


@dataclass
class Config:
    """Type definition for a Config."""

    name: str
    function_name: str
    # Fields with defaults
    time_stamp_name: Optional[bool] = False
    function_kwargs: Optional[StringKeyDict] = None
    function_args: Optional[list] = None
    log_file_path: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_tags: Optional[list[str]] = None

    @property
    def d(self):
        """Simple shorthand for self.__dict__"""
        return self.__dict__


@dataclass
class ProductExperimentSpec:
    """Type definition for a ProductExperimentSpec."""

    # Note that optional means that field can be None; but have no default values
    # (no defaults needed because this will always be constructed by read_experiment_set)
    repeats: int
    config_axes: ConfigPathAxes
    expt_root: Optional[str]
    expt_patch: Optional[str]


@dataclass
class MetaConfig:
    """Type definition for a MetaConfig."""

    experiments: list[ProductExperimentSpec]
    folder: Optional[str]
    common_root: Optional[str | list[str]]
    common_patch: Optional[str | list[str]]
    auto_increment_rng_seed: bool
    rng_seed_offset: int

    @property
    def d(self):
        """Simple shorthand for self.__dict__"""
        return self.__dict__


### Functions
def load_dict_from_yaml(yaml_path: str) -> StringKeyDict:
    """Loads a dictionary with string keys from file path (including .yaml extension)."""
    return load("yaml", yaml_path)


def load_config(cfg_path: str) -> Config:
    """Loads a config from corresponding file path (including .yaml extension)."""
    return Config(**load_dict_from_yaml(cfg_path))


def load_and_compose_config_steps(
    cfg_paths: list[str], compositions: Optional[dict[str, Callable]] = None
) -> Config:
    """Return single config from iteratively combining configs loaded from cfg_paths."""
    config_dict = {}

    for cfg_path in cfg_paths:
        partial_config_dict = load_dict_from_yaml(cfg_path)
        config_dict = recursive_dict_update(
            config_dict, partial_config_dict, compositions=compositions
        )

    return Config(**config_dict)


def parse_experiment_set(set_specific_dict: StringKeyDict) -> ProductExperimentSpec:
    """Can take in either a single experiment, or steps, or options or axes."""
    # check that only one of the single format options is present
    axes_info_formats = ["config_axes", "config_options", "config_steps", "config"]
    assert sum([key in set_specific_dict for key in axes_info_formats]) == 1
    # check that no other keys are present apart from those in the ProductExperimentSpec class
    assert set_specific_dict.keys() <= set(
        axes_info_formats + ["repeats", "expt_root", "expt_patch"]
    )

    if "config_axes" in set_specific_dict:
        axes = set_specific_dict["config_axes"]
    elif "config_options" in set_specific_dict:
        axes = [set_specific_dict["config_options"]]
    elif "config_steps" in set_specific_dict:
        axes = [[step] for step in set_specific_dict["config_steps"]]
    else:
        axes = [[set_specific_dict["config"]]]

    return ProductExperimentSpec(
        repeats=set_specific_dict.get("repeats", 1),
        config_axes=axes,
        expt_root=set_specific_dict.get("expt_root", None),
        expt_patch=set_specific_dict.get("expt_patch", None),
    )


def load_meta_config(meta_cfg_path: str) -> MetaConfig:
    """Loads a meta config from corresponding file path (including .yaml extension)."""
    mc_dict = load_dict_from_yaml(meta_cfg_path)
    experiments = [parse_experiment_set(specs) for specs in mc_dict["experiments"]]
    return MetaConfig(
        experiments=experiments,
        common_root=mc_dict.get("common_root", None),
        common_patch=mc_dict.get("common_patch", None),
        auto_increment_rng_seed=mc_dict.get("auto_increment_rng_seed", False),
        rng_seed_offset=mc_dict.get("rng_seed_offset", 0),
        folder=mc_dict.get("folder", ""),
    )


def execute_from_config(
    config: Config,  # Entire config is separately input to easily log it to wandb
    function_map: FunctionMap,
    function_name: str,
    function_args: Optional[list] = None,
    function_kwargs: Optional[StringKeyDict] = None,
    name: str = "unamed",
    time_stamp_name: bool = False,
    wandb_project: Optional[str] = None,
    wandb_group: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_tags: Optional[list[str]] = None,
    log_file_path: Optional[str] = None,
    run_name_dummy: str = "RUN_NAME",
):
    """
    Executes a function from a Config object.
    """
    check_name_sub = partial(check_name_sub_general, run_name_dummy=run_name_dummy)

    name_base = name

    if time_stamp_name:
        name = f"{name}_{get_time_stamp(include_seconds=True)}"

    # Add handler to log to file if necessary
    if log_file_path is not None:
        log_file_path, n_subs = check_name_sub(name, log_file_path, count=0)

        log_dir = path.dirname(log_file_path)

        if log_dir != "":
            makedirs(log_dir, exist_ok=True)

        root_logger = logging.getLogger()
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(root_logger.level)
        file_handler.setFormatter(root_logger.handlers[0].formatter)
        root_logger.addHandler(file_handler)
        file_log_cleanup_fn = lambda: root_logger.removeHandler(file_handler)
    else:
        n_subs = 0
        file_log_cleanup_fn = None

    log.info("========== Config Dict ===========\n" + pformat(config))
    log.info("Run Name: " + pformat(name))

    (function_args,) = nones_to_empty_lists(function_args)
    (function_kwargs,) = nones_to_empty_dicts(function_kwargs)

    # Substitute occurrences of RUN_NAME for the run name
    function_args, n_subs = check_name_sub(name, function_args, count=n_subs)
    function_kwargs, n_subs = check_name_sub(name, function_kwargs, count=n_subs)

    log.info(f"Made {n_subs} substitutions of {run_name_dummy} for {name}")

    if wandb_project is not None and is_main_process:
        group_name = wandb_group if wandb_group is not None else name_base

        with wandb.init(
            entity=wandb_entity,
            project=wandb_project,
            tags=wandb_tags,
            name=name,
            group=group_name,
            config=function_kwargs,
        ):  # type: ignore
            function_map[function_name](*function_args, **function_kwargs)

    else:
        function_map[function_name](*function_args, **function_kwargs)

    if has_jax:
        jax.clear_caches()

    if file_log_cleanup_fn is not None:
        file_log_cleanup_fn()


def combine_root_tgt_patch(
    tgt: str | list[str],
    common_root: Optional[str | list[str]] = None,
    common_patch: Optional[str | list[str]] = None,
) -> list[str]:
    """
    Combines together the target(s) with any common configs
    Options:
    - common_root: if str or list of str then these paths are prefixed to the start of tgt
    - common_patch: if str or list of str then these paths are appended to the end of tgt
    """

    if isinstance(tgt, str):
        tgt = [tgt]

    assert isinstance(tgt, list)

    if common_root is not None:
        if isinstance(common_root, str):
            common_root = [common_root]
        tgt = common_root + tgt

    if common_patch is not None:
        if isinstance(common_patch, str):
            common_patch = [common_patch]
        tgt = tgt + common_patch

    return tgt


def prepend_folder(
    config_stems: list[str],
    folder: Optional[str] = None,
) -> list[str]:
    """Prepends folder to each config stem in config_stems."""
    log.debug(f"Adding folder {folder} to config_stems {config_stems}")
    config_paths = [path.join(folder, t) for t in config_stems]
    return config_paths


def process_product_experiment_spec(
    product_experiment_specs: ProductExperimentSpec,
    folder: Optional[str] = None,
    common_root: Optional[str | list[str]] = None,
    common_patch: Optional[str | list[str]] = None,
) -> list[Config]:
    """
    Creates a list of 'product' configs specified via 'axes' of experiments.

    The 'axes' argument is a list of 'axis' objects representing different config settings to try.
    (each axis list of strings, where each string is a path to a config file).
    The function returns all combinations of experiments
    (by taking one experiment from each successive axis and concatenating),
    each of these combinations is repeated 'repeats' times.
    """
    axes = product_experiment_specs.config_axes
    repeats = product_experiment_specs.repeats
    expt_root = product_experiment_specs.expt_root
    expt_patch = product_experiment_specs.expt_patch

    assert repeats >= 0

    # get products of options from the axes
    stem_sequence_options = []
    for axis in axes:
        if not stem_sequence_options:
            stem_sequence_options = [[elem] for elem in axis]
        else:
            stem_sequence_options = [
                cfg + [elem] for elem in axis for cfg in stem_sequence_options
            ]

    # for each sequence of config path stems need to extend with shared configs and folder
    configs = []
    for config_stem_sequence in stem_sequence_options:
        for _ in range(repeats):
            stem_sequence_with_common = combine_root_tgt_patch(
                config_stem_sequence, common_root, common_patch
            )
            full_stem_sequence = combine_root_tgt_patch(
                stem_sequence_with_common, expt_root, expt_patch
            )
            full_path_sequence = prepend_folder(full_stem_sequence, folder)
            cfg = load_and_compose_config_steps(
                full_path_sequence,
                compositions={
                    "name": lambda x, y: f"{x}_{y}",
                    "wandb_tags": lambda x, y: x + y,  # string concatenation
                },
            )
            configs.append(cfg)

    return configs


def process_meta_config(mc: MetaConfig) -> list[Config]:
    """
    Generates a list of configs from the fields of a single meta config.
    """
    # experiments is a list of ProductExperimentSpec
    configs = []

    for exp_set_specs in mc.experiments:

        configs.extend(
            process_product_experiment_spec(
                exp_set_specs,
                folder=mc.folder,
                common_root=mc.common_root,
                common_patch=mc.common_patch,
            )
        )

    if mc.rng_seed_offset != 0 or mc.auto_increment_rng_seed:
        for i, config in enumerate(configs):
            config.function_kwargs["rng_seed"] = (
                mc.rng_seed_offset + config.function_kwargs.get("rng_seed", 0)
            )
            if mc.auto_increment_rng_seed:
                config.function_kwargs["rng_seed"] += i

    return configs


def execute_experiments(
    function_map: FunctionMap,
    config_path: Optional[str] = None,
    meta_config_path: Optional[str] = None,
) -> None:
    """Creates a sequence of configs from config_path or meta_config_path and executes them"""

    git_commit_hash = (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )
    log.info(f"Executing experiment, {git_commit_hash=}")

    if (config_path is not None) and (meta_config_path is not None):
        raise ValueError(
            "Exactly one of config_paths and meta_config_path must be specified"
        )

    if config_path is not None:
        log.info("Loading config...")
        configs = [load_config(config_path)]

    elif meta_config_path is not None:
        log.info("Loading meta config...")
        meta_config = load_meta_config(meta_config_path)
        log.info("========== Meta Config ===========\n" + pformat(meta_config))
        configs = process_meta_config(meta_config)

    else:
        log.warning("Please use -c or -m to specify a config or meta config to run!")
        configs = []

    for i, config in enumerate(configs):
        log.info(f"Executing config {i+1}/{len(configs)}")
        execute_from_config(config, function_map=function_map, **config.d)
