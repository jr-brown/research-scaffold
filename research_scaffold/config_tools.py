"""
Tools for loading and executing experiments from config files.
"""

# Standard Library
import logging
import subprocess
import sys
import time

import multiprocessing as mp
from dataclasses import replace, asdict
from os import path, makedirs
from pprint import pformat
from typing import Optional
from collections.abc import Callable

# Third Party
import gc

try:
    import jax

    has_jax = True
except ModuleNotFoundError:
    has_jax = False

try:
    import torch

    has_torch = True
except ModuleNotFoundError:
    has_torch = False

import wandb
import yaml


# Local
from .types import (
    StringKeyDict,
    FunctionMap,
    ConfigInput,
    ConfigInputMultiple,
    InstanceConfig,
    Config,
    ProductExperimentSpec,
    SweepExperimentSpec,
    SweepConfig,
    ExperimentSpec,
    MetaConfig,
    ResolvedExperiments,
)
from .util import (
    is_main_process,
    get_logger,
    get_time_stamp,
    nones_to_empty_lists,
    nones_to_empty_dicts,
    recursive_dict_update,
    load_config_dict,
    resolve_run_names,
    substitute_placeholders,
)
from .file_io import load, save

from .remote_execution import execute_config_remotely, execute_sweep_remotely


log = get_logger(__name__)


### Functions

def detect_config_type(config_dict: StringKeyDict) -> str:
    """
    Detect config type from its characteristic keys.
    Returns "single", "meta", or "sweep".
    """
    if "experiments" in config_dict or "base" in config_dict:
        return "meta"
    if "method" in config_dict and "parameters" in config_dict:
        return "sweep"
    return "single"


def get_git_commit_hash() -> str | None:
    """Get the current git commit hash, or None if not in a git repo."""
    try:
        result = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        )
        return result.decode("ascii").strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def load_dict_from_yaml(yaml_path: str) -> StringKeyDict:
    """Loads a dictionary with string keys from file path (including .yaml extension)."""
    return load("yaml", yaml_path)



def load_config(config_path_or_dict: ConfigInput) -> Config:
    """Loads a config from corresponding file path (including .yaml extension)."""
    config_dict = dict(load_config_dict(config_path_or_dict))
    if 'instance' in config_dict and isinstance(config_dict['instance'], dict):
        config_dict['instance'] = InstanceConfig(**config_dict['instance'])
    return Config(**config_dict)


def resolve_config_dict(config_path_or_dict: ConfigInput) -> StringKeyDict:
    """Load a config dict, automatically resolving meta-configs that produce exactly 1 config."""
    config_dict = load_config_dict(config_path_or_dict)
    if "experiments" not in config_dict:
        return config_dict
    meta_config = load_meta_config(config_dict)
    configs = process_meta_config(meta_config)
    if len(configs) != 1:
        raise ValueError(
            f"Meta-config used as base must produce exactly 1 config, got {len(configs)}."
        )
    return {k: v for k, v in configs[0].d.items() if v is not None}


def compose_config_dicts(
    config_dicts: list[StringKeyDict],
    compositions: Optional[dict[str, Callable]] = None,
) -> StringKeyDict:
    """Return single dict from iteratively merging config_dicts (later dicts take precedence)."""
    config_dict = {}
    for partial in config_dicts:
        config_dict = recursive_dict_update(
            config_dict, partial, compositions=compositions
        )
    return config_dict


def load_and_compose_config_steps(
    cfg_paths: list[ConfigInput],
    compositions: Optional[dict[str, Callable]] = None,
    bonus_dict: dict = {},
) -> Config:
    """Return single config from iteratively combining configs loaded from cfg_paths (paths or inline dicts)."""
    config_dict = compose_config_dicts(
        [resolve_config_dict(p) for p in cfg_paths] + [bonus_dict],
        compositions=compositions,
    )

    if 'instance' in config_dict and isinstance(config_dict['instance'], dict):
        config_dict['instance'] = InstanceConfig(**config_dict['instance'])
    return Config(**config_dict)


def parse_experiment_set(set_specific_dict: StringKeyDict) -> ExperimentSpec:
    """Can take in either a single experiment, or steps, or options or axes, or a sweep."""
    
    # Check if this is a sweep experiment
    if "sweep_config" in set_specific_dict:
        # Validate sweep experiment format
        allowed_keys = ["sweep_config", "base_config", "sweep_count"]
        if not set_specific_dict.keys() <= set(allowed_keys):
            raise ValueError(
                f"Sweep experiment has invalid keys: {set_specific_dict.keys() - set(allowed_keys)}"
            )

        return SweepExperimentSpec(
            sweep_config=set_specific_dict["sweep_config"],
            base_config=set_specific_dict.get("base_config", None),
            sweep_count=set_specific_dict.get("sweep_count", None),
        )
    
    # Otherwise, it's a regular product experiment
    # check that only one of the single format options is present
    axes_info_formats = ["config_axes", "config_options", "config_steps", "config"]
    if sum([key in set_specific_dict for key in axes_info_formats]) != 1:
        raise ValueError(
            f"Experiment must have exactly one of {axes_info_formats}, got keys: {list(set_specific_dict.keys())}"
        )
    allowed_keys = set(axes_info_formats + ["repeats", "expt_root", "expt_patch"])
    if not set_specific_dict.keys() <= allowed_keys:
        raise ValueError(
            f"Experiment has invalid keys: {set_specific_dict.keys() - allowed_keys}"
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


def resolve_meta_config_dict(meta_cfg_input: ConfigInput) -> StringKeyDict:
    """Load a meta config dict, recursively composing any `base` meta configs it builds on."""
    mc_dict = dict(load_config_dict(meta_cfg_input))
    base = mc_dict.pop("base", None)
    if base is None:
        return mc_dict
    if isinstance(base, (str, dict)):
        base = [base]
    return compose_config_dicts([resolve_meta_config_dict(b) for b in base] + [mc_dict])


def load_meta_config(meta_cfg_path: ConfigInput) -> MetaConfig:
    """Loads a meta config from path or inline dict (including .yaml extension if path)."""
    mc_dict = resolve_meta_config_dict(meta_cfg_path)
    experiments = [parse_experiment_set(specs) for specs in mc_dict["experiments"]]
    return MetaConfig(
        experiments=experiments,
        bonus_dict=mc_dict.get("bonus_dict", {}),
        common_root=mc_dict.get("common_root", None),
        common_patch=mc_dict.get("common_patch", None),
        auto_increment_rng_seed=mc_dict.get("auto_increment_rng_seed", False),
        rng_seed_offset=mc_dict.get("rng_seed_offset", 0),
        folder=mc_dict.get("folder", ""),
        parallel=mc_dict.get("parallel", False),
        start_method=mc_dict.get("start_method", None),
        max_concurrent=mc_dict.get("max_concurrent", None),
    )


def load_sweep_config(sweep_cfg_path: ConfigInput) -> SweepConfig:
    """Loads a sweep config from path or inline dict (including .yaml extension if path)."""
    sc_dict = dict(load_config_dict(sweep_cfg_path))
    if 'instance' in sc_dict and isinstance(sc_dict['instance'], dict):
        sc_dict['instance'] = InstanceConfig(**sc_dict['instance'])
    return SweepConfig(**sc_dict)


def execute_from_config(
    config: Config,
    function_map: FunctionMap,
    sweep_name: Optional[str] = None,
    launch_time_stamp: Optional[str] = None,
):
    """
    Executes a function from a Config object.
    """

    # Resolve names BEFORE dispatching to remote so timestamps are consistent
    resolved_names = resolve_run_names(
        name=config.name,
        time_stamp_name=config.time_stamp_name,
        time_stamp_group=config.time_stamp_group,
        wandb_group=config.wandb_group,
        sweep_name=sweep_name,
        launch_time_stamp=launch_time_stamp,
    )
    name = resolved_names["name"]
    group = resolved_names["group"]

    if config.instance is not None:
        instance = config.instance
        config.instance = None
        config.name = name
        config.time_stamp_name = False
        config.wandb_group = group
        config.time_stamp_group = False
        execute_config_remotely(instance, config, resolved_names)
        return

    def _sub(value):
        return substitute_placeholders(value, resolved_names)

    # Add handler to log to file if necessary
    if config.log_file_path is not None:
        log_file_path = _sub(config.log_file_path)

        log_dir = path.dirname(log_file_path)

        if log_dir != "":
            makedirs(log_dir, exist_ok=True)

        root_logger = logging.getLogger()
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(root_logger.level)
        if root_logger.handlers:
            file_handler.setFormatter(root_logger.handlers[0].formatter)
        root_logger.addHandler(file_handler)

    else:
        log_file_path = None
        file_handler = None

    try:
        log.info("========== Config Dict ===========\n" + pformat(config))
        log.info("Run Name: " + pformat(name))

        (function_args,) = nones_to_empty_lists(config.function_args)
        (function_kwargs,) = nones_to_empty_dicts(config.function_kwargs)

        # Substitute occurrences of RUN_NAME, RUN_GROUP, and SWEEP_NAME
        function_args = _sub(function_args)
        function_kwargs = _sub(function_kwargs)

        # Save config to file if requested
        if config.save_config_path is not None:
            save_config_path_sub = _sub(config.save_config_path)

            # Create a dict with the full config including substituted values
            config_to_save = {
                "name": name,
                "function_name": config.function_name,
                "function_args": function_args,
                "function_kwargs": function_kwargs,
                "wandb_project": config.wandb_project,
                "wandb_entity": config.wandb_entity,
                "wandb_group": config.wandb_group,
                "wandb_tags": config.wandb_tags,
                "log_file_path": log_file_path,
                "save_config_path": save_config_path_sub,
            }

            try:
                save("yaml", config_to_save, save_config_path_sub, overwrite=True)
                log.info(f"Saved config to: {save_config_path_sub}")
            except Exception as e:
                log.warning(f"Failed to save config to {save_config_path_sub}: {e}")

        if config.wandb_project is not None and is_main_process:
            with wandb.init(
                entity=config.wandb_entity,
                project=config.wandb_project,
                tags=config.wandb_tags,
                name=name,
                group=group,
                config=function_kwargs,
            ):  # type: ignore
                function_map[config.function_name](*function_args, **function_kwargs)

        else:
            function_map[config.function_name](*function_args, **function_kwargs)

        if has_jax:
            jax.clear_caches()

        if has_torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    finally:
        if file_handler is not None:
            logging.getLogger().removeHandler(file_handler)


def combine_root_tgt_patch(
    tgt: ConfigInputMultiple,
    common_root: Optional[ConfigInputMultiple] = None,
    common_patch: Optional[ConfigInputMultiple] = None,
) -> list[ConfigInput]:
    """
    Combines together the target(s) with any common configs
    Options:
    - common_root: if str/dict or list of str/dict then these are prefixed to the start of tgt
    - common_patch: if str/dict or list of str/dict then these are appended to the end of tgt
    """

    if isinstance(tgt, (str, dict)):
        tgt = [tgt]

    assert isinstance(tgt, list)

    if common_root is not None:
        if isinstance(common_root, (str, dict)):
            common_root = [common_root]
        tgt = common_root + tgt

    if common_patch is not None:
        if isinstance(common_patch, (str, dict)):
            common_patch = [common_patch]
        tgt = tgt + common_patch

    return tgt


def prepend_folder(
    config_stems: list[ConfigInput],
    folder: Optional[str] = None,
) -> list[ConfigInput]:
    """Prepends folder to each config stem in config_stems (only for str paths, not dicts)."""
    log.debug(f"Adding folder {folder} to config_stems {config_stems}")
    config_paths = [path.join(folder, t) if isinstance(t, str) else t for t in config_stems]
    return config_paths


def process_product_experiment_spec(
    product_experiment_specs: ProductExperimentSpec,
    folder: Optional[str] = None,
    common_root: Optional[ConfigInputMultiple] = None,
    common_patch: Optional[ConfigInputMultiple] = None,
    bonus_dict: StringKeyDict = {},
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
        for repeat_n in range(repeats):
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
                    "wandb_tags": lambda x, y: x + y,  # list concatenation
                },
                bonus_dict=bonus_dict,
            )
            if repeats > 1:
                cfg.name = f"{cfg.name}_r{repeat_n}"
            configs.append(cfg)

    return configs


def process_meta_config(mc: MetaConfig) -> list[Config]:
    """
    Generates a list of configs from the fields of a single meta config.
    Note: Sweep experiments are NOT processed here - only regular configs.
    """
    configs = []

    for exp_set_specs in mc.experiments:
        # Skip sweep experiments - they're handled separately
        if isinstance(exp_set_specs, SweepExperimentSpec):
            continue
        
        configs.extend(
            process_product_experiment_spec(
                exp_set_specs,
                folder=mc.folder,
                common_root=mc.common_root,
                common_patch=mc.common_patch,
                bonus_dict=mc.bonus_dict,
            )
        )

    if mc.rng_seed_offset != 0 or mc.auto_increment_rng_seed:
        for i, config in enumerate(configs):
            config.function_kwargs["rng_seed"] = (
                mc.rng_seed_offset + config.function_kwargs.get("rng_seed", 0)
            )
            if mc.auto_increment_rng_seed:
                config.function_kwargs["rng_seed"] += i

    names = [config.name for config in configs]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        log.warning(
            f"Meta config produces duplicate config names: {sorted(duplicates)} — "
            "runs may overwrite each other's RUN_NAME-derived outputs"
        )

    return configs


def process_sweep_experiment_spec(
    sweep_spec: SweepExperimentSpec,
    folder: Optional[str] = None,
    common_root: Optional[ConfigInputMultiple] = None,
    common_patch: Optional[ConfigInputMultiple] = None,
    bonus_dict: StringKeyDict = {},
) -> StringKeyDict:
    """
    Process a sweep experiment spec from meta-config, applying composition rules.
    Returns a sweep dict ready for execute_sweep_from_dict().
    """
    # Build full path to sweep config
    sweep_config_path = sweep_spec.sweep_config
    if folder and isinstance(sweep_config_path, str):
        sweep_config_path = path.join(folder, sweep_config_path)
    
    # Load sweep config
    sweep_dict = load_config_dict(sweep_config_path)
    
    # Handle base_config with composition
    base_config_path = sweep_spec.base_config or sweep_dict.get("base_config")
    
    if base_config_path is not None:
        # Note: folder is NOT applied to base_config from sweep_dict
        # because base_config paths in sweep configs are already relative to execution location
        # Only apply folder if base_config comes from sweep_spec (meta-config override)
        if sweep_spec.base_config is not None and folder and not path.isabs(base_config_path):
            base_config_path = path.join(folder, base_config_path)
        
        # Apply common_root, common_patch, and bonus_dict
        base_paths = combine_root_tgt_patch(base_config_path, common_root, common_patch)
        if bonus_dict:
            base_paths = base_paths + [bonus_dict]

        if len(base_paths) > 1:
            # Store as list - execute_sweep_from_dict will handle composition
            sweep_dict["base_config_paths"] = base_paths
        else:
            sweep_dict["base_config"] = base_config_path

    # Override sweep_count if specified in meta-config
    if sweep_spec.sweep_count is not None:
        sweep_dict["sweep_count"] = sweep_spec.sweep_count

    return sweep_dict


def remote_execute_sweep_from_dict(
    instance: InstanceConfig,
    function_map: FunctionMap,
    sweep_dict: StringKeyDict,
) -> None:
    """Execute a wandb sweep on a remote instance."""

    # Get sweep name for logging
    sweep_name = sweep_dict.get("sweep_name", "wandb_sweep")

    # Build resolved_names for sweep placeholder substitution
    resolved_names = resolve_run_names(
        name=sweep_name,
        time_stamp_name=False,
        sweep_name=sweep_name,
    )

    # Execute the sweep remotely
    execute_sweep_remotely(
        instance_config=instance,
        sweep_dict=sweep_dict,
        sweep_name=sweep_name,
        resolved_names=resolved_names,
    )

def execute_sweep_from_dict(
    function_map: FunctionMap,
    sweep_dict: StringKeyDict,
) -> None:
    """Execute a wandb sweep from sweep config dictionary."""
    
    # Extract custom fields
    instance = sweep_dict.pop("instance", None)
    if instance is not None:
        # Convert instance dict to InstanceConfig object if needed
        if isinstance(instance, dict):
            instance = InstanceConfig(**instance)
        sweep_dict["instance"] = None
        remote_execute_sweep_from_dict(instance, function_map, sweep_dict)
        return
    base_config_path = sweep_dict.pop("base_config", None)
    base_config_paths = sweep_dict.pop("base_config_paths", None)
    sweep_count = sweep_dict.pop("sweep_count", None)
    sweep_name = sweep_dict.pop("sweep_name", None)
    function_name = sweep_dict.pop("function_name", None)
    
    # Load base config if specified, otherwise create minimal config
    if base_config_paths is not None:
        # Compose multiple configs (from common_root/patch)
        log.info(f"Composing base config from {len(base_config_paths)} paths")
        base_config = load_and_compose_config_steps(base_config_paths)
    
    elif base_config_path is not None:
        log.info(f"Loading base config from {base_config_path if isinstance(base_config_path, str) else 'inline dict'}")
        base_config = load_config(resolve_config_dict(base_config_path))
    
    else:
        log.info("No base_config specified, using minimal config")
        # Create a minimal config - user must specify function_name in sweep or base
        base_config = Config(
            name="sweep_run",
            function_name=function_name or "",
        )
    
    # Extract wandb project/entity from sweep config or base config
    wandb_project = sweep_dict.pop("project", base_config.wandb_project)
    wandb_entity = sweep_dict.pop("entity", base_config.wandb_entity)
    
    if wandb_project is None:
        raise ValueError("wandb project must be specified in sweep config or base config")
    
    log.info(f"Creating wandb sweep in {wandb_entity}/{wandb_project}")
    log.info("========== Sweep Config ===========\n" + pformat(sweep_dict))
    
    # Initialize wandb sweep
    sweep_id = wandb.sweep(
        sweep=sweep_dict,
        project=wandb_project,
        entity=wandb_entity,
    )
    
    log.info(f"Created sweep with {sweep_id=}")
    
    # Use sweep_name if specified, otherwise use sweep_id
    if sweep_name is None:
        sweep_name = sweep_id
    
    log.info(f"Using {sweep_name=} for SWEEP_NAME substitution")
    
    # Define the train function that wandb.agent will call
    def train_function():
        # Initialize wandb run - this must be called first before accessing wandb.config or wandb.run
        with wandb.init(
            entity=wandb_entity,
            project=wandb_project,
            group=base_config.wandb_group,
            tags=base_config.wandb_tags,
        ):
            # wandb.config contains the sweep parameters (now available after init)
            sweep_params = dict(wandb.config)
            
            # Get wandb's auto-generated run name for RUN_NAME substitution
            wandb_run_name = wandb.run.name
            
            log.info("========== Sweep Run ===========")
            log.info(f"Wandb run name: {wandb_run_name}")
            log.info(f"Sweep params: {pformat(sweep_params)}")
            
            # Merge sweep params with base config's kwargs (sweep params override)
            # Use recursive merge to handle nested parameters properly.
            # No type check: sweep params may legitimately differ in type from base values.
            merged_kwargs = recursive_dict_update(
                base_config.function_kwargs or {},
                sweep_params,
                assert_type_match=False,
            )
            
            # Execute using execute_from_config with wandb disabled (already initialized above)
            # This gives us RUN_NAME, RUN_GROUP, SWEEP_NAME substitution + log file handling
            run_config = replace(
                base_config,
                function_kwargs=merged_kwargs,
                name=wandb_run_name,  # Use wandb's auto-generated name
                time_stamp_name=False,  # Already has timestamp from wandb
                time_stamp_group=False,
                wandb_project=None,  # Skip wandb.init (already initialized above)
            )
            execute_from_config(
                config=run_config,
                function_map=function_map,
                sweep_name=sweep_name,  # For SWEEP_NAME substitution
            )
        
        # Clean up GPU memory after wandb context exits to release any wandb-held references
        if has_torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            log.debug("Cleared CUDA cache after sweep run")
    
    # Run the sweep agent
    log.info(f"Starting sweep agent{f' for {sweep_count} runs' if sweep_count else ''}")
    wandb.agent(sweep_id, function=train_function, count=sweep_count, project=wandb_project, entity=wandb_entity)
    
    log.info("Sweep completed")


def execute_sweep(
    function_map: FunctionMap,
    sweep_config_path: ConfigInput,
) -> None:
    """Execute a wandb sweep from sweep config file or inline dict."""
    
    git_commit_hash = get_git_commit_hash()
    log.info(f"Executing sweep, {git_commit_hash=}")
    
    # Load sweep config
    log.info("Loading sweep config...")
    sweep_dict = load_config_dict(sweep_config_path)
    
    # Execute using the dict version
    execute_sweep_from_dict(function_map, sweep_dict)


def build_configs(
    config_path: Optional[ConfigInput] = None,
    meta_config_path: Optional[ConfigInput] = None,
    sweep_config_path: Optional[ConfigInput] = None,
) -> ResolvedExperiments:
    """
    Resolve config paths into the fully composed configs and sweep dicts they produce,
    without executing anything. This is the single resolution path used by both
    execute_experiments and dry runs, so a dry run cannot disagree with a real run.
    """
    if sweep_config_path is not None:
        log.info("Loading sweep config...")
        return ResolvedExperiments([], [load_config_dict(sweep_config_path)])

    if config_path is not None:
        log.info("Loading config...")
        config_type = detect_config_type(load_config_dict(config_path))
        log.info(f"Detected config type: {config_type}")

        if config_type == "meta":
            meta_config_path = config_path  # Reuse meta config logic below
        elif config_type == "sweep":
            return ResolvedExperiments([], [load_config_dict(config_path)])
        else:
            return ResolvedExperiments([load_config(config_path)], [])

    if meta_config_path is None:
        return ResolvedExperiments([], [])

    log.info("Loading meta config...")
    meta_config = load_meta_config(meta_config_path)
    log.info("========== Meta Config ===========\n" + pformat(meta_config))

    # Sweep experiments get the same composition rules as regular experiments
    sweep_dicts = [
        process_sweep_experiment_spec(
            exp,
            folder=meta_config.folder,
            common_root=meta_config.common_root,
            common_patch=meta_config.common_patch,
            bonus_dict=meta_config.bonus_dict,
        )
        for exp in meta_config.experiments
        if isinstance(exp, SweepExperimentSpec)
    ]
    return ResolvedExperiments(
        configs=process_meta_config(meta_config),
        sweep_dicts=sweep_dicts,
        parallel=meta_config.parallel,
        start_method=meta_config.start_method,
        max_concurrent=meta_config.max_concurrent,
    )


def _drop_unset(config_dict: StringKeyDict) -> StringKeyDict:
    """Drop top-level keys left as None, which are unset config fields rather than meaningful values."""
    return {k: v for k, v in config_dict.items() if v is not None}


def resolve_config_for_display(
    config: Config,
    launch_time_stamp: Optional[str] = None,
) -> StringKeyDict:
    """Return a config as a plain dict with RUN_NAME/RUN_GROUP resolved as they would be at run time."""
    resolved_names = resolve_run_names(
        name=config.name,
        time_stamp_name=config.time_stamp_name,
        time_stamp_group=config.time_stamp_group,
        wandb_group=config.wandb_group,
        launch_time_stamp=launch_time_stamp,
    )
    config_dict = asdict(config)
    config_dict["name"] = resolved_names["name"]
    config_dict["wandb_group"] = resolved_names["group"]
    del config_dict["time_stamp_name"], config_dict["time_stamp_group"]
    if config_dict["instance"] is not None:
        config_dict["instance"] = _drop_unset(config_dict["instance"])
    return _drop_unset(substitute_placeholders(config_dict, resolved_names))


def resolve_sweep_for_display(sweep_dict: StringKeyDict) -> StringKeyDict:
    """Return a sweep dict with its base config composed in and SWEEP_NAME resolved."""
    sweep_dict = dict(sweep_dict)
    base_config_paths = sweep_dict.pop("base_config_paths", None)
    base_config_path = sweep_dict.pop("base_config", None)

    if base_config_paths is not None:
        base_config = load_and_compose_config_steps(base_config_paths)
    elif base_config_path is not None:
        base_config = load_config(resolve_config_dict(base_config_path))
    else:
        base_config = None

    # RUN_NAME comes from wandb at run time, so it stays unresolved; substituting it with
    # itself is a no-op that leaves the placeholder visible.
    resolved_names = {
        "name": "RUN_NAME",
        "group": (base_config.wandb_group if base_config else None) or "RUN_GROUP",
        "sweep_name": sweep_dict.get("sweep_name") or "SWEEP_NAME",
    }
    display = {"sweep": substitute_placeholders(sweep_dict, resolved_names)}
    if base_config is not None:
        display["base_config"] = _drop_unset(
            substitute_placeholders(asdict(base_config), resolved_names)
        )
    return display


def check_function_names(configs: list[Config], function_map: FunctionMap) -> None:
    """Warn about function_names that are not in the function map, which would fail at run time."""
    missing = {c.function_name for c in configs} - set(function_map)
    if missing:
        log.warning(
            f"function_name(s) not in function_map: {sorted(missing)} — "
            f"these would fail at run time. Available: {sorted(function_map)}"
        )


def dry_run_report(
    resolved: ResolvedExperiments,
    launch_time_stamp: Optional[str] = None,
) -> str:
    """
    Render resolved configs as a multi-document YAML report.
    Output parses with yaml.safe_load_all, so it can be diffed or snapshot-tested.
    """
    configs, sweep_dicts = resolved.configs, resolved.sweep_dicts

    def _document(header: str, d: StringKeyDict) -> str:
        body = yaml.dump(d, sort_keys=False, default_flow_style=False).rstrip()
        return f"--- # {header}\n{body}"

    documents = [
        _document(f"Config {i+1}/{len(configs)}", resolve_config_for_display(c, launch_time_stamp))
        for i, c in enumerate(configs)
    ] + [
        _document(f"Sweep {i+1}/{len(sweep_dicts)}", resolve_sweep_for_display(s))
        for i, s in enumerate(sweep_dicts)
    ]
    header = f"# Dry run: {len(configs)} config(s), {len(sweep_dicts)} sweep(s)"
    return "\n".join([header] + documents)


def _parallel_config_worker(index, config, function_map, launch_time_stamp, error_queue):
    """Worker for parallel config execution. Runs in a subprocess."""
    try:
        execute_from_config(config, function_map=function_map, launch_time_stamp=launch_time_stamp)
    except Exception as e:
        error_queue.put((index, f"{type(e).__name__}: {e}"))


def execute_experiments(
    function_map: FunctionMap,
    config_path: Optional[str] = None,
    meta_config_path: Optional[str] = None,
    sweep_config_path: Optional[str] = None,
    dry_run: bool = False,
) -> None:
    """Creates a sequence of configs from config_path or meta_config_path and executes them"""

    git_commit_hash = get_git_commit_hash()
    log.info(f"Executing experiment, {git_commit_hash=}")

    # Capture a single timestamp at launch so all configs in a batch share it
    launch_time_stamp = get_time_stamp(include_seconds=True)

    # Check that only one execution mode is specified
    specified_modes = sum([
        config_path is not None,
        meta_config_path is not None,
        sweep_config_path is not None,
    ])

    if specified_modes > 1:
        raise ValueError(
            "Only one of config_path, meta_config_path, or sweep_config_path can be specified"
        )

    if specified_modes == 0:
        log.warning("Please use -c, -m, or -s to specify a config, meta config, or sweep config to run!")
        return

    resolved = build_configs(config_path, meta_config_path, sweep_config_path)
    configs, sweep_dicts = resolved.configs, resolved.sweep_dicts

    if dry_run:
        check_function_names(configs, function_map)
        print(dry_run_report(resolved, launch_time_stamp))
        return

    if not configs and not sweep_dicts:
        log.warning("Config produced nothing to execute!")
        return

    # Check if parallel execution applies
    has_remote = any(c.instance is not None for c in configs)
    use_parallel = resolved.parallel and len(configs) > 1 and not has_remote

    if resolved.parallel and has_remote:
        log.warning(
            "parallel=True is ignored when remote configs are present — use managed: true for parallel remote execution"
        )

    # Execute configs
    if use_parallel:
        # fork is unsafe once a child touches Metal (macOS) and unavailable on Windows
        start_method = resolved.start_method or ("fork" if sys.platform == "linux" else "spawn")
        limit = resolved.max_concurrent if resolved.max_concurrent is not None else len(configs)
        log.info(f"Executing {len(configs)} configs in parallel ({start_method}, {limit} at a time)")
        ctx = mp.get_context(start_method)
        error_queue = ctx.Queue()
        processes = []
        for i, config in enumerate(configs):
            while sum(p.is_alive() for p in processes) >= limit:
                time.sleep(0.1)
            p = ctx.Process(
                target=_parallel_config_worker,
                args=(i, config, function_map, launch_time_stamp, error_queue),
            )
            processes.append(p)
            p.start()
        for i, p in enumerate(processes):
            p.join()
            log.info(f"Config {i+1}/{len(configs)} {'completed' if p.exitcode == 0 else 'failed'}")
        # Hard crashes (OOM kill, segfault) leave the queue empty but exit nonzero
        failures = {i: f"exit code {p.exitcode}" for i, p in enumerate(processes) if p.exitcode != 0}
        while not error_queue.empty():
            idx, msg = error_queue.get_nowait()
            failures[idx] = msg
        if failures:
            idx = min(failures)
            raise RuntimeError(f"Config {idx+1}/{len(configs)} failed: {failures[idx]}")
    else:
        for i, config in enumerate(configs):
            log.info(f"Executing config {i+1}/{len(configs)}")
            execute_from_config(config, function_map=function_map, launch_time_stamp=launch_time_stamp)

    # Execute sweeps (always sequential)
    for i, sweep_dict in enumerate(sweep_dicts):
        log.info(f"Executing sweep {i+1}/{len(sweep_dicts)}")
        execute_sweep_from_dict(function_map, sweep_dict)
