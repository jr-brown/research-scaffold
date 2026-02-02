"""
Tools for loading and executing experiments from config files.
"""

# Standard Library
import logging
import subprocess

from dataclasses import replace
from os import path, makedirs
from pprint import pformat
from typing import Optional
from collections.abc import Callable
from functools import partial

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
)
from .util import (
    is_main_process,
    get_logger,
    nones_to_empty_lists,
    nones_to_empty_dicts,
    get_time_stamp,
    recursive_dict_update,
    check_name_sub_general,
    load_config_dict,
)
from .file_io import load, save

from .remote_execution import execute_config_remotely, execute_sweep_remotely


log = get_logger(__name__)


### Functions

def detect_config_type(config_dict: StringKeyDict) -> str:
    """
    Detect config type by trying to instantiate each dataclass.
    Returns "single", "meta", or "sweep".
    """
    # Try MetaConfig
    try:
        load_meta_config(config_dict)
        return "meta"
    except Exception:
        pass
    
    # Try SweepConfig  
    try:
        load_sweep_config(config_dict)
        return "sweep"
    except Exception:
        pass
    
    # Try Config
    try:
        _ = load_config(config_dict)
        return "single"
    except Exception:
        pass
    
    raise ValueError(
        f"Could not identify config type. Got keys: {list(config_dict.keys())}"
    )


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
    config_dict = load_config_dict(config_path_or_dict)
    if 'instance' in config_dict and isinstance(config_dict['instance'], dict):
        config_dict['instance'] = InstanceConfig(**config_dict['instance'])
    return Config(**config_dict)


def load_and_compose_config_steps(
    cfg_paths: list[ConfigInput],
    compositions: Optional[dict[str, Callable]] = None,
    bonus_dict: dict = {},
) -> Config:
    """Return single config from iteratively combining configs loaded from cfg_paths (paths or inline dicts)."""
    config_dict = {}

    for cfg_path in cfg_paths:
        partial_config_dict = load_config_dict(cfg_path)
        config_dict = recursive_dict_update(
            config_dict, partial_config_dict, compositions=compositions
        )

    config_dict = recursive_dict_update(
        config_dict, bonus_dict, compositions=compositions
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
        assert set_specific_dict.keys() <= set(allowed_keys), \
            f"Sweep experiment has invalid keys: {set_specific_dict.keys() - set(allowed_keys)}"
        
        return SweepExperimentSpec(
            sweep_config=set_specific_dict["sweep_config"],
            base_config=set_specific_dict.get("base_config", None),
            sweep_count=set_specific_dict.get("sweep_count", None),
        )
    
    # Otherwise, it's a regular product experiment
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


def load_meta_config(meta_cfg_path: ConfigInput) -> MetaConfig:
    """Loads a meta config from path or inline dict (including .yaml extension if path)."""
    mc_dict = load_config_dict(meta_cfg_path)
    experiments = [parse_experiment_set(specs) for specs in mc_dict["experiments"]]
    return MetaConfig(
        experiments=experiments,
        bonus_dict=mc_dict.get("bonus_dict", {}),
        common_root=mc_dict.get("common_root", None),
        common_patch=mc_dict.get("common_patch", None),
        auto_increment_rng_seed=mc_dict.get("auto_increment_rng_seed", False),
        rng_seed_offset=mc_dict.get("rng_seed_offset", 0),
        folder=mc_dict.get("folder", ""),
    )


def load_sweep_config(sweep_cfg_path: ConfigInput) -> SweepConfig:
    """Loads a sweep config from path or inline dict (including .yaml extension if path)."""
    sc_dict = load_config_dict(sweep_cfg_path)
    if 'instance' in sc_dict and isinstance(sc_dict['instance'], dict):
        sc_dict['instance'] = InstanceConfig(**sc_dict['instance'])
    return SweepConfig(**sc_dict)


def remote_execute_from_config(
    instance: InstanceConfig,
    config: Config,
):
    """Execute a function on a remote instance."""
    
    execute_config_remotely(instance, config)

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
    save_config_path: Optional[str] = None,
    run_name_dummy: str = "RUN_NAME",
    run_group_dummy: str = "RUN_GROUP",
    sweep_name: Optional[str] = None,
    sweep_name_dummy: str = "SWEEP_NAME",
    instance: Optional[InstanceConfig] = None,
):
    """
    Executes a function from a Config object.
    """

    if instance is not None:
        config.instance = None
        remote_execute_from_config(instance, config)
        return

    name_base = name
    group = wandb_group if wandb_group is not None else name_base

    if time_stamp_name:
        name = f"{name}_{get_time_stamp(include_seconds=True)}"

    check_name_sub = partial(
        check_name_sub_general, new_name=name, run_name_dummy=run_name_dummy
    )
    check_group_sub = partial(
        check_name_sub_general, new_name=group, run_name_dummy=run_group_dummy
    )
    check_sweep_name_sub = partial(
        check_name_sub_general, new_name=sweep_name or "", run_name_dummy=sweep_name_dummy
    ) if sweep_name else None

    # Add handler to log to file if necessary
    if log_file_path is not None:
        log_file_path, n_name_subs = check_name_sub(log_file_path, count=0)
        log_file_path, n_group_subs = check_group_sub(log_file_path, count=0)
        if check_sweep_name_sub:
            log_file_path, n_sweep_subs = check_sweep_name_sub(log_file_path, count=0)
        else:
            n_sweep_subs = 0

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
        n_name_subs = 0
        n_group_subs = 0
        n_sweep_subs = 0
        file_log_cleanup_fn = None

    log.info("========== Config Dict ===========\n" + pformat(config))
    log.info("Run Name: " + pformat(name))

    (function_args,) = nones_to_empty_lists(function_args)
    (function_kwargs,) = nones_to_empty_dicts(function_kwargs)

    # Substitute occurrences of RUN_NAME, RUN_GROUP, and SWEEP_NAME
    function_args, n_name_subs = check_name_sub(function_args, count=n_name_subs)
    function_args, n_group_subs = check_group_sub(function_args, count=n_group_subs)
    function_kwargs, n_name_subs = check_name_sub(function_kwargs, count=n_name_subs)
    function_kwargs, n_group_subs = check_group_sub(function_kwargs, count=n_group_subs)
    
    if check_sweep_name_sub:
        function_args, n_sweep_subs = check_sweep_name_sub(function_args, count=n_sweep_subs)
        function_kwargs, n_sweep_subs = check_sweep_name_sub(function_kwargs, count=n_sweep_subs)

    # Save config to file if requested
    if save_config_path is not None:
        save_config_path_sub, _ = check_name_sub(save_config_path, count=0)
        save_config_path_sub, _ = check_group_sub(save_config_path_sub, count=0)
        if check_sweep_name_sub:
            save_config_path_sub, _ = check_sweep_name_sub(save_config_path_sub, count=0)
        
        # Create a dict with the full config including substituted values
        config_to_save = {
            "name": name,
            "function_name": function_name,
            "function_args": function_args,
            "function_kwargs": function_kwargs,
            "wandb_project": wandb_project,
            "wandb_entity": wandb_entity,
            "wandb_group": wandb_group,
            "wandb_tags": wandb_tags,
            "log_file_path": log_file_path,
            "save_config_path": save_config_path_sub,
        }
        
        try:
            save("yaml", config_to_save, save_config_path_sub, overwrite=True)
            log.info(f"Saved config to: {save_config_path_sub}")
        except Exception as e:
            log.warning(f"Failed to save config to {save_config_path_sub}: {e}")

    if wandb_project is not None and is_main_process:
        with wandb.init(
            entity=wandb_entity,
            project=wandb_project,
            tags=wandb_tags,
            name=name,
            group=group,
            config=function_kwargs,
        ):  # type: ignore
            function_map[function_name](*function_args, **function_kwargs)

    else:
        function_map[function_name](*function_args, **function_kwargs)

    if has_jax:
        jax.clear_caches()

    if has_torch and torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    if file_log_cleanup_fn is not None:
        file_log_cleanup_fn()


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
                bonus_dict=bonus_dict,
            )
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
        
        # Apply common_root and common_patch
        if common_root or common_patch:
            # Combine root, target, and patch into list of paths
            base_paths = combine_root_tgt_patch(base_config_path, common_root, common_patch)
            
            # Store as list - execute_sweep_from_dict will handle composition
            sweep_dict["base_config_paths"] = base_paths
        else:
            sweep_dict["base_config"] = base_config_path
    
    # Override sweep_count if specified in meta-config
    if sweep_spec.sweep_count is not None:
        sweep_dict["sweep_count"] = sweep_spec.sweep_count
    
    # Note: bonus_dict could be merged into base_config here if needed
    # For now, we don't apply bonus_dict to sweeps
    
    return sweep_dict


def remote_execute_sweep_from_dict(
    instance: InstanceConfig,
    function_map: FunctionMap,
    sweep_dict: StringKeyDict,
) -> None:
    """Execute a wandb sweep on a remote instance."""
    
    # Get sweep name for logging
    sweep_name = sweep_dict.get("sweep_name", "wandb_sweep")
    
    # Execute the sweep remotely
    execute_sweep_remotely(
        instance_config=instance,
        sweep_dict=sweep_dict,
        sweep_name=sweep_name,
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
    
    # Load base config if specified, otherwise create minimal config
    if base_config_paths is not None:
        # Compose multiple configs (from common_root/patch)
        log.info(f"Composing base config from {len(base_config_paths)} paths")
        base_config = load_and_compose_config_steps(base_config_paths)
    
    elif base_config_path is not None:
        log.info(f"Loading base config from {base_config_path if isinstance(base_config_path, str) else 'inline dict'}")
        
        # Check if it's a meta-config
        base_dict = load_config_dict(base_config_path)
        is_meta = "experiments" in base_dict
        
        if is_meta:
            log.info("Base config is a meta-config, processing...")
            meta_config = load_meta_config(base_config_path)
            base_configs = process_meta_config(meta_config)
            
            # Validate: meta-config must produce exactly ONE config
            if len(base_configs) != 1:
                raise ValueError(
                    f"base_config meta-config must produce exactly 1 config, got {len(base_configs)}. "
                    f"Meta-configs with config_axes, repeats>1, or multiple experiments cannot be used as sweep base."
                )
            
            base_config = base_configs[0]
            log.info(f"Meta-config produced single composed config")
        else:
            base_config = load_config(base_config_path)
    
    else:
        log.info("No base_config specified, using minimal config")
        # Create a minimal config - user must specify function_name in sweep or base
        base_config = Config(
            name="sweep_run",
            function_name=sweep_dict.get("function_name", ""),
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
            # Use recursive merge to handle nested parameters properly
            merged_kwargs = recursive_dict_update(
                base_config.function_kwargs or {},
                sweep_params
            )
            
            # Create a copy of base_config with the merged kwargs so logging shows actual values
            run_config = replace(base_config, function_kwargs=merged_kwargs)
            
            # Execute using execute_from_config with wandb disabled (already initialized above)
            # This gives us RUN_NAME, RUN_GROUP, SWEEP_NAME substitution + log file handling
            execute_from_config(
                config=run_config,
                function_map=function_map,
                function_name=run_config.function_name,
                function_args=run_config.function_args,
                function_kwargs=merged_kwargs,
                name=wandb_run_name,  # Use wandb's auto-generated name
                time_stamp_name=False,  # Already has timestamp from wandb
                wandb_project=None,  # Skip wandb.init (already initialized above)
                wandb_group=base_config.wandb_group,
                wandb_entity=base_config.wandb_entity,
                wandb_tags=base_config.wandb_tags,
                log_file_path=base_config.log_file_path,
                save_config_path=base_config.save_config_path,
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


def execute_experiments(
    function_map: FunctionMap,
    config_path: Optional[str] = None,
    meta_config_path: Optional[str] = None,
    sweep_config_path: Optional[str] = None,
) -> None:
    """Creates a sequence of configs from config_path or meta_config_path and executes them"""

    git_commit_hash = get_git_commit_hash()
    log.info(f"Executing experiment, {git_commit_hash=}")

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
    
    if sweep_config_path is not None:
        execute_sweep(function_map, sweep_config_path)
        return

    configs = []
    
    if config_path is not None:
        log.info("Loading config...")
        
        # Auto-detect config type
        config_dict = load_config_dict(config_path)
        config_type = detect_config_type(config_dict)
        log.info(f"Detected config type: {config_type}")
        
        # Delegate to appropriate handler
        if config_type == "meta":
            meta_config_path = config_path  # Reuse existing meta config logic below
        elif config_type == "sweep":
            execute_sweep(function_map, config_path)
            return
        else:  # single
            configs = [load_config(config_path)]

    if meta_config_path is not None:
        log.info("Loading meta config...")
        meta_config = load_meta_config(meta_config_path)
        log.info("========== Meta Config ===========\n" + pformat(meta_config))
        
        # Separate sweep experiments from regular experiments
        sweep_experiments = [exp for exp in meta_config.experiments if isinstance(exp, SweepExperimentSpec)]
        
        # Process regular configs (skips sweeps)
        configs = process_meta_config(meta_config)
        
        # Execute regular configs
        for i, config in enumerate(configs):
            log.info(f"Executing config {i+1}/{len(configs)}")
            execute_from_config(config, function_map=function_map, **config.d)
        
        # Execute sweep experiments
        for i, sweep_exp in enumerate(sweep_experiments):
            log.info(f"Executing sweep {i+1}/{len(sweep_experiments)}")
            
            # Process sweep spec with same composition rules as regular experiments
            sweep_dict = process_sweep_experiment_spec(
                sweep_exp,
                folder=meta_config.folder,
                common_root=meta_config.common_root,
                common_patch=meta_config.common_patch,
                bonus_dict=meta_config.bonus_dict,
            )
            
            # Execute the sweep
            execute_sweep_from_dict(function_map, sweep_dict)
        
        return

    if not configs:
        log.warning("Please use -c, -m, or -s to specify a config, meta config, or sweep config to run!")
        return

    for i, config in enumerate(configs):
        log.info(f"Executing config {i+1}/{len(configs)}")
        execute_from_config(config, function_map=function_map, **config.d)
