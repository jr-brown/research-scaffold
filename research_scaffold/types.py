"""
Type definitions for research-scaffold.
"""

from typing import Any, Optional, Union
from collections.abc import Callable
from dataclasses import dataclass


### Type definitions
StringKeyDict = dict[str, Any]
FunctionMap = dict[str, Callable]
ConfigInput = Union[str, StringKeyDict]  # Single config path or inline dict
ConfigInputMultiple = Union[ConfigInput, list[ConfigInput]]  # Single or list of configs
ConfigPathAxes = list[ConfigInputMultiple]


@dataclass
class InstanceConfig:
    """Type definition for remote instance configuration."""
    
    sky_config: Optional[str] = None  # Path to SkyPilot YAML config file (or use SKY_PATH env var)
    patch: Optional[Union[StringKeyDict, str]] = None  # Inline dict or path to patch YAML
    commit: Optional[list[str]] = None  # List of paths/patterns to commit and push (e.g., ["outputs/**", "logs/**"])


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
    save_config_path: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_tags: Optional[list[str]] = None
    instance: Optional["InstanceConfig"] = None

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
    expt_root: Optional[ConfigInputMultiple]
    expt_patch: Optional[ConfigInputMultiple]


@dataclass
class SweepExperimentSpec:
    """Type definition for a SweepExperimentSpec."""

    sweep_config: ConfigInput
    base_config: Optional[ConfigInput]
    sweep_count: Optional[int]


@dataclass
class SweepConfig:
    """Type definition for a standalone Sweep Config."""
    
    method: str
    parameters: dict
    metric: Optional[dict] = None
    base_config: Optional[ConfigInput] = None
    base_config_paths: Optional[list[ConfigInput]] = None
    sweep_count: Optional[int] = None
    sweep_name: Optional[str] = None
    project: Optional[str] = None
    entity: Optional[str] = None
    instance: Optional["InstanceConfig"] = None


ExperimentSpec = Union[ProductExperimentSpec, SweepExperimentSpec]


@dataclass
class MetaConfig:
    """Type definition for a MetaConfig."""

    experiments: list[ExperimentSpec]
    folder: Optional[str]
    common_root: Optional[ConfigInputMultiple]
    common_patch: Optional[ConfigInputMultiple]
    auto_increment_rng_seed: bool
    rng_seed_offset: int
    bonus_dict: Optional[StringKeyDict]

    @property
    def d(self):
        """Simple shorthand for self.__dict__"""
        return self.__dict__

