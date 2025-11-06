"""Test basic config loading functionality"""

import os
from research_scaffold.config_tools import (
    load_config,
    load_and_compose_config_steps,
)


def test_load_simple_config(example_dir):
    config = load_config(str(example_dir / "configs/simple_config.yaml"))
    
    assert config.name == "simple_config_test"
    assert config.function_name == "example_simple_config"
    assert config.function_kwargs["dummy_str"] == "foo"
    assert config.function_kwargs["dummy_int"] == 4
    assert config.function_kwargs["dummy_bool"] is True


def test_load_config_with_wandb(example_dir):
    config = load_config(str(example_dir / "configs/wandb_and_tags.yaml"))
    
    assert config.wandb_project == "research_scaffold_test"
    assert "tag1" in config.wandb_tags


def test_compose_config_steps(example_dir):
    config = load_and_compose_config_steps([
        str(example_dir / "configs/lennie/root.yaml"),
        str(example_dir / "configs/lennie/level1a.yaml"),
    ])
    
    # Check merging happened
    assert "root_arg" in config.function_kwargs
    assert "arg1" in config.function_kwargs

