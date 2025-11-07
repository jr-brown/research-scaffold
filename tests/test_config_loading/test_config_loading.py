"""Test basic config loading functionality"""

from pathlib import Path
from research_scaffold.config_tools import (
    load_config,
    load_and_compose_config_steps,
)


TEST_DIR = Path(__file__).parent


def test_load_simple_config():
    config = load_config(str(TEST_DIR / "configs/simple_config.yaml"))
    
    assert config.name == "simple_config_test"
    assert config.function_name == "example_simple_config"
    assert config.function_kwargs["dummy_str"] == "foo"
    assert config.function_kwargs["dummy_int"] == 4
    assert config.function_kwargs["dummy_bool"] is True


def test_load_config_with_wandb():
    config = load_config(str(TEST_DIR / "configs/wandb_and_tags.yaml"))
    
    assert config.wandb_project == "research_scaffold_test"
    assert "tag1" in config.wandb_tags


def test_compose_config_steps():
    config = load_and_compose_config_steps([
        str(TEST_DIR / "configs/lennie/root.yaml"),
        str(TEST_DIR / "configs/lennie/level1a.yaml"),
    ])
    
    # Check merging happened
    assert "root_arg" in config.function_kwargs
    assert "arg1" in config.function_kwargs

