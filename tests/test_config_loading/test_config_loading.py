"""Test basic config loading functionality"""

from pathlib import Path
from research_scaffold.config_tools import (
    load_config,
    load_and_compose_config_steps,
    load_meta_config,
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


def test_meta_config_base():
    mc = load_meta_config({
        "base": str(TEST_DIR / "configs/base_meta.yaml"),
        "folder": "debug_configs",
        "bonus_dict": {"function_kwargs": {"n_eval": 2}},
    })

    assert mc.folder == "debug_configs"
    assert mc.bonus_dict["function_kwargs"]["n_eval"] == 2
    assert mc.bonus_dict["function_kwargs"]["dataset"] == "full"
    assert len(mc.experiments) == 1



def test_replace_marker_discards_inherited_block():
    config = load_and_compose_config_steps([
        {
            "name": "base",
            "function_name": "f",
            "function_kwargs": {
                "keep_me": 1,
                "side_task": {
                    "type": "rolling_mod_sum",
                    "n_samples": 10000,
                    "modulus": 3,
                    "nested": {"a": 1},
                },
            },
        },
        {
            "function_kwargs": {
                "side_task": {
                    "<replace>": True,
                    "type": "knights_and_knaves",
                    "n_people": 3,
                },
            },
        },
    ])

    assert config.function_kwargs["side_task"] == {
        "type": "knights_and_knaves",
        "n_people": 3,
    }
    assert config.function_kwargs["keep_me"] == 1


def test_replace_marker_is_deep_and_stripped():
    config = load_and_compose_config_steps([
        {
            "name": "base",
            "function_name": "f",
            "function_kwargs": {"block": {"outer": 1, "inner": {"a": 1, "b": 2}}},
        },
        {
            "function_kwargs": {
                "block": {
                    "<replace>": True,
                    "inner": {"<replace>": True, "c": 3},
                },
            },
        },
    ])

    assert config.function_kwargs["block"] == {"inner": {"c": 3}}


def test_replace_marker_on_new_key_is_stripped():
    config = load_and_compose_config_steps([
        {"name": "base", "function_name": "f", "function_kwargs": {}},
        {"function_kwargs": {"fresh": {"<replace>": True, "a": 1}}},
    ])

    assert config.function_kwargs["fresh"] == {"a": 1}


def test_merging_is_still_the_default():
    config = load_and_compose_config_steps([
        {"name": "base", "function_name": "f", "function_kwargs": {"block": {"a": 1}}},
        {"function_kwargs": {"block": {"b": 2}}},
    ])

    assert config.function_kwargs["block"] == {"a": 1, "b": 2}
