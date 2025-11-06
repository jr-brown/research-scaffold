"""Test experiment execution"""

from research_scaffold.config_tools import execute_experiments, execute_from_config, load_config
from unittest.mock import Mock


def test_execute_single_config(example_dir, mock_git):
    call_tracker = []
    
    def test_fn(**kwargs):
        call_tracker.append(kwargs)
    
    function_map = {"example_simple_config": test_fn}
    
    execute_experiments(
        function_map=function_map,
        config_path=str(example_dir / "configs/simple_config.yaml"),
    )
    
    assert len(call_tracker) == 1
    assert call_tracker[0]["dummy_str"] == "foo"
    assert call_tracker[0]["dummy_int"] == 4


def test_execute_from_config_calls_function(mock_git):
    from research_scaffold.config_tools import Config
    
    call_tracker = []
    
    def my_func(a, b):
        call_tracker.append({"a": a, "b": b})
    
    config = Config(
        name="test",
        function_name="my_func",
        function_kwargs={"a": 1, "b": 2},
    )
    
    execute_from_config(
        config=config,
        function_map={"my_func": my_func},
        **config.d
    )
    
    assert len(call_tracker) == 1
    assert call_tracker[0]["a"] == 1
    assert call_tracker[0]["b"] == 2


def test_execute_with_wandb(example_dir, mock_git, mock_wandb):
    call_tracker = []
    
    def test_fn(**kwargs):
        call_tracker.append(kwargs)
    
    function_map = {"example_simple_config": test_fn}
    
    execute_experiments(
        function_map=function_map,
        config_path=str(example_dir / "configs/wandb_and_tags.yaml"),
    )
    
    # Verify wandb.init was called
    assert mock_wandb.init.called
    
    # Verify function was called
    assert len(call_tracker) == 1

