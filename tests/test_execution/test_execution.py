"""Test experiment execution"""

import re
from pathlib import Path
from research_scaffold.config_tools import execute_experiments, execute_from_config, load_config
from research_scaffold.util import resolve_run_names
from unittest.mock import Mock


TEST_DIR = Path(__file__).parent


def test_execute_single_config(mock_git):
    call_tracker = []
    
    def test_fn(**kwargs):
        call_tracker.append(kwargs)
    
    function_map = {"example_simple_config": test_fn}
    
    execute_experiments(
        function_map=function_map,
        config_path=str(TEST_DIR / "configs/simple_config.yaml"),
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


def test_execute_with_wandb(mock_git, mock_wandb):
    call_tracker = []
    
    def test_fn(**kwargs):
        call_tracker.append(kwargs)
    
    function_map = {"example_simple_config": test_fn}
    
    execute_experiments(
        function_map=function_map,
        config_path=str(TEST_DIR / "configs/wandb_and_tags.yaml"),
    )
    
    # Verify wandb.init was called
    assert mock_wandb['init'].called
    
    # Verify function was called
    assert len(call_tracker) == 1


TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")


def test_resolve_run_names_time_stamp_group():
    result = resolve_run_names(name="exp", time_stamp_group=True, wandb_group="grp")
    assert result["group"].startswith("grp_")
    assert TIMESTAMP_RE.search(result["group"])
    assert result["name"] == "exp"


def test_resolve_run_names_time_stamp_group_defaults_to_name():
    result = resolve_run_names(name="exp", time_stamp_group=True)
    assert result["group"].startswith("exp_")
    assert TIMESTAMP_RE.search(result["group"])


def test_resolve_run_names_no_time_stamp_group():
    result = resolve_run_names(name="exp", wandb_group="grp")
    assert result["group"] == "grp"


def test_execute_with_time_stamp_group(mock_git):
    call_tracker = []

    def test_fn(**kwargs):
        call_tracker.append(kwargs)

    function_map = {"example_simple_config": test_fn}

    execute_experiments(
        function_map=function_map,
        config_path=str(TEST_DIR / "configs/timestamped_group.yaml"),
    )

    assert len(call_tracker) == 1
    output_dir = call_tracker[0]["output_dir"]
    assert output_dir.startswith("outputs/my_group_")
    assert TIMESTAMP_RE.search(output_dir.split("/")[1])


def test_resolve_run_names_uses_launch_time_stamp():
    result = resolve_run_names(name="exp", time_stamp_group=True, launch_time_stamp="2025-01-01_00-00-00")
    assert result["group"] == "exp_2025-01-01_00-00-00"


def test_meta_config_shared_group_timestamp(mock_git, tmp_path):
    """All configs from a meta-config share the same group timestamp."""
    import time

    call_tracker = []

    def test_fn(**kwargs):
        time.sleep(0.1)
        call_tracker.append(kwargs)

    # Write two simple base configs
    for name in ("a", "b"):
        (tmp_path / f"{name}.yaml").write_text(
            f"name: '{name}'\n"
            "function_name: test_fn\n"
            "time_stamp_group: true\n"
            "wandb_group: shared\n"
            "function_kwargs:\n"
            "  out: 'results/RUN_GROUP/data'\n"
        )

    # Write a meta-config that runs both
    (tmp_path / "meta.yaml").write_text(
        "experiments:\n"
        "  - config_axes:\n"
        f"    - ['{tmp_path}/a.yaml', '{tmp_path}/b.yaml']\n"
    )

    execute_experiments(
        function_map={"test_fn": test_fn},
        meta_config_path=str(tmp_path / "meta.yaml"),
    )

    assert len(call_tracker) == 2
    groups = [c["out"].split("/")[1] for c in call_tracker]
    assert groups[0] == groups[1]

