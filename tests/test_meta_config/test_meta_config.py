"""Test meta-config functionality"""

import json
import os
import threading
import pytest
from pathlib import Path
from research_scaffold.config_tools import (
    load_meta_config,
    process_meta_config,
    execute_experiments,
)


TEST_DIR = Path(__file__).parent


def test_meta_config_with_axes(mock_git):
    """Test meta-config with config_axes (cartesian product)"""
    old_cwd = os.getcwd()
    try:
        os.chdir(TEST_DIR)
        meta = load_meta_config("meta_configs/lennie_tests.yaml")
        configs = process_meta_config(meta)
        
        # Should produce multiple configs from axes combinations
        assert len(configs) > 1
        
        # All should have bonus_dict applied (names are concatenated)
        for cfg in configs:
            assert "bonus_name" in cfg.name
            assert cfg.function_kwargs.get("bonus_arg") == "bonus_arg"
        
    finally:
        os.chdir(old_cwd)


def test_meta_config_with_repeats(mock_git):
    """Test that repeats work correctly"""
    old_cwd = os.getcwd()
    try:
        os.chdir(TEST_DIR)
        meta = load_meta_config("meta_configs/lennie_tests.yaml")
        
        # Last experiment has repeats: 2
        assert meta.experiments[-1].repeats == 2
        
        configs = process_meta_config(meta)
        
        # Should have multiple configs including repeated ones
        assert len(configs) >= 2
        
    finally:
        os.chdir(old_cwd)


def test_meta_config_bonus_dict(mock_git):
    """Test that bonus_dict is applied to all configs"""
    old_cwd = os.getcwd()
    try:
        os.chdir(TEST_DIR)
        meta = load_meta_config("meta_configs/lennie_tests.yaml")
        
        assert meta.bonus_dict is not None
        assert meta.bonus_dict["name"] == "bonus_name"
        
        configs = process_meta_config(meta)
        
        # All configs should have bonus_dict applied (names are concatenated)
        for cfg in configs:
            assert "bonus_name" in cfg.name
            assert "bonus_arg" in cfg.function_kwargs
            
    finally:
        os.chdir(old_cwd)


def test_meta_config_common_root_patch(mock_git):
    """Test that common_root and common_patch are applied"""
    old_cwd = os.getcwd()
    try:
        os.chdir(TEST_DIR)
        meta = load_meta_config("meta_configs/lennie_tests.yaml")
        
        assert meta.common_root == ["root.yaml"]
        assert meta.common_patch == ["patch.yaml"]
        
        configs = process_meta_config(meta)
        
        # All configs should have root and patch applied
        for cfg in configs:
            # root.yaml has root_arg
            assert "root_arg" in cfg.function_kwargs
            # patch.yaml has patch_arg
            assert "patch_arg" in cfg.function_kwargs
            
    finally:
        os.chdir(old_cwd)


def test_execute_meta_config_full(mock_git):
    """Test executing a full meta-config"""
    call_tracker = []
    
    def test_fn(**kwargs):
        call_tracker.append(kwargs)
    
    function_map = {"example_multi_arg_config": test_fn}
    
    old_cwd = os.getcwd()
    try:
        os.chdir(TEST_DIR)
        execute_experiments(
            function_map=function_map,
            meta_config_path="meta_configs/lennie_tests.yaml",
        )
        
        # Should have executed multiple configs
        assert len(call_tracker) > 1
        
        # All should have bonus_arg from bonus_dict
        for call in call_tracker:
            assert call.get("bonus_arg") == "bonus_arg"

    finally:
        os.chdir(old_cwd)


def test_parallel_execution(mock_git, tmp_path):
    """Test that parallel=True runs experiments in different processes"""

    def test_fn(**kwargs):
        (tmp_path / f"pid_{os.getpid()}").write_text("done")

    function_map = {"example_multi_arg_config": test_fn}

    old_cwd = os.getcwd()
    try:
        os.chdir(TEST_DIR)

        meta = load_meta_config("meta_configs/parallel_test.yaml")
        assert meta.parallel is True

        execute_experiments(
            function_map=function_map,
            meta_config_path="meta_configs/parallel_test.yaml",
        )

        pid_files = list(tmp_path.glob("pid_*"))
        assert len(pid_files) == 2
        pids = {f.name for f in pid_files}
        assert len(pids) == 2  # two different PIDs
    finally:
        os.chdir(old_cwd)


def test_sequential_execution_without_parallel(mock_git):
    """Test that without parallel=True, all experiments run on the main thread"""
    thread_ids = []

    def test_fn(**kwargs):
        thread_ids.append(threading.current_thread().ident)

    function_map = {"example_multi_arg_config": test_fn}

    old_cwd = os.getcwd()
    try:
        os.chdir(TEST_DIR)

        meta = load_meta_config("meta_configs/lennie_tests.yaml")
        assert meta.parallel is False

        execute_experiments(
            function_map=function_map,
            meta_config_path="meta_configs/lennie_tests.yaml",
        )

        assert len(thread_ids) > 1
        assert all(t == threading.current_thread().ident for t in thread_ids)
    finally:
        os.chdir(old_cwd)


def test_parallel_error_propagation(mock_git, tmp_path):
    """Test that if one parallel experiment fails, the other still runs and the error is re-raised"""

    def test_fn(**kwargs):
        (tmp_path / f"pid_{os.getpid()}").write_text(kwargs.get("arg1", ""))
        if kwargs.get("arg1") == "read from level1a.yaml":
            raise ValueError("intentional failure")

    function_map = {"example_multi_arg_config": test_fn}

    old_cwd = os.getcwd()
    try:
        os.chdir(TEST_DIR)

        with pytest.raises(RuntimeError, match="intentional failure"):
            execute_experiments(
                function_map=function_map,
                meta_config_path="meta_configs/parallel_test.yaml",
            )

        # Both processes should have started
        assert len(list(tmp_path.glob("pid_*"))) == 2
    finally:
        os.chdir(old_cwd)


def test_parallel_same_results_as_sequential(mock_git, tmp_path):
    """Test that parallel execution produces the same function calls as sequential"""
    from research_scaffold.config_tools import execute_from_config

    sequential_kwargs = []

    def seq_fn(**kwargs):
        sequential_kwargs.append(kwargs)

    def par_fn(**kwargs):
        (tmp_path / f"kwargs_{os.getpid()}.json").write_text(json.dumps(kwargs))

    old_cwd = os.getcwd()
    try:
        os.chdir(TEST_DIR)

        # Sequential baseline: manually loop over configs from the same YAML
        meta = load_meta_config("meta_configs/parallel_test.yaml")
        configs = process_meta_config(meta)
        for config in configs:
            execute_from_config(config, function_map={"example_multi_arg_config": seq_fn}, **config.d)

        # Parallel via execute_experiments
        execute_experiments(
            function_map={"example_multi_arg_config": par_fn},
            meta_config_path="meta_configs/parallel_test.yaml",
        )

        parallel_kwargs = [json.loads(f.read_text()) for f in sorted(tmp_path.glob("kwargs_*.json"))]
        assert len(sequential_kwargs) == len(parallel_kwargs)
        # Sort both by a stable key since parallel order is nondeterministic
        seq_sorted = sorted(sequential_kwargs, key=lambda d: d.get("arg1", ""))
        par_sorted = sorted(parallel_kwargs, key=lambda d: d.get("arg1", ""))
        assert seq_sorted == par_sorted
    finally:
        os.chdir(old_cwd)

