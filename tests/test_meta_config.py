"""Test meta-config functionality"""

import os
from research_scaffold.config_tools import (
    load_meta_config,
    process_meta_config,
    execute_experiments,
)


def test_meta_config_with_axes(example_dir, mock_git):
    """Test meta-config with config_axes (cartesian product)"""
    old_cwd = os.getcwd()
    try:
        os.chdir(example_dir)
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


def test_meta_config_with_repeats(example_dir, mock_git):
    """Test that repeats work correctly"""
    old_cwd = os.getcwd()
    try:
        os.chdir(example_dir)
        meta = load_meta_config("meta_configs/lennie_tests.yaml")
        
        # Last experiment has repeats: 2
        assert meta.experiments[-1].repeats == 2
        
        configs = process_meta_config(meta)
        
        # Should have multiple configs including repeated ones
        assert len(configs) >= 2
        
    finally:
        os.chdir(old_cwd)


def test_meta_config_bonus_dict(example_dir, mock_git):
    """Test that bonus_dict is applied to all configs"""
    old_cwd = os.getcwd()
    try:
        os.chdir(example_dir)
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


def test_meta_config_common_root_patch(example_dir, mock_git):
    """Test that common_root and common_patch are applied"""
    old_cwd = os.getcwd()
    try:
        os.chdir(example_dir)
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


def test_execute_meta_config_full(example_dir, mock_git):
    """Test executing a full meta-config"""
    call_tracker = []
    
    def test_fn(**kwargs):
        call_tracker.append(kwargs)
    
    function_map = {"example_multi_arg_config": test_fn}
    
    old_cwd = os.getcwd()
    try:
        os.chdir(example_dir)
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

