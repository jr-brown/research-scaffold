"""Test inline config dictionaries in place of file paths"""

import os
import logging
from pathlib import Path
from research_scaffold.config_tools import execute_experiments


TEST_DIR = Path(__file__).parent


def test_inline_config_in_config_steps(mock_git, tmp_path):
    """Test that inline dicts can be used in config_steps instead of file paths"""
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(module)-15s %(levelname)-8s %(message)s",
        force=True,
    )
    
    call_tracker = []
    def test_fn(**kwargs):
        call_tracker.append(kwargs)
    
    function_map = {"example_function": test_fn}
    
    old_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        
        # Create meta-config with inline config dicts
        import yaml
        meta_config = {
            "experiments": [
                {
                    "config_steps": [
                        # Inline config dict instead of file path
                        {
                            "name": "test_inline",
                            "function_name": "example_function",
                            "function_kwargs": {
                                "param1": 100,
                                "param2": "base_value"
                            }
                        },
                        # Another inline dict to patch/update
                        {
                            "function_kwargs": {
                                "param2": "patched_value",
                                "param3": True
                            }
                        }
                    ]
                }
            ]
        }
        
        meta_config_dir = tmp_path / "meta_configs"
        meta_config_dir.mkdir()
        
        with open(meta_config_dir / "inline_test.yaml", "w") as f:
            yaml.dump(meta_config, f)
        
        execute_experiments(
            function_map=function_map,
            meta_config_path="meta_configs/inline_test.yaml",
        )
        
        # Verify function was called with merged params
        assert len(call_tracker) == 1
        assert call_tracker[0]["param1"] == 100
        assert call_tracker[0]["param2"] == "patched_value"  # Patched value
        assert call_tracker[0]["param3"] is True
        
    finally:
        os.chdir(old_cwd)


def test_mixed_inline_and_file_configs(mock_git, tmp_path):
    """Test mixing inline dicts and file paths in config_steps"""
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(module)-15s %(levelname)-8s %(message)s",
        force=True,
    )
    
    call_tracker = []
    def test_fn(**kwargs):
        call_tracker.append(kwargs)
    
    function_map = {"example_function": test_fn}
    
    old_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        
        # Create a file-based config
        import yaml
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        
        base_config = {
            "name": "test_mixed",
            "function_name": "example_function",
            "function_kwargs": {
                "from_file": "file_value",
                "shared_param": "file"
            }
        }
        
        with open(config_dir / "base.yaml", "w") as f:
            yaml.dump(base_config, f)
        
        # Create meta-config mixing file and inline
        meta_config = {
            "experiments": [
                {
                    "config_steps": [
                        "configs/base.yaml",  # File path
                        {  # Inline dict
                            "function_kwargs": {
                                "from_inline": "inline_value",
                                "shared_param": "inline"  # Override file value
                            }
                        }
                    ]
                }
            ]
        }
        
        meta_config_dir = tmp_path / "meta_configs"
        meta_config_dir.mkdir()
        
        with open(meta_config_dir / "mixed_test.yaml", "w") as f:
            yaml.dump(meta_config, f)
        
        execute_experiments(
            function_map=function_map,
            meta_config_path="meta_configs/mixed_test.yaml",
        )
        
        # Verify function was called with both file and inline params
        assert len(call_tracker) == 1
        assert call_tracker[0]["from_file"] == "file_value"
        assert call_tracker[0]["from_inline"] == "inline_value"
        assert call_tracker[0]["shared_param"] == "inline"  # Inline overrides file
        
    finally:
        os.chdir(old_cwd)


def test_inline_common_root_and_patch(mock_git, tmp_path):
    """Test that inline dicts work in common_root and common_patch"""
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(module)-15s %(levelname)-8s %(message)s",
        force=True,
    )
    
    call_tracker = []
    def test_fn(**kwargs):
        call_tracker.append(kwargs)
    
    function_map = {"example_function": test_fn}
    
    old_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        
        import yaml
        
        # Create a minimal target config file
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        
        target_config = {
            "name": "test_target",
            "function_name": "example_function",
            "function_kwargs": {
                "target_param": "target_value"
            }
        }
        
        with open(config_dir / "target.yaml", "w") as f:
            yaml.dump(target_config, f)
        
        # Create meta-config with inline common_root and common_patch
        meta_config = {
            "common_root": {  # Inline dict for common_root
                "function_kwargs": {
                    "root_param": "root_value"
                }
            },
            "common_patch": {  # Inline dict for common_patch
                "function_kwargs": {
                    "patch_param": "patch_value"
                }
            },
            "experiments": [
                {
                    "config": "configs/target.yaml"
                }
            ]
        }
        
        meta_config_dir = tmp_path / "meta_configs"
        meta_config_dir.mkdir()
        
        with open(meta_config_dir / "common_test.yaml", "w") as f:
            yaml.dump(meta_config, f)
        
        execute_experiments(
            function_map=function_map,
            meta_config_path="meta_configs/common_test.yaml",
        )
        
        # Verify all params from root, target, and patch are present
        assert len(call_tracker) == 1
        assert call_tracker[0]["root_param"] == "root_value"
        assert call_tracker[0]["target_param"] == "target_value"
        assert call_tracker[0]["patch_param"] == "patch_value"
        
    finally:
        os.chdir(old_cwd)

