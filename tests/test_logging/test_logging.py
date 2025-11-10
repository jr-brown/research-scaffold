"""Test logging functionality including log_file_path and config logging"""

import os
import logging
import yaml
from pathlib import Path
from research_scaffold.config_tools import execute_experiments


TEST_DIR = Path(__file__).parent


def test_log_file_creation_with_run_name_substitution(mock_git, tmp_path):
    """Test that log files are created with RUN_NAME properly substituted"""
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(module)-15s %(levelname)-8s %(message)s",
        force=True,
    )
    
    call_tracker = []
    def test_fn(**kwargs):
        call_tracker.append(kwargs)
    
    function_map = {"example_multi_arg_config": test_fn}
    
    old_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        
        # Copy config that uses log_file_path
        import shutil
        shutil.copytree(TEST_DIR / "configs", tmp_path / "configs")
        
        execute_experiments(
            function_map=function_map,
            config_path="configs/config_with_logging.yaml",
        )
        
        # Verify log file created with RUN_NAME substituted
        log_dir = tmp_path / "logs"
        assert log_dir.exists()
        
        log_subdirs = list(log_dir.iterdir())
        assert len(log_subdirs) == 1
        assert log_subdirs[0].name.startswith("logging_test_run_")
        
        log_file = log_subdirs[0] / "output.log"
        assert log_file.exists()
        
        # Verify log has content
        log_contents = log_file.read_text()
        assert len(log_contents) > 0
        assert "Config Dict" in log_contents
        assert "logging_test_run" in log_contents
        
    finally:
        os.chdir(old_cwd)


def test_separate_log_files_per_run(mock_git, tmp_path):
    """Test that each run in a meta-config gets its own log directory"""
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(module)-15s %(levelname)-8s %(message)s",
        force=True,
    )
    
    call_tracker = []
    def test_fn(**kwargs):
        call_tracker.append(kwargs)
    
    function_map = {"example_multi_arg_config": test_fn}
    
    old_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        
        # Copy test configs and meta-config
        import shutil
        shutil.copytree(TEST_DIR / "configs", tmp_path / "configs")
        shutil.copytree(TEST_DIR / "meta_configs", tmp_path / "meta_configs")
        
        execute_experiments(
            function_map=function_map,
            meta_config_path="meta_configs/two_runs.yaml",
        )
        
        # Verify two separate log directories created
        log_dir = tmp_path / "logs"
        assert log_dir.exists()
        
        log_subdirs = list(log_dir.iterdir())
        assert len(log_subdirs) == 2
        
        # Each should have its own log file
        run_names = set()
        for subdir in log_subdirs:
            assert (subdir / "output.log").exists()
            run_names.add(subdir.name)
        
        # Verify correct run names
        assert any("run_a" in name for name in run_names)
        assert any("run_b" in name for name in run_names)
        
        # Verify different params were used
        assert call_tracker[0]['arg1'] == 10
        assert call_tracker[1]['arg1'] == 20
        
    finally:
        os.chdir(old_cwd)


def test_save_config_path_with_run_name_substitution(mock_git, tmp_path):
    """Test that config files are saved with RUN_NAME properly substituted"""
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(module)-15s %(levelname)-8s %(message)s",
        force=True,
    )
    
    call_tracker = []
    def test_fn(**kwargs):
        call_tracker.append(kwargs)
    
    function_map = {"example_multi_arg_config": test_fn}
    
    old_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        
        # Create config with save_config_path
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        
        config_content = {
            "name": "config_save_test",
            "time_stamp_name": True,
            "function_name": "example_multi_arg_config",
            "save_config_path": "outputs/RUN_NAME/config.yaml",
            "function_kwargs": {
                "arg1": 42,
                "arg2": "test_value"
            }
        }
        
        with open(config_dir / "test_config.yaml", "w") as f:
            yaml.dump(config_content, f)
        
        execute_experiments(
            function_map=function_map,
            config_path="configs/test_config.yaml",
        )
        
        # Verify config file created with RUN_NAME substituted
        output_dir = tmp_path / "outputs"
        assert output_dir.exists()
        
        output_subdirs = list(output_dir.iterdir())
        assert len(output_subdirs) == 1
        assert output_subdirs[0].name.startswith("config_save_test_")
        
        config_file = output_subdirs[0] / "config.yaml"
        assert config_file.exists()
        
        # Verify config has correct content
        with open(config_file, "r") as f:
            saved_config = yaml.safe_load(f)
        
        assert saved_config["function_name"] == "example_multi_arg_config"
        assert saved_config["function_kwargs"]["arg1"] == 42
        assert saved_config["function_kwargs"]["arg2"] == "test_value"
        assert "config_save_test_" in saved_config["name"]  # Should have timestamp
        assert saved_config["save_config_path"].startswith("outputs/config_save_test_")
        
        # Verify function was actually called
        assert len(call_tracker) == 1
        assert call_tracker[0]["arg1"] == 42
        
    finally:
        os.chdir(old_cwd)

