"""Test wandb sweep functionality"""

import os
from pathlib import Path
from research_scaffold.config_tools import execute_sweep, load_dict_from_yaml


TEST_DIR = Path(__file__).parent


def test_sweep_config_loading():
    """Test that sweep configs load correctly"""
    sweep_path = TEST_DIR / "sweep_configs/simple_sweep.yaml"
    sweep_dict = load_dict_from_yaml(str(sweep_path))
    
    # Check custom fields
    assert "base_config" in sweep_dict
    assert sweep_dict["base_config"] == "configs/sweep_base.yaml"
    assert sweep_dict["sweep_count"] == 5
    assert sweep_dict["project"] == "research-scaffold-test"
    
    # Check wandb fields
    assert sweep_dict["method"] == "bayes"
    assert sweep_dict["metric"]["name"] == "dummy_metric"
    assert sweep_dict["metric"]["goal"] == "maximize"
    
    # Check parameters
    assert "dummy_int" in sweep_dict["parameters"]
    assert "dummy_str" in sweep_dict["parameters"]


def test_execute_sweep_basic(mock_wandb, mock_git):
    """Test basic sweep execution"""
    call_tracker = []
    
    def test_fn(**kwargs):
        call_tracker.append(kwargs)
    
    function_map = {"example_sweep_function": test_fn}
    
    old_cwd = os.getcwd()
    try:
        os.chdir(TEST_DIR)
        execute_sweep(
            function_map=function_map,
            sweep_config_path="sweep_configs/simple_sweep.yaml",
        )
    finally:
        os.chdir(old_cwd)
    
    # Verify wandb.sweep was called
    assert mock_wandb['sweep'].called
    
    # Check sweep config passed to wandb
    sweep_call = mock_wandb['sweep'].call_args
    assert sweep_call[1]['project'] == 'research-scaffold-test'
    
    # Verify sweep config doesn't have custom fields
    sweep_config = sweep_call[1]['sweep']
    assert 'base_config' not in sweep_config
    assert 'sweep_count' not in sweep_config
    assert sweep_config['method'] == 'bayes'


def test_sweep_agent_called_correctly(mock_wandb, mock_git):
    """Test that wandb.agent is called with correct parameters"""
    function_map = {"example_sweep_function": lambda **k: None}
    
    old_cwd = os.getcwd()
    try:
        os.chdir(TEST_DIR)
        execute_sweep(
            function_map=function_map,
            sweep_config_path="sweep_configs/simple_sweep.yaml",
        )
    finally:
        os.chdir(old_cwd)
    
    # Verify wandb.agent was called
    assert mock_wandb['agent'].called
    
    # Check agent was called with sweep_id
    agent_call_info = mock_wandb['agent_call_info']['called_with']
    assert agent_call_info['sweep_id'] == 'mock-sweep-123'
    assert agent_call_info['count'] == 5
    assert agent_call_info['project'] == 'research-scaffold-test'


def test_sweep_params_override_base_config(mock_wandb, mock_git):
    """Test that sweep parameters override base config values"""
    call_tracker = []
    
    def test_fn(**kwargs):
        call_tracker.append(kwargs)
    
    function_map = {"example_sweep_function": test_fn}
    
    old_cwd = os.getcwd()
    try:
        os.chdir(TEST_DIR)
        execute_sweep(
            function_map=function_map,
            sweep_config_path="sweep_configs/simple_sweep.yaml",
        )
        
        # Get the train function
        train_fn = mock_wandb['train_function']['function']
        assert train_fn is not None
        
        # Simulate sweep providing parameters
        mock_wandb['config']['dummy_int'] = 99  # Override base config value (5)
        mock_wandb['config']['dummy_str'] = 'bar'  # Override base config value ('foo')
        
        # Call train function
        train_fn()
        
    finally:
        os.chdir(old_cwd)
    
    # Verify function was called with sweep params
    assert len(call_tracker) == 1
    assert call_tracker[0]['dummy_int'] == 99  # Sweep value, not base config (5)
    assert call_tracker[0]['dummy_str'] == 'bar'  # Sweep value, not base config ('foo')
    assert call_tracker[0]['dummy_bool'] is True  # From base config (not swept)


def test_sweep_preserves_non_swept_params(mock_wandb, mock_git):
    """Test that non-swept parameters from base config are preserved"""
    call_tracker = []
    
    def test_fn(**kwargs):
        call_tracker.append(kwargs)
    
    function_map = {"example_sweep_function": test_fn}
    
    old_cwd = os.getcwd()
    try:
        os.chdir(TEST_DIR)
        execute_sweep(
            function_map=function_map,
            sweep_config_path="sweep_configs/simple_sweep.yaml",
        )
        
        train_fn = mock_wandb['train_function']['function']
        
        # Only set swept params
        mock_wandb['config']['dummy_int'] = 10
        mock_wandb['config']['dummy_str'] = 'baz'
        # dummy_bool is NOT swept, should come from base config
        
        train_fn()
        
    finally:
        os.chdir(old_cwd)
    
    # Non-swept param should come from base config
    assert call_tracker[0]['dummy_bool'] is True  # From base config


def test_sweep_train_function_multiple_calls(mock_wandb, mock_git):
    """Test that train function can be called multiple times with different params"""
    call_tracker = []
    
    def test_fn(**kwargs):
        call_tracker.append(kwargs.copy())
    
    function_map = {"example_sweep_function": test_fn}
    
    old_cwd = os.getcwd()
    try:
        os.chdir(TEST_DIR)
        execute_sweep(
            function_map=function_map,
            sweep_config_path="sweep_configs/simple_sweep.yaml",
        )
        
        train_fn = mock_wandb['train_function']['function']
        
        # Simulate multiple sweep iterations
        for i in range(3):
            mock_wandb['config']['dummy_int'] = i * 10
            mock_wandb['config']['dummy_str'] = ['foo', 'bar', 'baz'][i]
            train_fn()
        
    finally:
        os.chdir(old_cwd)
    
    # Verify function called 3 times with different params
    assert len(call_tracker) == 3
    assert call_tracker[0]['dummy_int'] == 0
    assert call_tracker[1]['dummy_int'] == 10
    assert call_tracker[2]['dummy_int'] == 20
    
    assert call_tracker[0]['dummy_str'] == 'foo'
    assert call_tracker[1]['dummy_str'] == 'bar'
    assert call_tracker[2]['dummy_str'] == 'baz'


def test_sweep_without_base_config(mock_wandb, mock_git):
    """Test sweep execution without base_config (minimal config)"""
    function_map = {"example_sweep_function": lambda **k: None}
    
    old_cwd = os.getcwd()
    try:
        os.chdir(TEST_DIR)
        execute_sweep(
            function_map=function_map,
            sweep_config_path="sweep_configs/minimal_sweep.yaml",
        )
    finally:
        os.chdir(old_cwd)
    
    # Should work without base_config
    assert mock_wandb['sweep'].called
    assert mock_wandb['agent'].called
    
    sweep_config_arg = mock_wandb['sweep'].call_args[1]['sweep']
    assert sweep_config_arg['method'] == 'random'
    assert 'base_config' not in sweep_config_arg




def test_sweep_with_compositional_meta_config_base(mock_wandb, mock_git):
    """Test sweep with compositional meta-config as base (produces ONE config)"""
    call_tracker = []
    
    def multi_fn(**kwargs):
        call_tracker.append(kwargs)
    
    function_map = {"example_multi_arg_config": multi_fn}
    
    old_cwd = os.getcwd()
    try:
        os.chdir(TEST_DIR)
        execute_sweep(
            function_map=function_map,
            sweep_config_path="sweep_configs/sweep_with_compositional_meta_base.yaml",
        )
        
        train_fn = mock_wandb['train_function']['function']
        mock_wandb['config']['arg1'] = "overridden_by_sweep"
        
        train_fn()
        
    finally:
        os.chdir(old_cwd)
    
    # Should have composed config with root, level1a, and patch
    assert len(call_tracker) == 1
    assert call_tracker[0]['root_arg'] == 'read from root.yaml'
    assert call_tracker[0]['patch_arg'] == 'read from patch.yaml'
    assert call_tracker[0]['arg1'] == 'overridden_by_sweep'  # Sweep overrides


def test_sweep_rejects_multi_config_meta_base(mock_wandb, mock_git):
    """Test that meta-configs producing multiple configs are rejected as sweep base"""
    import pytest
    
    function_map = {"example_multi_arg_config": lambda **k: None}
    
    old_cwd = os.getcwd()
    try:
        os.chdir(TEST_DIR)
        
        with pytest.raises(ValueError, match="must produce exactly 1 config"):
            execute_sweep(
                function_map=function_map,
                sweep_config_path="sweep_configs/sweep_with_multi_config_base.yaml",
            )
        
    finally:
        os.chdir(old_cwd)


def test_meta_config_with_multiple_sweeps(mock_wandb, mock_git):
    """Test meta-config that runs multiple sweeps sequentially"""
    from research_scaffold.config_tools import execute_experiments
    
    call_tracker = []
    
    def sweep_fn(**kwargs):
        call_tracker.append(kwargs)
    
    function_map = {"example_sweep_function": sweep_fn}
    
    old_cwd = os.getcwd()
    try:
        os.chdir(TEST_DIR)
        execute_experiments(
            function_map=function_map,
            meta_config_path="meta_configs/multi_sweep_test.yaml",
        )
    finally:
        os.chdir(old_cwd)
    
    # wandb.sweep should be called twice (two sweep experiments)
    assert mock_wandb['sweep'].call_count == 2
    
    # wandb.agent should be called twice
    assert mock_wandb['agent'].call_count == 2


def test_sweep_with_common_root_patch(mock_wandb, mock_git):
    """Test that common_root and common_patch are applied to sweep base_config"""
    from research_scaffold.config_tools import execute_experiments
    
    call_tracker = []
    
    def multi_fn(**kwargs):
        call_tracker.append(kwargs)
    
    function_map = {
        "example_simple_config": lambda **k: None,
        "example_log_levels": lambda **k: None,
        "example_multi_arg_config": multi_fn,
    }
    
    old_cwd = os.getcwd()
    try:
        os.chdir(TEST_DIR)
        execute_experiments(
            function_map=function_map,
            meta_config_path="meta_configs/sweep_with_common_root.yaml",
        )
        
        # Get train function and simulate a sweep run
        train_fn = mock_wandb['train_function']['function']
        mock_wandb['config']['arg1'] = 999
        mock_wandb['config']['arg2'] = False
        
        train_fn()
        
    finally:
        os.chdir(old_cwd)
    
    # Verify composed config was used (common_root + base + common_patch)
    assert len(call_tracker) == 1
    
    # Should have root_arg from common_root (root.yaml)
    assert call_tracker[0]['root_arg'] == 'read from root.yaml'
    
    # Should have patch_arg from common_patch (patch.yaml)  
    assert call_tracker[0]['patch_arg'] == 'read from patch.yaml'
    
    # Should have sweep params
    assert call_tracker[0]['arg1'] == 999
    assert call_tracker[0]['arg2'] is False
    
    assert mock_wandb['sweep'].called


def test_sweep_with_nested_parameters(mock_wandb, mock_git):
    """Test that nested parameters are properly merged (recursive, not shallow)"""
    call_tracker = []
    
    def test_fn(**kwargs):
        call_tracker.append(kwargs)
    
    function_map = {"example_sweep_function": test_fn}
    
    old_cwd = os.getcwd()
    try:
        os.chdir(TEST_DIR)
        execute_sweep(
            function_map=function_map,
            sweep_config_path="sweep_configs/nested_params_sweep.yaml",
        )
        
        train_fn = mock_wandb['train_function']['function']
        
        # Simulate wandb providing nested sweep parameters
        mock_wandb['config']['dummy_int'] = 10  # Top-level override
        mock_wandb['config']['model'] = {
            'learning_rate': 0.005,  # Override from sweep
            'hidden_size': 256,      # Override from sweep
            # dropout NOT in sweep params - should preserve from base
        }
        mock_wandb['config']['optimizer'] = {
            'momentum': 0.92,  # Override from sweep
            # weight_decay NOT in sweep params - should preserve from base
        }
        
        train_fn()
        
    finally:
        os.chdir(old_cwd)
    
    # Verify function was called
    assert len(call_tracker) == 1
    call = call_tracker[0]
    
    # Top-level param should be overridden
    assert call['dummy_int'] == 10
    
    # Nested model params: swept values override, non-swept preserved
    assert call['model']['learning_rate'] == 0.005  # From sweep
    assert call['model']['hidden_size'] == 256  # From sweep
    assert call['model']['dropout'] == 0.1  # From base config (preserved!)
    
    # Nested optimizer params: same behavior
    assert call['optimizer']['momentum'] == 0.92  # From sweep
    assert call['optimizer']['weight_decay'] == 0.0001  # From base config (preserved!)


def test_sweep_name_substitution(mock_wandb, mock_git):
    """Test that SWEEP_NAME placeholder is substituted in function kwargs"""
    call_tracker = []
    
    def test_fn(**kwargs):
        call_tracker.append(kwargs)
    
    function_map = {"example_sweep_function": test_fn}
    
    old_cwd = os.getcwd()
    try:
        os.chdir(TEST_DIR)
        execute_sweep(
            function_map=function_map,
            sweep_config_path="sweep_configs/sweep_with_name.yaml",
        )
        
        train_fn = mock_wandb['train_function']['function']
        mock_wandb['config']['dummy_int'] = 10
        
        train_fn()
        
    finally:
        os.chdir(old_cwd)
    
    # Verify SWEEP_NAME was substituted
    assert len(call_tracker) == 1
    call = call_tracker[0]
    
    # SWEEP_NAME should be replaced with sweep_name from config
    assert call['output_dir'] == "results/my_hyperparameter_search/run_output"
    assert call['checkpoint_path'] == "checkpoints/my_hyperparameter_search/model.pt"


def test_sweep_name_defaults_to_sweep_id(mock_wandb, mock_git):
    """Test that SWEEP_NAME uses sweep_id when sweep_name not specified"""
    call_tracker = []
    
    def test_fn(**kwargs):
        call_tracker.append(kwargs)
    
    function_map = {"example_sweep_function": test_fn}
    
    old_cwd = os.getcwd()
    try:
        os.chdir(TEST_DIR)
        # Use a sweep config without sweep_name - should default to sweep_id
        execute_sweep(
            function_map=function_map,
            sweep_config_path="sweep_configs/simple_sweep.yaml",
        )
        
    finally:
        os.chdir(old_cwd)
    
    # Verify sweep was created (sweep_id was used as default sweep_name)
    assert mock_wandb['sweep'].called
    assert mock_wandb['sweep'].return_value == "mock-sweep-123"

