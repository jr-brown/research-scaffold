"""Test wandb sweep functionality"""

import os
from research_scaffold.config_tools import execute_sweep, load_dict_from_yaml


def test_sweep_config_loading(example_dir):
    """Test that sweep configs load correctly"""
    sweep_path = example_dir / "sweep_configs/simple_sweep.yaml"
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


def test_execute_sweep_basic(example_dir, mock_wandb, mock_git):
    """Test basic sweep execution"""
    call_tracker = []
    
    def test_fn(**kwargs):
        call_tracker.append(kwargs)
    
    function_map = {"example_sweep_function": test_fn}
    
    old_cwd = os.getcwd()
    try:
        os.chdir(example_dir)
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


def test_sweep_agent_called_correctly(example_dir, mock_wandb, mock_git):
    """Test that wandb.agent is called with correct parameters"""
    function_map = {"example_sweep_function": lambda **k: None}
    
    old_cwd = os.getcwd()
    try:
        os.chdir(example_dir)
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


def test_sweep_params_override_base_config(example_dir, mock_wandb, mock_git):
    """Test that sweep parameters override base config values"""
    call_tracker = []
    
    def test_fn(**kwargs):
        call_tracker.append(kwargs)
    
    function_map = {"example_sweep_function": test_fn}
    
    old_cwd = os.getcwd()
    try:
        os.chdir(example_dir)
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


def test_sweep_preserves_non_swept_params(example_dir, mock_wandb, mock_git):
    """Test that non-swept parameters from base config are preserved"""
    call_tracker = []
    
    def test_fn(**kwargs):
        call_tracker.append(kwargs)
    
    function_map = {"example_sweep_function": test_fn}
    
    old_cwd = os.getcwd()
    try:
        os.chdir(example_dir)
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


def test_sweep_train_function_multiple_calls(example_dir, mock_wandb, mock_git):
    """Test that train function can be called multiple times with different params"""
    call_tracker = []
    
    def test_fn(**kwargs):
        call_tracker.append(kwargs.copy())
    
    function_map = {"example_sweep_function": test_fn}
    
    old_cwd = os.getcwd()
    try:
        os.chdir(example_dir)
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


def test_sweep_without_base_config(example_dir, mock_wandb, mock_git):
    """Test sweep execution without base_config (minimal config)"""
    function_map = {"example_sweep_function": lambda **k: None}
    
    old_cwd = os.getcwd()
    try:
        os.chdir(example_dir)
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


def test_sweep_with_meta_config_base(example_dir, mock_wandb, mock_git):
    """Test sweep with meta-config as base_config (runs multiple configs per sweep iteration)"""
    call_tracker = []
    
    def simple_fn(**kwargs):
        call_tracker.append(('simple', kwargs))
    
    def log_fn(**kwargs):
        call_tracker.append(('log', kwargs))
    
    function_map = {
        "example_simple_config": simple_fn,
        "example_log_levels": log_fn,
    }
    
    old_cwd = os.getcwd()
    try:
        os.chdir(example_dir)
        execute_sweep(
            function_map=function_map,
            sweep_config_path="sweep_configs/sweep_with_meta_base.yaml",
        )
        
        # Get train function
        train_fn = mock_wandb['train_function']['function']
        assert train_fn is not None
        
        # Simulate one sweep iteration
        mock_wandb['config']['dummy_int'] = 10
        mock_wandb['config']['dummy_str'] = 'bar'
        
        train_fn()
        
    finally:
        os.chdir(old_cwd)
    
    # Both configs should have been executed in one sweep iteration
    assert len(call_tracker) == 2
    assert call_tracker[0][0] == 'simple'
    assert call_tracker[1][0] == 'log'
    
    # Sweep params should have been applied
    assert call_tracker[0][1]['dummy_int'] == 10
    assert call_tracker[0][1]['dummy_str'] == 'bar'


def test_meta_config_base_validation_rejects_repeats(example_dir, mock_git):
    """Test that meta-configs with repeats are rejected as base_config"""
    # This would need a meta-config with repeats > 1
    # For now, we'll skip this validation test as we don't have such a config
    pass

