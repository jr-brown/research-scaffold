import pytest
from unittest.mock import Mock, patch
from pathlib import Path


@pytest.fixture
def mock_wandb():
    with patch('research_scaffold.config_tools.wandb') as mock:
        # Mock wandb.init for regular experiments
        mock.init.return_value.__enter__ = Mock()
        mock.init.return_value.__exit__ = Mock()
        
        # Mock wandb.sweep for sweep functionality
        mock.sweep.return_value = "mock-sweep-123"
        
        # Store train function and agent call info for testing
        train_function_holder = {'function': None}
        agent_call_holder = {'called_with': None}
        
        def mock_agent(sweep_id, function, count=None, project=None, entity=None):
            train_function_holder['function'] = function
            agent_call_holder['called_with'] = {
                'sweep_id': sweep_id,
                'function': function,
                'count': count,
                'project': project,
                'entity': entity,
            }
        
        mock.agent.side_effect = mock_agent
        
        # Mock wandb.config for sweep parameters
        mock.config = {}
        
        # Mock wandb.run for sweep run name
        mock.run = Mock()
        mock.run.name = "test-sweep-run-123"
        
        # Yield dict with everything
        yield {
            'wandb': mock,
            'sweep': mock.sweep,
            'agent': mock.agent,
            'init': mock.init,
            'config': mock.config,
            'run': mock.run,
            'train_function': train_function_holder,
            'agent_call_info': agent_call_holder,
        }


@pytest.fixture
def mock_git():
    with patch('research_scaffold.config_tools.subprocess.check_output') as mock:
        mock.return_value = b'abc1234\n'
        yield mock



