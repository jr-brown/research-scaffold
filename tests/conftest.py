import pytest
from unittest.mock import Mock, patch
from pathlib import Path


@pytest.fixture
def mock_wandb():
    with patch('research_scaffold.config_tools.wandb') as mock:
        mock.init.return_value.__enter__ = Mock()
        mock.init.return_value.__exit__ = Mock()
        yield mock


@pytest.fixture
def mock_git():
    with patch('research_scaffold.config_tools.subprocess.check_output') as mock:
        mock.return_value = b'abc1234\n'
        yield mock


@pytest.fixture
def example_dir():
    return Path(__file__).parent.parent / "example"

