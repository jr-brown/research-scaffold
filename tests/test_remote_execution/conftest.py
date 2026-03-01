import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path


TEST_DIR = Path(__file__).parent
SKY_CONFIG_PATH = str(TEST_DIR / "sky_config.yaml")
REPO_ROOT = str(Path(__file__).parent.parent.parent)


@pytest.fixture
def mock_sky():
    """Mock the sky module for remote execution tests."""
    with patch('research_scaffold.remote_execution.sky') as mock:
        # Mock sky.Task.from_yaml to return a mock task
        mock_task = MagicMock()
        mock.Task.from_yaml.return_value = mock_task

        # Mock sky.launch to return a request_id string
        mock.launch.return_value = "req-launch-123"

        # Mock sky.get to return (job_id, handle)
        mock.get.return_value = (1, MagicMock())

        # Mock sky.status
        mock.status.return_value = "req-status-123"

        # Mock sky.jobs.launch to return a request_id string
        mock.jobs.launch.return_value = "req-managed-456"

        # Mock sky.jobs.tail_logs
        mock.jobs.tail_logs.return_value = None

        yield {
            'sky': mock,
            'task': mock_task,
            'launch': mock.launch,
            'get': mock.get,
            'jobs_launch': mock.jobs.launch,
            'jobs_tail_logs': mock.jobs.tail_logs,
            'Task': mock.Task,
        }


@pytest.fixture
def mock_git_info():
    """Mock get_git_info and get_git_user_info."""
    with patch('research_scaffold.remote_execution.get_git_info') as mock_info, \
         patch('research_scaffold.remote_execution.get_git_user_info') as mock_user:
        mock_info.return_value = (REPO_ROOT, "main", "https://github.com/test/repo.git")
        mock_user.return_value = ("Test User", "test@example.com")
        yield {
            'git_info': mock_info,
            'git_user_info': mock_user,
            'repo_root': REPO_ROOT,
        }
