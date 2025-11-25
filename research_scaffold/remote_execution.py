
import os
import json
import base64
import uuid
import yaml
import tempfile
from typing import Optional

import git
import sky

from .types import Config, InstanceConfig
from .util import get_logger, load_config_dict, deep_update

log = get_logger(__name__)


def get_git_info() -> tuple[str, str, str]:
    """Get git repository information using GitPython.
    
    Returns:
        tuple[str, str, str]: (repo_root, current_branch, repo_url)
    """
    try:
        repo = git.Repo(search_parent_directories=True)
        repo_root = repo.working_tree_dir
        branch = repo.active_branch.name
        url = repo.remotes.origin.url

        return repo_root, branch, url
    except git.InvalidGitRepositoryError:
        raise RuntimeError("Not in a git repository")


def build_experiment_run_command(
    config_dict: dict,
    rel_cwd: str,
    job_name: str,
    current_branch: str,
    commit_paths: Optional[list[str]],
    git_user_name: str,
    git_user_email: str,
) -> str:
    """Build the run command that executes the actual experiment.
    
    If commit_paths is provided, commits the specified paths/patterns to the current branch.
    
    Args:
        config_dict: The experiment configuration
        rel_cwd: Relative path from repo root to current working directory
        job_name: Name of the job
        current_branch: Current git branch name
        commit_paths: List of paths/patterns to commit (e.g., ["outputs/**", "logs/**"])
        git_user_name: Git user name for commits
        git_user_email: Git user email for commits
        
    Returns:
        Shell command string to execute the experiment
    """
    # Serialize config as base64-encoded JSON
    config_json = json.dumps(config_dict)
    config_b64 = base64.b64encode(config_json.encode()).decode()
    
    commands = []
    
    # Change to the working directory if needed
    if rel_cwd:
        commands.append(f"cd {rel_cwd}")
    
    # Execute the experiment
    commands.append(f"""python3 -c "
import json
import base64
import sys

config_b64 = '{config_b64}'
config_dict = json.loads(base64.b64decode(config_b64).decode())

from research_scaffold.config_tools import execute_from_config
from research_scaffold.types import Config
from main import function_map

config = Config(**config_dict)
execute_from_config(config, function_map=function_map, **config_dict)
" """)
    
    # Commit and push results if requested
    if commit_paths:
        # Build git add commands for each path
        add_commands = "\n".join([f"git add {path}" for path in commit_paths])
        
        commands.append(f"""git config --global user.email "{git_user_email}"
git config --global user.name "{git_user_name}"

# Add specified paths
{add_commands}

# Commit and push if there are changes
if git commit -m "Results from {job_name}"; then
    # Try pushing to the current branch
    if git push origin {current_branch}; then
        echo "✅ Results committed and pushed to branch: {current_branch}"
    else
        # Push failed - create a new branch and push there as failsafe
        echo "⚠️  Push to {current_branch} failed. Creating fallback branch..."
        FALLBACK_BRANCH="{current_branch}-results-$(date +%Y%m%d-%H%M%S)"
        git checkout -b "$FALLBACK_BRANCH"
        if git push origin "$FALLBACK_BRANCH"; then
            echo "✅ Results committed and pushed to fallback branch: $FALLBACK_BRANCH"
            echo "⚠️  IMPORTANT: Merge $FALLBACK_BRANCH back into {current_branch} manually"
        else
            echo "❌ Failed to push even to fallback branch. Results are committed locally but not pushed."
            exit 1
        fi
    fi
fi
""")
    
    return "\n".join(commands)


def build_sweep_run_command(
    sweep_dict: dict,
    rel_cwd: str,
    job_name: str,
    current_branch: str,
    commit_paths: Optional[list[str]],
    git_user_name: str,
    git_user_email: str,
) -> str:
    """Build the run command that executes a wandb sweep remotely.
    
    Args:
        sweep_dict: The sweep configuration dictionary
        rel_cwd: Relative path from repo root to current working directory
        job_name: Name of the job
        current_branch: Current git branch name
        commit_paths: List of paths/patterns to commit (e.g., ["outputs/**", "logs/**"])
        git_user_name: Git user name for commits
        git_user_email: Git user email for commits
        
    Returns:
        Shell command string to execute the sweep
    """
    # Serialize sweep config as base64-encoded JSON
    sweep_json = json.dumps(sweep_dict)
    sweep_b64 = base64.b64encode(sweep_json.encode()).decode()
    
    commands = []
    
    # Change to the working directory if needed
    if rel_cwd:
        commands.append(f"cd {rel_cwd}")
    
    # Execute the sweep
    commands.append(f"""python3 -c "
import json
import base64
import sys
import os

sweep_b64 = '{sweep_b64}'
sweep_dict = json.loads(base64.b64decode(sweep_b64).decode())

from research_scaffold.config_tools import execute_sweep_from_dict
from main import function_map

execute_sweep_from_dict(function_map=function_map, sweep_dict=sweep_dict)
" """)
    
    # Commit and push results if requested
    if commit_paths:
        # Build git add commands for each path
        add_commands = "\n".join([f"git add {path}" for path in commit_paths])
        
        commands.append(f"""git config --global user.email "{git_user_email}"
git config --global user.name "{git_user_name}"

# Add specified paths
{add_commands}

# Commit and push if there are changes
if git commit -m "Results from {job_name}"; then
    # Try pushing to the current branch
    if git push origin {current_branch}; then
        echo "✅ Results committed and pushed to branch: {current_branch}"
    else
        # Push failed - create a new branch and push there as failsafe
        echo "⚠️  Push to {current_branch} failed. Creating fallback branch..."
        FALLBACK_BRANCH="{current_branch}-results-$(date +%Y%m%d-%H%M%S)"
        git checkout -b "$FALLBACK_BRANCH"
        if git push origin "$FALLBACK_BRANCH"; then
            echo "✅ Results committed and pushed to fallback branch: $FALLBACK_BRANCH"
            echo "⚠️  IMPORTANT: Merge $FALLBACK_BRANCH back into {current_branch} manually"
        else
            echo "❌ Failed to push even to fallback branch. Results are committed locally but not pushed."
            exit 1
        fi
    fi
fi
""")
    
    return "\n".join(commands)


def launch_remote_job(
    instance_config: InstanceConfig,
    job_name: str,
    run_command: str,
) -> str:
    """Launch a remote job using SkyPilot (fire and forget).
    
    This launches the job asynchronously and returns immediately without
    waiting for the job to complete. Use `sky status` or `sky queue` to
    check job status, and `sky logs <cluster_name>` to view logs.
    
    Args:
        instance_config: Instance configuration specifying sky_config and patches
        job_name: Name of the job for logging
        run_command: The command to run remotely
        
    Returns:
        The cluster name for later reference (e.g., to check status or view logs)
        
    Raises:
        RuntimeError: If neither sky_config nor SKY_PATH environment variable is set
    """
    # Determine sky config path: use instance_config.sky_config or fall back to SKY_PATH env var
    sky_config_path = instance_config.sky_config
    if sky_config_path is None:
        sky_config_path = os.environ.get('SKY_PATH')
        if sky_config_path is None:
            raise RuntimeError(
                "No sky_config specified in instance config and SKY_PATH environment variable not set. "
                "Either set 'sky_config' in your instance config or set the SKY_PATH environment variable."
            )
        log.info(f"Using SKY_PATH environment variable: {sky_config_path}")
    
    original_cwd = os.getcwd()
    
    try:
        repo_root, _, _ = get_git_info()
        os.chdir(repo_root)
        
        log.info(f"Loading Sky config from: {sky_config_path}")
        sky_config = load_config_dict(sky_config_path)
        
        if instance_config.patch:
            log.info("Applying Sky config patch")
            patch_dict = load_config_dict(instance_config.patch)
            sky_config = deep_update(sky_config, patch_dict)
        
        sky_config['workdir'] = repo_root
        
        # Inject the run command into the Sky config
        original_run = sky_config.get('run', '')
        if original_run and not original_run.endswith('\n'):
            original_run += '\n'
        
        sky_config['run'] = original_run + run_command
        
        # Create SkyPilot task from the config
        log.info("Creating SkyPilot task")
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(sky_config, f)
            temp_config_path = f.name
        
        try:
            task = sky.Task.from_yaml(temp_config_path)
        finally:
            os.unlink(temp_config_path)
        
        # Generate cluster name
        cluster_name = f"c{uuid.uuid4().hex[:8]}"
        
        log.info(f"Cluster name: {cluster_name}")
        log.info("Launching remote job...")
        
        # Launch the task and wait for cluster to be provisioned
        # (but don't wait for the job to complete)
        request_id = sky.launch(
            task,
            cluster_name=cluster_name,
            down=True,  # Auto tear down after completion
        )
        
        # Wait for cluster provisioning to complete before continuing
        # This prevents race conditions when launching multiple jobs
        log.info("Waiting for cluster to be provisioned...")
        job_id, handle = sky.get(request_id)
        
        log.info("")
        log.info(f"🚀 Job '{job_name}' running on cluster: {cluster_name}")
        if job_id:
            log.info(f"   Job ID: {job_id}")
        log.info(f"   View logs:   sky logs {cluster_name}")
        log.info("   Check status: sky status")
        log.info("   Cluster will auto-terminate after job completes")
        
        return cluster_name
        
    finally:
        os.chdir(original_cwd)


def execute_config_remotely(
    instance_config: InstanceConfig,
    config: Config,
) -> str:
    """Execute a config remotely using SkyPilot (fire and forget).
    
    Args:
        instance_config: Instance configuration specifying sky_config and patches
        config: Experiment configuration to execute
        
    Returns:
        The cluster name for later reference
        
    Raises:
        RuntimeError: If neither sky_config nor SKY_PATH environment variable is set
    """
    log.info(f"Launching remote execution for config: {config.name}")
    
    # Get git information
    repo_root, current_branch, _ = get_git_info()
    original_cwd = os.getcwd()
    rel_cwd = os.path.relpath(original_cwd, repo_root)
    if rel_cwd == ".":
        rel_cwd = ""
    
    # Get local git user config to use on remote
    repo = git.Repo(repo_root)
    try:
        git_user_name = repo.config_reader().get_value("user", "name")
        git_user_email = repo.config_reader().get_value("user", "email")
        log.info(f"Using git identity: {git_user_name} <{git_user_email}>")
    except Exception:
        # Fallback if user hasn't configured git locally
        git_user_name = "Research Scaffold Remote"
        git_user_email = "remote-execution@research-scaffold"
        log.warning(f"Local git user not configured, using fallback: {git_user_name} <{git_user_email}>")
    
    # Prepare config dict (remove instance to avoid circular reference)
    config_dict = config.d.copy()
    config_dict.pop('instance', None)
    
    # Build the experiment run command
    experiment_run_cmd = build_experiment_run_command(
        config_dict=config_dict,
        rel_cwd=rel_cwd,
        job_name=config.name,
        current_branch=current_branch,
        commit_paths=instance_config.commit,
        git_user_name=git_user_name,
        git_user_email=git_user_email,
    )
    
    # Launch the remote job and return cluster name
    return launch_remote_job(
        instance_config=instance_config,
        job_name=config.name,
        run_command=experiment_run_cmd,
    )


def execute_sweep_remotely(
    instance_config: InstanceConfig,
    sweep_dict: dict,
    sweep_name: str,
) -> str:
    """Execute a wandb sweep remotely using SkyPilot (fire and forget).
    
    Args:
        instance_config: Instance configuration specifying sky_config and patches
        sweep_dict: Sweep configuration dictionary
        sweep_name: Name of the sweep for logging
        
    Returns:
        The cluster name for later reference
        
    Raises:
        RuntimeError: If neither sky_config nor SKY_PATH environment variable is set
    """
    log.info(f"Launching remote execution for sweep: {sweep_name}")
    
    # Get git information
    repo_root, current_branch, _ = get_git_info()
    original_cwd = os.getcwd()
    rel_cwd = os.path.relpath(original_cwd, repo_root)
    if rel_cwd == ".":
        rel_cwd = ""
    
    # Get local git user config to use on remote
    repo = git.Repo(repo_root)
    try:
        git_user_name = repo.config_reader().get_value("user", "name")
        git_user_email = repo.config_reader().get_value("user", "email")
        log.info(f"Using git identity: {git_user_name} <{git_user_email}>")
    except Exception:
        # Fallback if user hasn't configured git locally
        git_user_name = "Research Scaffold Remote"
        git_user_email = "remote-execution@research-scaffold"
        log.warning(f"Local git user not configured, using fallback: {git_user_name} <{git_user_email}>")
    
    # Build the sweep run command
    sweep_run_cmd = build_sweep_run_command(
        sweep_dict=sweep_dict,
        rel_cwd=rel_cwd,
        job_name=sweep_name,
        current_branch=current_branch,
        commit_paths=instance_config.commit,
        git_user_name=git_user_name,
        git_user_email=git_user_email,
    )
    
    # Launch the remote job and return cluster name
    return launch_remote_job(
        instance_config=instance_config,
        job_name=sweep_name,
        run_command=sweep_run_cmd,
    )
