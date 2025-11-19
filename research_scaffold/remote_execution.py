
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
    repo_url: str,
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
        repo_url: Git repository URL
        
    Returns:
        Shell command string to execute the experiment
    """
    # Serialize config as base64-encoded JSON
    config_json = json.dumps(config_dict)
    config_b64 = base64.b64encode(config_json.encode()).decode()
    
    commands = []
    
    # Set git safe directory first (needed for any git commands)
    commands.append("git config --global --add safe.directory ~/sky_workdir")
    
    # Convert git remote to SSH (workdir is synced by now)
    commands.append("git remote set-url origin $(git config --get remote.origin.url | sed 's|https://github.com/|git@github.com:|')")
    
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
    repo_url: str,
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
        repo_url: Git repository URL
        
    Returns:
        Shell command string to execute the sweep
    """
    # Serialize sweep config as base64-encoded JSON
    sweep_json = json.dumps(sweep_dict)
    sweep_b64 = base64.b64encode(sweep_json.encode()).decode()
    
    commands = []
    
    # Set git safe directory first (needed for any git commands)
    commands.append("git config --global --add safe.directory ~/sky_workdir")
    
    # Convert git remote to SSH (workdir is synced by now)
    commands.append("git remote set-url origin $(git config --get remote.origin.url | sed 's|https://github.com/|git@github.com:|')")
    
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
) -> None:
    """Launch a remote job using SkyPilot.
    
    This is the common logic for launching any type of remote job (config or sweep).
    
    Args:
        instance_config: Instance configuration specifying sky_config and patches
        job_name: Name of the job for logging
        run_command: The command to run remotely
        
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
    
    # Ensure WANDB_API_KEY is available for environment variable expansion
    if "WANDB_API_KEY" not in os.environ:
        raise RuntimeError(
            "WANDB_API_KEY not found in environment variables. "
            "Please set the WANDB_API_KEY environment variable."
        )
    # Load the Sky configuration
    log.info(f"Loading Sky config from: {sky_config_path}")
    
    # Read the original file as text to preserve formatting
    with open(sky_config_path, 'r') as f:
        sky_config_text = f.read()
    
    # Expand environment variables in the YAML text (handles ${VAR} syntax)
    sky_config_text = os.path.expandvars(sky_config_text)
    
    # Now parse the expanded text
    sky_config = yaml.safe_load(sky_config_text)
    
    # Apply patch if provided
    if instance_config.patch:
        log.info("Applying Sky config patch")
        patch_dict = load_config_dict(instance_config.patch)
        sky_config = deep_update(sky_config, patch_dict)
    
    # Get git information and override workdir to use repo root
    repo_root, _, _ = get_git_info()
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
        # Clean up the temp file
        os.unlink(temp_config_path)
    
    # Generate cluster name
    cluster_name = f"c{uuid.uuid4().hex[:8]}"
    
    log.info(f"Cluster name: {cluster_name}")
    log.info("Launching task...")
    
    # Launch the task
    request_id = sky.launch(
        task,
        cluster_name=cluster_name,
        down=True,  # Auto tear down after completion
    )
    
    log.info(f"Launch request submitted. Request ID: {request_id}")
    log.info(f"Job is running remotely on cluster: {cluster_name}")
    log.info("")
    log.info(f"To check status:  sky status {cluster_name}")
    log.info(f"To view logs:     sky logs {cluster_name}")
    log.info(f"To tear down:     sky down {cluster_name}")
    log.info("")
    log.info("Cluster will automatically tear down after completion (down=True)")
    
    # Job is now running remotely - we can return immediately (fire and forget)


def execute_config_remotely(
    instance_config: InstanceConfig,
    config: Config,
) -> None:
    """Execute a config remotely using SkyPilot.
    
    Args:
        instance_config: Instance configuration specifying sky_config and patches
        config: Experiment configuration to execute
        
    Raises:
        RuntimeError: If neither sky_config nor SKY_PATH environment variable is set
    """
    log.info(f"Launching remote execution for config: {config.name}")
    
    # Get git information
    repo_root, current_branch, repo_url = get_git_info()
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
        repo_url=repo_url,
    )
    
    # Launch the remote job
    launch_remote_job(
        instance_config=instance_config,
        job_name=config.name,
        run_command=experiment_run_cmd,
    )


def execute_sweep_remotely(
    instance_config: InstanceConfig,
    sweep_dict: dict,
    sweep_name: str,
) -> None:
    """Execute a wandb sweep remotely using SkyPilot.
    
    Args:
        instance_config: Instance configuration specifying sky_config and patches
        sweep_dict: Sweep configuration dictionary
        sweep_name: Name of the sweep for logging
        
    Raises:
        RuntimeError: If neither sky_config nor SKY_PATH environment variable is set
    """
    log.info(f"Launching remote execution for sweep: {sweep_name}")
    
    # Get git information
    repo_root, current_branch, repo_url = get_git_info()
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
        repo_url=repo_url,
    )
    
    # Launch the remote job
    launch_remote_job(
        instance_config=instance_config,
        job_name=sweep_name,
        run_command=sweep_run_cmd,
    )
