"""
Remote Execution Module for Research Scaffold

This module enables remote execution of experiments using SkyPilot.

Usage:
------
1. Create a SkyPilot YAML config file (e.g., sky_config.yaml) in your repo with:
   - resources: GPU/CPU requirements, cloud provider, etc.
   - envs: Environment variables (WANDB_API_KEY, etc.)
   - setup: Commands to set up the environment (install deps, etc.)
   - run: Base run command (experiment command will be appended)

2. Reference it in your experiment config:
   instance:
     sky_config: "path/to/sky_config.yaml"  # Optional: uses SKY_PATH env var if not set
     patch:  # Optional: Override sky config values
       resources:
         accelerators: "A100:1"
     commit_results: true  # Auto-commit and push results

3. Run your experiment normally - it will execute remotely!

See example/sky_config.yaml and example/basic/with_remote.yaml for examples.
"""

import os
import json
import base64
import uuid
import yaml
import tempfile

import git
import sky

from .types import Config, InstanceConfig
from .util import get_logger

log = get_logger(__name__)


def deep_update(base_dict: dict, update_dict: dict) -> dict:
    """Deep merge update_dict into base_dict.
    
    Args:
        base_dict: Base dictionary
        update_dict: Dictionary with updates to merge
        
    Returns:
        Merged dictionary
    """
    result = base_dict.copy()
    for key, value in update_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


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


def load_sky_config(config_path: str) -> dict:
    """Load SkyPilot YAML configuration file.
    
    Args:
        config_path: Path to the Sky YAML config file
        
    Returns:
        Dictionary containing the Sky config
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def apply_sky_patch(sky_config: dict, patch: dict | str) -> dict:
    """Apply a patch to the Sky configuration.
    
    Args:
        sky_config: Base Sky configuration
        patch: Either an inline dict patch or path to a YAML patch file
        
    Returns:
        Patched Sky configuration
    """
    if isinstance(patch, str):
        # Load patch from file
        with open(patch, 'r') as f:
            patch_dict = yaml.safe_load(f)
    else:
        patch_dict = patch
    
    # Deep merge the patch into the config
    return deep_update(sky_config.copy(), patch_dict)


def build_experiment_run_command(
    config_dict: dict,
    rel_cwd: str,
    job_name: str,
    current_branch: str,
    commit_results: bool,
    git_user_name: str,
    git_user_email: str,
) -> str:
    """Build the run command that executes the actual experiment.
    
    Args:
        config_dict: The experiment configuration
        rel_cwd: Relative path from repo root to current working directory
        job_name: Name of the job
        current_branch: Current git branch name
        commit_results: Whether to commit and push results
        git_user_name: Git user name for commits
        git_user_email: Git user email for commits
        
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
    commands.append("# Convert git remote from HTTPS to SSH")
    commands.append("git remote set-url origin $(git config --get remote.origin.url | sed 's|https://github.com/|git@github.com:|')")
    
    # Change to the working directory if needed
    if rel_cwd:
        commands.append(f"cd {rel_cwd}")
    
    # Execute the experiment
    commands.append(f"""python3 -c "
import json
import base64
import sys
import os

# Decode config
config_b64 = '{config_b64}'
config_dict = json.loads(base64.b64decode(config_b64).decode())

# Import and execute
from research_scaffold.config_tools import execute_from_config
from research_scaffold.types import Config
from main import function_map

config = Config(**config_dict)
execute_from_config(config, function_map=function_map, **config_dict)
" """)
    
    # Commit and push results if requested
    if commit_results:
        commands.append(f"""
# Commit results
git config --global user.email "{git_user_email}"
git config --global user.name "{git_user_name}"
git add -A
if git commit -m "Results from {job_name}"; then
    git push origin {current_branch}
else
    echo "Nothing to commit"
fi
""")
    
    return "\n".join(commands)


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
    
    # Load the Sky configuration
    log.info(f"Loading Sky config from: {sky_config_path}")
    sky_config = load_sky_config(sky_config_path)
    
    # Apply patch if provided
    if instance_config.patch:
        log.info("Applying Sky config patch")
        sky_config = apply_sky_patch(sky_config, instance_config.patch)
    
    # Override workdir to use repo root
    sky_config['workdir'] = repo_root
    log.info(f"Setting workdir to repo root: {repo_root}")
    
    # Prepare config dict (remove instance to avoid circular reference)
    config_dict = config.d.copy()
    config_dict.pop('instance', None)
    
    # Build the experiment run command
    experiment_run_cmd = build_experiment_run_command(
        config_dict=config_dict,
        rel_cwd=rel_cwd,
        job_name=config.name,
        current_branch=current_branch,
        commit_results=instance_config.commit_results,
        git_user_name=git_user_name,
        git_user_email=git_user_email,
    )
    
    # Inject the experiment run command into the Sky config
    # The original 'run' in the Sky config should set up the environment
    # We append our experiment execution to it
    original_run = sky_config.get('run', '')
    if original_run and not original_run.endswith('\n'):
        original_run += '\n'
    
    sky_config['run'] = original_run + experiment_run_cmd
    
    # Create SkyPilot task from the config
    # Write config to a temporary file since sky.Task.from_yaml expects a file path
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
    
    # Wait for provisioning, then stream the actual job logs
    try:
        job_id, handle = sky.get(request_id)
        
        if handle:
            log.info(f"Cluster provisioned. Job ID: {job_id}")
            log.info("=" * 80)
            log.info("STREAMING JOB LOGS:")
            log.info("=" * 80)
            
            # Stream the actual job execution logs
            exit_code = sky.tail_logs(cluster_name, job_id, follow=True)
            
            log.info("=" * 80)
            log.info(f"JOB COMPLETE - Exit code: {exit_code}")
            log.info("=" * 80)
            log.info("Cluster will be automatically torn down.")
        else:
            log.warning("Job completed but no handle returned")
            
    except Exception as e:
        log.error(f"Failed to execute remote job: {e}")
        log.info(f"Check status with: sky status {cluster_name}")
        log.info(f"View logs: sky logs {cluster_name}")
        raise
