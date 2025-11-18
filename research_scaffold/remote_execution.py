
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
    git push origin {current_branch}
    echo "✅ Results committed and pushed to branch: {current_branch}"
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
    sky_config = load_config_dict(sky_config_path)
    
    # Apply patch if provided
    if instance_config.patch:
        log.info("Applying Sky config patch")
        patch_dict = load_config_dict(instance_config.patch)
        sky_config = deep_update(sky_config, patch_dict)
    
    # Override workdir to use repo root
    sky_config['workdir'] = repo_root
    
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
            log.info(f"Cluster provisioned. Streaming logs for job {job_id}...")
            
            # Stream the actual job execution logs
            exit_code = sky.tail_logs(cluster_name, job_id, follow=True)
            
            if exit_code == 0:
                log.info("Job completed successfully. Cluster will be torn down.")
            else:
                log.warning(f"Job failed with exit code {exit_code}. Cluster will be torn down.")
        else:
            log.warning("Job completed but no handle returned")
            
    except Exception as e:
        log.error(f"Failed to execute remote job: {e}")
        log.info(f"Check status: sky status {cluster_name}")
        raise
