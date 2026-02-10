
import os
import json
import base64
import uuid
import yaml
import tempfile
import subprocess
from typing import Optional

import git
import sky

from .types import Config, InstanceConfig
from .util import get_logger, load_config_dict, deep_update

log = get_logger(__name__)


def start_log_streaming(request_id: str, cluster_name: str, log_file_path: str) -> subprocess.Popen:
    """Stream logs from a SkyPilot job to a local file using the SDK.
    
    Chains two SDK calls:
    1. sky.stream_and_get() - streams launch logs (provisioning, setup)
    2. sky.tail_logs() - streams actual job output
    
    Args:
        request_id: The request ID from sky.launch()
        cluster_name: The cluster name for tailing job logs
        log_file_path: Local path to write logs to
        
    Returns:
        The background process
    """
    import sys
    
    log_dir = os.path.dirname(log_file_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    
    log.info(f"📝 Streaming logs to: {log_file_path}")
    
    # Stream launch logs, then tail job logs
    # Errors are written to the log file so they're visible
    script = f'''
import sky
import traceback

with open("{log_file_path}", "w") as f:
    try:
        # Stream launch logs (provisioning, setup, job submission)
        sky.stream_and_get("{request_id}", output_stream=f)
        f.write("\\n--- Job output ---\\n")
        f.flush()
        # Stream actual job logs
        sky.tail_logs("{cluster_name}", job_id=None, follow=True, output_stream=f)
    except Exception as e:
        f.write(f"\\n\\n=== Log streaming error ===\\n{{e}}\\n")
        f.write(traceback.format_exc())
        f.flush()
'''
    
    # Use the same Python interpreter that's running this code
    process = subprocess.Popen(
        [sys.executable, '-c', script],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    
    log.info(f"   Monitor: tail -f {log_file_path}")
    return process


def save_config_locally(config_dict: dict, save_path: str, run_name: str) -> str:
    """Save config to a local file.
    
    Args:
        config_dict: The config dictionary to save
        save_path: Path template (may contain RUN_NAME placeholder)
        run_name: Name to substitute for RUN_NAME
        
    Returns:
        The actual path where config was saved
    """
    # Replace RUN_NAME placeholder
    actual_path = save_path.replace("RUN_NAME", run_name)
    
    # Ensure directory exists
    config_dir = os.path.dirname(actual_path)
    if config_dir:
        os.makedirs(config_dir, exist_ok=True)
    
    # Save as YAML
    with open(actual_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    
    log.info(f"💾 Config saved locally: {actual_path}")
    return actual_path


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


def get_existing_cluster(cluster_name: str) -> Optional[dict]:
    """Check if a cluster with the given name already exists.

    Args:
        cluster_name: Name of the cluster to check

    Returns:
        Cluster record dict if exists, None otherwise
    """
    try:
        request_id = sky.status(cluster_names=[cluster_name])
        clusters = sky.get(request_id)
        if clusters:
            cluster = clusters[0]
            # SkyPilot 0.11+ returns pydantic models; convert to dict
            if hasattr(cluster, 'model_dump'):
                return cluster.model_dump()
            elif isinstance(cluster, dict):
                return cluster
            else:
                return {'status': getattr(cluster, 'status', 'UNKNOWN'), 'name': cluster_name}
    except Exception as e:
        log.warning(f"Failed to check cluster status: {e}")
    return None


def launch_remote_job(
    instance_config: InstanceConfig,
    job_name: str,
    run_command: str,
) -> tuple[str, Optional[str], bool]:
    """Launch a remote job using SkyPilot (fire and forget).
    
    This launches the job asynchronously and returns immediately without
    waiting for the job to complete.
    
    If instance_config.name is provided and a cluster with that name already
    exists, the function will skip launching and return the existing cluster name.
    
    Args:
        instance_config: Instance configuration specifying sky_config and patches
        job_name: Name of the job for logging
        run_command: The command to run remotely
        
    Returns:
        Tuple of (cluster_name, request_id, was_already_running):
        - cluster_name: The cluster name for later reference
        - request_id: The SkyPilot request ID (for log streaming), None if already running
        - was_already_running: True if cluster already existed (no new job launched)
        
    Raises:
        RuntimeError: If neither sky_config nor SKY_PATH environment variable is set
    """
    # Determine cluster name: use custom name or generate one
    if instance_config.name:
        cluster_name = instance_config.name
        
        # Check if cluster with this name already exists
        existing = get_existing_cluster(cluster_name)
        if existing:
            status = existing['status']
            log.info("")
            log.info(f"⏭️  Instance '{cluster_name}' is already running (status: {status})")
            log.info(f"   View logs:   sky logs {cluster_name}")
            log.info("   Check status: sky status")
            log.info("   Skipping launch, continuing without spinning up new instance.")
            return cluster_name, None, True  # Already running
    else:
        cluster_name = f"c{uuid.uuid4().hex[:8]}"
    
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

        # Optionally inject git commit hash so the remote checks out exact code.
        # Only set GIT_COMMIT if explicitly specified; otherwise the remote stays
        # on its current branch (allowing pushes without detached HEAD).
        if instance_config.git_commit:
            envs = sky_config.setdefault('envs', {})
            envs['GIT_COMMIT'] = instance_config.git_commit
            log.info(f"Pinning remote to specified commit: {instance_config.git_commit[:8]}")

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
        
        log.info(f"Cluster name: {cluster_name}")
        log.info("Launching remote job...")
        
        # Launch the task - returns a request_id for async tracking
        request_id = sky.launch(
            task,
            cluster_name=cluster_name,
            down=True,  # Auto tear down after completion
        )

        log.info(f"🚀 Job '{job_name}' launched (request: {request_id})")
        log.info(f"   Cluster: {cluster_name}")
        log.info("   Cluster will auto-terminate after job completes")

        # Wait for cluster to be fully provisioned before returning
        # This prevents double-booking GPUs when launching multiple jobs
        log.info("   Waiting for cluster to be UP...")
        job_id, handle = sky.get(request_id)  # Blocks until cluster is UP and job is submitted
        log.info("   ✅ Cluster is UP, job submitted")
        
        return cluster_name, str(request_id), False  # Newly launched
        
    finally:
        os.chdir(original_cwd)


def execute_config_remotely(
    instance_config: InstanceConfig,
    config: Config,
) -> str:
    """Execute a config remotely using SkyPilot (fire and forget).
    
    Config is saved locally (if save_config_path is set) and logs are streamed
    back to the local machine (if log_file_path is set).
    
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
    
    # Save config locally (before launching remote job)
    local_log_file_path = config_dict.pop('log_file_path', None)
    local_save_config_path = config_dict.pop('save_config_path', None)
    
    if local_save_config_path:
        save_config_locally(config_dict, local_save_config_path, config.name)
    
    # Replace RUN_NAME in instance name if present
    if instance_config.name:
        instance_config.name = instance_config.name.replace("RUN_NAME", config.name)
    
    # Build the experiment run command (config sent to remote has no log/save paths)
    experiment_run_cmd = build_experiment_run_command(
        config_dict=config_dict,
        rel_cwd=rel_cwd,
        job_name=config.name,
        current_branch=current_branch,
        commit_paths=instance_config.commit,
        git_user_name=git_user_name,
        git_user_email=git_user_email,
    )
    
    # Launch the remote job
    cluster_name, request_id, was_already_running = launch_remote_job(
        instance_config=instance_config,
        job_name=config.name,
        run_command=experiment_run_cmd,
    )
    
    # Start streaming logs to local file (if configured and job was actually launched)
    if local_log_file_path and request_id and not was_already_running:
        actual_log_path = local_log_file_path.replace("RUN_NAME", config.name)
        start_log_streaming(request_id, cluster_name, actual_log_path)
    
    return cluster_name


def execute_sweep_remotely(
    instance_config: InstanceConfig,
    sweep_dict: dict,
    sweep_name: str,
    log_file_path: Optional[str] = None,
    save_config_path: Optional[str] = None,
) -> str:
    """Execute a wandb sweep remotely using SkyPilot (fire and forget).
    
    Sweep config is saved locally (if save_config_path is set) and logs are 
    streamed back to the local machine (if log_file_path is set).
    
    Args:
        instance_config: Instance configuration specifying sky_config and patches
        sweep_dict: Sweep configuration dictionary
        sweep_name: Name of the sweep for logging
        log_file_path: Optional local path to stream logs to (may contain RUN_NAME)
        save_config_path: Optional local path to save sweep config (may contain RUN_NAME)
        
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
    
    # Save sweep config locally (before launching remote job)
    if save_config_path:
        save_config_locally(sweep_dict, save_config_path, sweep_name)
    
    # Replace RUN_NAME in instance name if present
    if instance_config.name:
        instance_config.name = instance_config.name.replace("RUN_NAME", sweep_name)
    
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
    
    # Launch the remote job
    cluster_name, request_id, was_already_running = launch_remote_job(
        instance_config=instance_config,
        job_name=sweep_name,
        run_command=sweep_run_cmd,
    )
    
    # Start streaming logs to local file (if configured and job was actually launched)
    if log_file_path and request_id and not was_already_running:
        actual_log_path = log_file_path.replace("RUN_NAME", sweep_name)
        start_log_streaming(request_id, cluster_name, actual_log_path)
    
    return cluster_name
