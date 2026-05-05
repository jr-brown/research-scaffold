# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install
uv sync                               # install all deps (including dev)

# Run experiments
uv run python main.py -c config.yaml         # single config
uv run python main.py -m meta_config.yaml    # meta config (composition, grid search)
uv run python main.py -s sweep.yaml          # wandb sweep

# Test
uv run pytest                                # run all tests
uv run pytest tests/test_config_loading/     # run specific test directory
uv run pytest -k "test_name"                 # run specific test by name
```

## Architecture

**Config-driven experiment execution** with three modes:
1. **Single config** (`-c`): One experiment from a YAML config
2. **Meta config** (`-m`): Multiple experiments with composition and grid search
3. **Sweep config** (`-s`): W&B hyperparameter sweeps

### Core Flow
```
execute_experiments() → load_config/load_meta_config/load_sweep_config
                      → process_meta_config (if meta)
                      → execute_from_config / execute_sweep_from_dict
```

### Key Modules
- `research_scaffold/config_tools.py`: Main execution engine, config loading, composition logic
- `research_scaffold/types.py`: Dataclass definitions (Config, MetaConfig, SweepConfig, etc.)
- `research_scaffold/remote_execution.py`: SkyPilot/Vast.ai cloud execution
- `research_scaffold/util.py`: Logging, dict merging, placeholder substitution

### Config Composition

Configs compose via `config_axes` (cartesian product) and `config_steps` (sequential layering):
```yaml
# Meta config with grid search
common_root: "base.yaml"      # applied first to all
common_patch: "patch.yaml"    # applied last to all
experiments:
  - config_axes:              # cartesian product
      - ["option_a.yaml", "option_b.yaml"]
      - ["small.yaml", "large.yaml"]
```

### Placeholders
- `RUN_NAME`: Replaced with experiment name (timestamped if `time_stamp_name: true`)
- `RUN_GROUP`: Replaced with wandb group (timestamped if `time_stamp_group: true`)
- `SWEEP_NAME`: Replaced with sweep name

Used in paths like `log_file_path: "outputs/RUN_GROUP/output.log"`

`time_stamp_group: true` appends a timestamp to the group name, useful for creating unique output directories per run group:
```yaml
wandb_group: "my_experiment"
time_stamp_group: true
log_file_path: "outputs/RUN_GROUP/output.log"  # → outputs/my_experiment_2025-01-15_14-30-22/output.log
```

### Function Map Pattern

User code registers functions by name for configs to reference:
```python
function_map = {"my_func": my_func}
execute_experiments(function_map=function_map, ...)
```

Config references: `function_name: "my_func"`

### Remote Execution

Remote execution uses SkyPilot (`research_scaffold/remote_execution.py`). Experiments and sweeps can run on cloud GPUs by adding an `instance` block to the config:

```yaml
instance:
  sky_config: "sky_config.yaml"     # SkyPilot task YAML (or set SKY_PATH env var)
  patch: "sky_patch.yaml"           # Optional: override sky_config fields
  name: "my-cluster-RUN_NAME"      # Optional: custom cluster name (supports RUN_NAME placeholder)
  commit:                           # Optional: paths to commit and push after completion
    - "outputs/**"
    - "logs/**"
  git_commit: "abc123"             # Optional: pin remote to a specific git commit
  retry_until_up: true             # Optional: retry provisioning until cluster is up
  managed: true                    # Optional: use SkyPilot managed jobs (fire-and-forget)
```

#### Standard vs Managed Jobs

**Standard** (`managed: false`, default): Calls `sky.launch()`, blocks until the cluster is UP, then streams logs. Good for interactive use where you want to wait for results.

**Managed** (`managed: true`): Calls `sky.jobs.launch()`, returns immediately (fire-and-forget). SkyPilot's jobs controller handles lifecycle, automatic teardown, and spot recovery. Good for submitting many experiments at once (e.g. via meta-configs). Monitor with:
```bash
sky jobs queue              # check status
sky jobs logs <job-name>    # stream logs
sky jobs cancel <job-id>    # cancel a job
```

#### RunPod Notes

When using RunPod with managed jobs, SkyPilot provisions a CPU-only "jobs controller" pod. RunPod CPU pods have a max disk size of 40 GB, but SkyPilot defaults to 50 GB. To fix this, create a `.sky.yaml` in the project root:
```yaml
jobs:
  controller:
    resources:
      disk_size: 40
```

If `git_commit` is set, it is injected as a `GIT_COMMIT` environment variable into the SkyPilot task and the remote checks out that exact commit (detached HEAD). If unset, `GIT_COMMIT` is not set and the remote stays on its current branch (allowing pushes). The sky config's `run:` block should handle both cases:

```yaml
run: |
  # Load environment variables
  set -a
  source ~/sky_workdir/.env
  set +a

  # Activate the uv-managed venv
  source ~/sky_workdir/.venv/bin/activate

  # Checkout the exact commit pinned at launch time (injected by research_scaffold)
  cd ~/sky_workdir
  git fetch origin
  if [ -n "${GIT_COMMIT}" ]; then
    git checkout "${GIT_COMMIT}"
    echo "Checked out pinned commit: $(git rev-parse --short HEAD)"
  else
    git pull origin $(git rev-parse --abbrev-ref HEAD) --ff-only
    echo "Pulled latest: $(git rev-parse --short HEAD)"
  fi

  # Experiment command is injected here by remote_execution.py
```

## Testing

Tests use pytest with mocked wandb. Test fixtures in `tests/conftest.py` mock wandb.init, wandb.sweep, and wandb.agent.

## Key Dependencies
- wandb: Experiment tracking and sweeps
- beartype: Runtime type checking (applied via `beartype_this_package()` in `__init__.py`)
- skypilot[vast]: Cloud GPU execution (Vast.ai)
- skypilot[runpod]: Cloud GPU execution (RunPod)
- pyyaml: Config parsing
