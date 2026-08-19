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

# Dry run: print the fully composed config(s) that would run, then exit
uv run python main.py -c config.yaml -d       # works with -c, -m, and -s

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
execute_experiments() → build_configs()  # detects config type, loads, composes
                      → ResolvedExperiments(configs, sweep_dicts, parallel)
                      → execute_from_config / execute_sweep_from_dict
                        (or dry_run_report, with -d)
```

`build_configs()` is the single resolution path, so a `-d` dry run cannot disagree with a
real run. It returns fully composed configs; nothing is executed until the caller acts on them.

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

#### Replacing a Block Instead of Merging Into It

Nested dicts merge key by key by default. A block containing `<replace>: true` discards the
inherited block entirely instead, taking no keys from it at any depth:

```yaml
# earlier in the chain
side_task:
  type: rolling_mod_sum
  n_samples: 10000
  modulus: 3

# later in the chain
side_task:
  <replace>: true
  type: knights_and_knaves
  n_people: 3

# result: {type: knights_and_knaves, n_people: 3}
```

The marker is stripped from the composed config. Use it for tagged-union blocks, where the
valid keys depend on a `type` field and inherited keys from a different variant are never
correct. At the top level of a config file it discards everything earlier in the chain.

### Parallel Local Execution

A meta config can run its composed configs concurrently in separate processes:

```yaml
parallel: true
max_concurrent: 2         # Optional: how many run at once (unset = no limit, all at once)
start_method: "spawn"     # Optional: fork on Linux, spawn elsewhere
experiments: [...]
```

`parallel: true` is ignored when any config has an `instance` block — use `managed: true` for
parallel remote execution instead.

**`max_concurrent`** bounds how many configs run simultaneously. Leaving it unset means no limit —
every config starts at once. Set it when the runs contend for a resource, e.g. `max_concurrent: 2`
for GPU fine-tuning jobs that would otherwise all grab memory at the same time.

**`start_method`** picks the multiprocessing start method. The default is `fork` on Linux and
`spawn` everywhere else — `fork` is unsafe on macOS once a worker touches Metal (MPS runs either
segfault or fail with "Failed to created pipeline state object"). Override it only to force a
specific method.

Under `spawn`, `function_map` is pickled, so worker functions must be importable at module level.
Functions imported from a module, `functools.partial` objects, and callable class instances all
work; functions defined *inside* another function and lambdas do not, and fail immediately at
launch with `Can't pickle local object`. Each worker also re-imports the caller's `main.py`, so it
needs an `if __name__ == "__main__":` guard and module-level work runs once per config.

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

#### `vast_filters`

`vast_filters` is a sky-config key (not a SkyPilot field, and not part of `instance`): a raw
Vast offer-query fragment. The scaffold strips it from the task YAML — SkyPilot's schema would
reject the unknown key — and writes it verbatim to `~/.sky/vast_filters/<cluster_name>`, where
provisioning-side tooling picks it up. The string is opaque to the scaffold: no parsing, no
validation. An empty string is treated as unset.

Since the fragment narrows the offer pool, an over-restrictive one surfaces as a provisioning
failure (`Failed to acquire resources`), not as a scaffold error.

```yaml
# shared sky_config.yaml
vast_filters: "cuda_max_good>=13.0 gpu_ram>=40"

# per-experiment instance.patch — overrides just this key via the usual merge
vast_filters: "cuda_max_good>=13.0 gpu_ram>=130"
```

#### `sync` and the Sync-Back Watcher

`instance.sync` lists folders to rsync back when the job finishes. A watcher process per
launch tails the job log, rsyncs those folders, then tears the cluster down.

`sky.tail_logs` returns when the job exits — but it also returns normally, with no exception,
when the stream drops under API-server load. So the watcher **only tears down after a
confirmed terminal job status**. A status check that fails, comes back empty, or yields a
`None` state means "could not determine", never "finished": the watcher backs off and
re-tails. If it runs out of re-tails it still syncs whatever exists, but leaves the cluster
up for `SYNC_AUTOSTOP_MINUTES` autostop to reclaim once it is genuinely idle.

This matters because one API-server hiccup hits every watcher that is mid-drop at the same
moment, so a naive "stream ended → done" rule scales its damage with the number of
concurrent runs.

Note the watcher script is baked into `python -c` at launch time, so **already-running
watchers keep the code they started with** — after upgrading, restart long-lived watchers
to pick up changes.

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
