# Research Scaffold

Run experiments from YAML configs with support for composition, sweeps, and remote execution.

## Usage

```bash
uv sync                                      # install all deps (including dev)
uv run python main.py -c config.yaml        # single experiment
uv run python main.py -m meta_config.yaml   # multiple experiments, composition, grid search
uv run python main.py -s sweep.yaml         # wandb hyperparameter sweep
```

The `-c` flag accepts config files, inline dicts, or config paths. You can also pass multiple configs that get composed together.

## Features

**Config composition** - Layer multiple configs or run grid searches over parameter combinations

**Sweeps** - Run wandb hyperparameter searches with `-s`

**Remote execution** - Add an `instance` config to run on cloud GPUs via SkyPilot:
```yaml
instance:
  sky_config: "path/to/sky_config.yaml"  # or set SKY_PATH env var
  patch:                                  # optional sky config overrides
    resources:
      accelerators: "V100:1"
  commit:                                 # paths to commit/push after run
    - "outputs/**"
  git_commit: "abc123def"                # optional: pin to a specific commit (detaches HEAD)
```

If `git_commit` is specified, the remote checks out that exact commit via a `GIT_COMMIT` environment variable injected into the SkyPilot task. If omitted, the remote stays on its current branch (allowing pushes).

See `example/` for more.
