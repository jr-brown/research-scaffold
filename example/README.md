# Research Scaffold Examples

## Features

### Basic
- **Single config** → `uv run python main.py -c basic/simple.yaml`
- **Logging + RUN_NAME** → `uv run python main.py -c basic/with_logging.yaml`

### Composition
- **Config layering** → `uv run python main.py -m composition/composed.yaml`
- **Common root/patch** → `uv run python main.py -m composition/meta_with_common.yaml`
- **Grid search (axes)** → `uv run python main.py -m composition/grid_search.yaml`

### Sweeps
- **Basic sweep** → `uv run python main.py -s sweeps/sweep_configs/basic_sweep.yaml`
- **Nested parameters** → `uv run python main.py -s sweeps/sweep_configs/nested_sweep.yaml`
- **With composition** → `uv run python main.py -m sweeps/meta_configs/sweep_with_composition.yaml`
- **Multiple sweeps** → `uv run python main.py -m sweeps/meta_configs/multi_sweep.yaml`
- **Sweep with logging** → `uv run python main.py -s sweeps/sweep_configs/sweep_with_logging.yaml`

### Remote Execution
Set `export SKY_PATH="example/sky_config.yaml"` before running:
- **Remote config** → `uv run python main.py -c basic/with_remote_env.yaml`
- **Remote sweep** → `uv run python main.py -s sweeps/sweep_configs/remote_sweep.yaml`

## CLI Options

- `-c` config - Single experiment
- `-m` meta-config - Multiple experiments  
- `-s` sweep - Wandb hyperparameter search
