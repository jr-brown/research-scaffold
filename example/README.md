# Research Scaffold Examples

## Features

### Basic
- **Single config** → `python main.py -c basic/simple.yaml`
- **Logging + RUN_NAME** → `python main.py -c basic/with_logging.yaml`

### Composition
- **Config layering** → `python main.py -m composition/composed.yaml`
- **Common root/patch** → `python main.py -m composition/meta_with_common.yaml`
- **Grid search (axes)** → `python main.py -m composition/grid_search.yaml`

### Sweeps
- **Basic sweep** → `python main.py -s sweeps/sweep_configs/basic_sweep.yaml`
- **Nested parameters** → `python main.py -s sweeps/sweep_configs/nested_sweep.yaml`
- **With composition** → `python main.py -m sweeps/meta_configs/sweep_with_composition.yaml`
- **Multiple sweeps** → `python main.py -m sweeps/meta_configs/multi_sweep.yaml`
- **Sweep with logging** → `python main.py -s sweeps/sweep_configs/sweep_with_logging.yaml`

### Remote Execution
Set `export SKY_PATH="example/sky_config.yaml"` before running:
- **Remote config** → `python main.py -c basic/with_remote_env.yaml`
- **Remote sweep** → `python main.py -s sweeps/sweep_configs/remote_sweep.yaml`

## CLI Options

- `-c` config - Single experiment
- `-m` meta-config - Multiple experiments  
- `-s` sweep - Wandb hyperparameter search
