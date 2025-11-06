# Using Wandb Sweeps with Research Scaffold

This guide explains how to use the new sweep functionality in research-scaffold.

## Basic Usage

Run a sweep using the `-s` flag:

```bash
python main.py -s sweep_configs/simple_sweep.yaml
```

## Sweep Config Format

Sweep configs extend standard wandb sweep configs with custom fields:

```yaml
# Custom fields (research-scaffold specific)
base_config: "configs/sweep_base.yaml"  # Base configuration
sweep_count: 5                           # Number of sweep runs (optional)
project: "my-project"                    # Wandb project (overrides base_config)

# Standard wandb sweep configuration
method: bayes  # or 'grid', 'random'
metric:
  name: my_metric_name
  goal: maximize  # or 'minimize'

parameters:
  learning_rate:
    distribution: log_uniform_values
    min: 0.0001
    max: 0.1
  batch_size:
    values: [32, 64, 128]
```

## How It Works

1. **Base Config**: Provides default parameters and configuration
   - Can be any regular config file
   - Specifies `function_name`, base `function_kwargs`, etc.

2. **Sweep Parameters**: Override base config parameters
   - Parameters in the sweep config replace those in base_config
   - Non-swept parameters from base_config are preserved

3. **Execution**: 
   - Creates a wandb sweep with the specified method (bayes, grid, random)
   - Runs `sweep_count` iterations (or infinite if not specified)
   - Each run logs to wandb automatically

## Example: Hyperparameter Search

**Base Config** (`configs/training_base.yaml`):
```yaml
name: "my_experiment"
function_name: "train_model"

function_kwargs:
  model_type: "resnet"
  num_epochs: 100
  learning_rate: 0.001  # Will be swept
  batch_size: 64        # Will be swept
  optimizer: "adam"
```

**Sweep Config** (`sweep_configs/hp_search.yaml`):
```yaml
base_config: "configs/training_base.yaml"
sweep_count: 50
project: "my-ml-project"

method: bayes
metric:
  name: val_accuracy
  goal: maximize

parameters:
  learning_rate:
    distribution: log_uniform_values
    min: 0.0001
    max: 0.1
  batch_size:
    values: [32, 64, 128, 256]
```

**Run it**:
```bash
python main.py -s sweep_configs/hp_search.yaml
```

## Requirements

Your training function must:
1. Accept parameters that will be swept as keyword arguments
2. Log metrics using `wandb.log()` that match the metric name in sweep config

Example:
```python
def train_model(learning_rate, batch_size, **kwargs):
    # Your training code here
    for epoch in range(num_epochs):
        # ... training ...
        accuracy = evaluate()
        wandb.log({"val_accuracy": accuracy})  # Must match metric.name
```

## Advanced Features (Coming Soon)

- Using meta-configs as base_config
- Running multiple sweeps from meta-configs
- Sweep-specific logging and checkpointing

