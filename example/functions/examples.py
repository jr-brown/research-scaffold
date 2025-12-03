"""Example functions for research-scaffold demonstrations"""

import logging
import random
import time
from typing import Optional
from pathlib import Path
import wandb

log = logging.getLogger(__name__)


def example_sleep(
    sleep_seconds: int = 300,
    message: str = "Sleeping...",
) -> None:
    """Simple function that sleeps for a given time. Useful for testing remote execution."""
    
    log.info(f"Starting sleep: {message}")
    log.info(f"  sleep_seconds: {sleep_seconds}")
    
    for i in range(sleep_seconds):
        if i % 60 == 0:
            log.info(f"  {i}/{sleep_seconds} seconds elapsed...")
        time.sleep(1)
    
    log.info(f"Sleep complete: {message}")


def example_simple(
    param_a: str,
    param_b: int,
    param_c: bool = True,
) -> None:
    """Simple example function demonstrating basic usage"""
    
    log.info("Running example_simple")
    log.info(f"  param_a: {param_a}")
    log.info(f"  param_b: {param_b}")
    log.info(f"  param_c: {param_c}")
    
    # In a real experiment, this would be your training/evaluation code
    result = param_b * 2 if param_c else param_b
    log.info(f"  result: {result}")


def example_with_logging(
    experiment_name: str,
    iterations: int,
    learning_rate: float = 0.01,
    rng_seed: int = 42,
) -> None:
    """Example showing logging and RNG usage"""
    
    log.info(f"Starting experiment: {experiment_name}")
    log.info(f"  iterations: {iterations}")
    log.info(f"  learning_rate: {learning_rate}")
    log.info(f"  rng_seed: {rng_seed}")
    
    random.seed(rng_seed)
    
    for i in range(iterations):
        value = random.random() * learning_rate
        log.debug(f"Iteration {i}: {value}")
    
    log.info("Experiment complete")


def example_composition(
    base_param: str,
    patch_param: str,
    optional_param: str = "default",
) -> None:
    """Example for testing config composition"""
    
    log.info("Running example_composition")
    log.info(f"  base_param: {base_param}")
    log.info(f"  patch_param: {patch_param}")
    log.info(f"  optional_param: {optional_param}")


def example_sweep(
    learning_rate: float,
    batch_size: int,
    model_size: int = 128,
    output_dir: str = "outputs/",
    model: Optional[dict] = None,
    optimizer: Optional[dict] = None,
) -> None:
    """Example for sweep demonstrations"""
    
    log.info("Running example_sweep")
    log.info(f"  learning_rate: {learning_rate}")
    log.info(f"  batch_size: {batch_size}")
    log.info(f"  model_size: {model_size}")
    log.info(f"  output_dir: {output_dir}")
    
    # Handle nested parameters if provided
    model_dropout = 0.0  # Default value
    if model is not None:
        log.info(f"  model: {model}")
        # Extract nested model parameters if needed
        model_hidden_dim = model.get("hidden_dim", model_size)
        model_dropout = model.get("dropout", 0.0)
        # Use model parameters in metric calculation
        model_size = model_hidden_dim
    
    optimizer_momentum = 0.9  # Default value
    if optimizer is not None:
        log.info(f"  optimizer: {optimizer}")
        # Extract nested optimizer parameters if needed
        optimizer_momentum = optimizer.get("momentum", 0.9)
    
    # Simulate training and calculate a metric
    # Include model dropout and optimizer momentum in calculation
    metric_value = (1.0 / learning_rate) + batch_size + model_size + (optimizer_momentum * 10) - (model_dropout * 5)
    log.info(f"  metric_value: {metric_value}")
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create artifact.md file
    artifact_path = output_path / "artifact.md"
    with open(artifact_path, "w") as f:
        f.write("# Sweep Run Artifact\n\n")
        f.write("## Parameters\n\n")
        f.write(f"- **Learning Rate**: {learning_rate}\n")
        f.write(f"- **Batch Size**: {batch_size}\n")
        f.write(f"- **Model Size**: {model_size}\n")
        if model is not None:
            f.write(f"- **Model Dropout**: {model_dropout}\n")
            f.write(f"- **Model Hidden Dim**: {model.get('hidden_dim', model_size)}\n")
        if optimizer is not None:
            f.write(f"- **Optimizer Momentum**: {optimizer_momentum}\n")
        f.write("\n## Results\n\n")
        f.write(f"- **Performance Metric**: {metric_value:.4f}\n")
        f.write(f"- **Wandb Run Name**: {wandb.run.name if wandb.run else 'N/A'}\n")
    
    log.info(f"  Created artifact at: {artifact_path}")
    
    # In a real sweep, you'd log this to wandb
    wandb.log({"performance_metric": metric_value})

