import logging

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False


log = logging.getLogger(__name__)


def example_simple_config(
    dummy_str: str,
    dummy_int: int,
    dummy_bool: bool,
    dummy_str_default: str = "dummy_str_default",
) -> None:

    print("Testing simple config")

    assert isinstance(dummy_str, str)
    assert isinstance(dummy_int, int)
    assert isinstance(dummy_bool, bool)
    assert isinstance(dummy_str_default, str)

    print(f"{dummy_str=}")
    print(f"{dummy_int=}")
    print(f"{dummy_bool=}")
    print(f"{dummy_str_default=}")


def example_log_levels() -> None:
    log.debug("Log debug test")
    log.info("Log info test")
    log.warning("Log warn test")
    log.error("Log error test")
    log.critical("Log critical test")


def example_sweep_function(
    dummy_str: str,
    dummy_int: int,
    dummy_bool: bool = True,
    dummy_str_default: str = "default",
) -> None:
    """Example function for testing sweeps - logs a metric based on parameters."""
    
    print("Testing sweep function")
    print(f"{dummy_str=}, {dummy_int=}, {dummy_bool=}, {dummy_str_default=}")
    
    # Calculate a dummy metric based on parameters
    # In a real use case, this would be your actual training/evaluation
    metric_value = dummy_int * 2.0
    if dummy_str == "foo":
        metric_value += 5.0
    elif dummy_str == "bar":
        metric_value += 10.0
    elif dummy_str == "baz":
        metric_value += 15.0
    
    if dummy_bool:
        metric_value *= 1.5
    
    print(f"Calculated metric: {metric_value}")
    
    # Log to wandb if available
    if has_wandb:
        wandb.log({"dummy_metric": metric_value})
        log.info(f"Logged dummy_metric={metric_value} to wandb")
    else:
        log.warning("wandb not available, metric not logged")
