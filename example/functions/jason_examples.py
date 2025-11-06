import logging


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
