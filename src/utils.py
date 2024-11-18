import time
import argparse
from typing import Any

from ddp_utils import setup_logger


def exec_time(func):
    """Decorates a function to measure its execution time in hours and minutes."""

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time

        logger = kwargs.get("logger")  # Get logger from kwargs
        if not logger:  # Find logger in positional arguments
            for arg in args:
                if isinstance(arg, type(setup_logger())):
                    logger = arg
                    break

        if logger:
            logger.success(
                f"Training completed ({int(execution_time // 3600)}h {int((execution_time % 3600) // 60)}min)"
            )
        return result

    return wrapper


def add_dataclass_args(parser: argparse.ArgumentParser, dataclass_type: Any):
    for field_info in dataclass_type.__dataclass_fields__.values():
        # Skip properties (those methods marked with @property)
        if isinstance(field_info.type, property):
            continue

        # bool type
        if field_info.type is bool:
            parser.add_argument(
                f"--{field_info.name}",
                action="store_true" if not field_info.default else "store_false",
                help=f"Set {field_info.name} to {not field_info.default}",
            )
        # tuple, list, float
        elif isinstance(field_info.default, tuple):
            parser.add_argument(
                f"--{field_info.name}",
                type=lambda x: tuple(map(float, x.split(","))),
                default=field_info.default,
                help=f"Set {field_info.name} to a tuple of floats (e.g., 0.9,0.999)",
            )
        elif isinstance(field_info.default, list):
            parser.add_argument(
                f"--{field_info.name}",
                type=lambda x: list(map(float, x.split(","))),
                default=field_info.default,
                help=f"Set {field_info.name} to a list of floats (e.g., 0.1,0.2,0.3)",
            )
        else:
            parser.add_argument(
                f"--{field_info.name}",
                type=field_info.type,
                default=field_info.default,
                help=f"Set {field_info.name} to a value of type {field_info.type.__name__}",
            )
