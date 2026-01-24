# L20 compliant helpers - strict types, no magic
import os
from typing import overload


@overload
def getenv(key: str) -> str | None: ...


@overload
def getenv(key: str, default: str) -> str: ...


@overload
def getenv(key: str, default: int) -> int: ...


def getenv(key: str, default: str | int | None = None) -> str | int | None:
    """
    Get environment variable with type-inferred default.
    
    L20: Proper typing, no Any, handles int coercion.
    """
    val = os.environ.get(key)
    if val is None:
        return default
    if isinstance(default, int):
        return int(val)
    return val


def colored(text: str, color: str) -> str:
    """
    ANSI color wrapper for terminal output.
    
    L20: No external deps for simple coloring.
    """
    colors: dict[str, str] = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "reset": "\033[0m",
    }
    code = colors.get(color.lower(), "")
    reset = colors["reset"]
    return f"{code}{text}{reset}"
