# shared utils, tinygrad style
import os, functools
from typing import TypeVar, Any
from contextvars import ContextVar as _ContextVar

T = TypeVar('T')

@functools.lru_cache(maxsize=None)
def getenv(key: str, default: T = "") -> T:  # cached env lookup
  return type(default)(os.environ.get(key, default)) if default != "" else os.environ.get(key)

def colored(text: str, color: str) -> str:  # terminal colors
  colors = {"red": "\033[91m", "green": "\033[92m", "yellow": "\033[93m", "blue": "\033[94m", "reset": "\033[0m"}
  return f"{colors.get(color, '')}{text}{colors['reset']}"

class ContextVar:  # debug flag pattern from tinygrad
  _cache: dict[str, _ContextVar] = {}
  def __new__(cls, key: str, default: Any = None):
    if key in cls._cache: return cls._cache[key]
    instance = super().__new__(cls)
    instance.ctx = _ContextVar(key, default=default)
    cls._cache[key] = instance
    return instance
  def __init__(self, key: str, default: Any = None): self.key, self.default = key, default
  def __enter__(self): self._token = self.ctx.set(True); return self
  def __exit__(self, *_): self.ctx.reset(self._token)
  @property
  def value(self) -> Any: return self.ctx.get()

DEBUG = ContextVar("DEBUG", False)

# dense one-liners
def prod(x): return functools.reduce(lambda a, b: a * b, x, 1)
def dedup(x): return list(dict.fromkeys(x))
def flatten(x): return [item for sublist in x for item in (sublist if isinstance(sublist, list) else [sublist])]
def load_csv(path: str):
  import pandas as pd
  return pd.read_csv(path)
