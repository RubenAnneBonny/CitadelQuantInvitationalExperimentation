"""
Alpha registry: maps string names to BaseAlpha subclasses.

Usage:
  @register("bollinger")
  class BollingerAlpha(BaseAlpha): ...

  alphas = get_all()   # one instance of every registered alpha
"""
from .base_alpha import BaseAlpha

ALPHA_REGISTRY: dict = {}   # {name: class}


def register(name: str):
    """Decorator that registers a BaseAlpha subclass under a string key."""
    def decorator(cls):
        cls.name = name
        ALPHA_REGISTRY[name] = cls
        return cls
    return decorator


def get_all() -> list:
    """Return a fresh instance of every registered alpha."""
    return [cls() for cls in ALPHA_REGISTRY.values()]


def get(name: str) -> BaseAlpha:
    """Return a fresh instance of the named alpha."""
    if name not in ALPHA_REGISTRY:
        raise KeyError(f"Alpha '{name}' not registered. Available: {list(ALPHA_REGISTRY)}")
    return ALPHA_REGISTRY[name]()
