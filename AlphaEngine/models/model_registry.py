"""
Model registry: maps string names to BaseModel subclasses.

Usage:
  @model_register("bollinger")
  class BollingerModel(BaseModel): ...

  model = get_model("bollinger")
"""
from .base_model import BaseModel

MODEL_REGISTRY: dict = {}   # {name: class}


def model_register(name: str):
    """Decorator that registers a BaseModel subclass under a string key."""
    def decorator(cls):
        cls.model_name = name
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def get_model(name: str) -> BaseModel:
    """Instantiate a registered model by name."""
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Model '{name}' not registered. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name]()
