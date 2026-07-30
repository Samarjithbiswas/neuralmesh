"""The compared architectures and the blocks they are built from."""

from .architectures import (
    ARCHITECTURES,
    MeshGraphNet,
    MeshGraphTransformer,
    ModelConfig,
    NodeMLP,
    build_model,
    count_parameters,
    match_capacity,
)
from .blocks import MLP, MessagePassingBlock, PhysicsAttention, scatter_mean, scatter_sum

__all__ = [
    "ARCHITECTURES",
    "MLP",
    "MeshGraphNet",
    "MeshGraphTransformer",
    "MessagePassingBlock",
    "ModelConfig",
    "NodeMLP",
    "PhysicsAttention",
    "build_model",
    "count_parameters",
    "match_capacity",
    "scatter_mean",
    "scatter_sum",
]
