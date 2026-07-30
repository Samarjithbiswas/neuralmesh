"""neuralmesh: does global attention fix under-reaching in mesh graph networks?

A message-passing network with :math:`L` layers can only see :math:`L` hops. For an
elliptic PDE the solution at an interior node depends on *every* boundary value, so
once the graph diameter exceeds :math:`L` the architecture cannot represent the
solution no matter how long it trains. That is under-reaching, and this package is a
controlled measurement of it.

The point of the package is the controls, not the model:

* parameter counts are matched across architectures, so the comparison measures
  design rather than capacity
* a no-communication model is included as a floor, so any result it also reaches is
  not evidence about message passing
* error is reported by distance from the driven boundary, because the prediction is
  worse error *in the middle*, not uniformly worse

Ground truth comes from a P1 finite element solver in this repository, verified to
converge at second order, so the labels are not themselves a black box.

Quick start
-----------
>>> from neuralmesh import rectangle_mesh, solve_poisson
>>> mesh = rectangle_mesh(12, 12, jitter=0.0)
>>> sol = solve_poisson(mesh, source=1.0, dirichlet_value=0.0)
>>> sol.u.shape == (mesh.n_nodes,)
True
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Samarjith Biswas"
__license__ = "MIT"

from .data.dataset import (
    PARAM_BOUNDS,
    PARAM_NAMES,
    Dataset,
    Normaliser,
    generate_dataset,
    latin_hypercube,
    load_or_generate,
    make_case,
    scale_to_bounds,
)
from .evaluate.underreach import (
    ArchResult,
    UnderReachResult,
    run_underreach_study,
    save_result,
    strip_dataset,
)
from .fem.poisson import (
    PoissonSolution,
    apply_dirichlet,
    assemble_load,
    assemble_mass,
    assemble_stiffness,
    shape_gradients,
    solve_poisson,
)
from .mesh.geometry import (
    TriMesh,
    annulus_mesh,
    boundary_nodes_of,
    rectangle_mesh,
    refine,
    strip_mesh,
    unit_square_mesh,
)
from .mesh.graph import (
    MeshGraph,
    build_graph,
    cell_to_node,
    graph_diameter,
    graph_from_solution,
)
from .models.architectures import (
    ARCHITECTURES,
    MeshGraphNet,
    MeshGraphTransformer,
    ModelConfig,
    NodeMLP,
    build_model,
    count_parameters,
    match_capacity,
)
from .models.blocks import MLP, MessagePassingBlock, PhysicsAttention
from .train.trainer import (
    History,
    TensorGraph,
    TrainConfig,
    evaluate,
    masked_mse,
    relative_l2,
    train_model,
)

__all__ = [
    "__version__",
    # mesh
    "TriMesh",
    "unit_square_mesh",
    "rectangle_mesh",
    "strip_mesh",
    "annulus_mesh",
    "refine",
    "boundary_nodes_of",
    "MeshGraph",
    "build_graph",
    "graph_from_solution",
    "graph_diameter",
    "cell_to_node",
    # fem
    "PoissonSolution",
    "solve_poisson",
    "assemble_stiffness",
    "assemble_mass",
    "assemble_load",
    "apply_dirichlet",
    "shape_gradients",
    # models
    "ModelConfig",
    "NodeMLP",
    "MeshGraphNet",
    "MeshGraphTransformer",
    "ARCHITECTURES",
    "build_model",
    "count_parameters",
    "match_capacity",
    "MLP",
    "MessagePassingBlock",
    "PhysicsAttention",
    # data
    "Dataset",
    "Normaliser",
    "PARAM_BOUNDS",
    "PARAM_NAMES",
    "make_case",
    "latin_hypercube",
    "scale_to_bounds",
    "generate_dataset",
    "load_or_generate",
    # training
    "TrainConfig",
    "History",
    "TensorGraph",
    "train_model",
    "evaluate",
    "masked_mse",
    "relative_l2",
    # the experiment
    "run_underreach_study",
    "strip_dataset",
    "UnderReachResult",
    "ArchResult",
    "save_result",
]
