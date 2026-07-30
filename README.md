# neuralmesh

**Does global attention actually fix under-reaching in mesh graph networks, or does it
just add parameters?**

[![CI](https://github.com/Samarjithbiswas/neuralmesh/actions/workflows/ci.yml/badge.svg)](https://github.com/Samarjithbiswas/neuralmesh/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CPU%20only-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-34d399)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-72-34d399)](tests/)

A message-passing network with `L` layers can only see `L` hops. For an elliptic PDE
such as steady diffusion, the solution at an interior node depends on **every**
boundary value, so once the graph diameter exceeds `L` the architecture is
structurally incapable of representing the solution. Not undertrained. Incapable. This
is called **under-reaching**, and no amount of width or epochs fixes it.

Interleaving global attention between message-passing blocks removes the dependence of
receptive field on depth. This repository measures whether that helps, with the
controls needed for the answer to mean anything.

## The claim, and why it needs controls

Plenty of papers report that adding attention improves a graph network. Most of those
comparisons cannot distinguish four different explanations:

1. attention genuinely fixed a reach problem
2. attention added parameters, and any extra capacity would have helped
3. the task never needed long-range information in the first place
4. the error got smaller everywhere, which is not what under-reaching predicts

So this repository does four things differently:

| Control | Why it exists |
|---|---|
| **Parameter counts are matched.** The transformer's width is reduced until it lands within a few percent of the baseline. | Otherwise the experiment measures capacity, not architecture. Rules out (2). |
| **A no-communication model is included.** `NodeMLP` cannot see any other node at all. | Any error level it also reaches was never about the graph. Rules out (3). |
| **Error is reported by distance from the driven boundary.** | Under-reaching predicts worse error *in the middle*, not uniform degradation. Rules out (4). |
| **Ground truth comes from a solver in this repo, verified to second order.** | If the labels were wrong, every learned number would be measuring the wrong thing and still look fine. |

The domain is a long thin strip, because that makes graph diameter grow while node
count stays affordable, and the only long-range driver is the difference between the
left and right Dirichlet values. A model that cannot carry information along the strip
cannot get the interior right.

## Results

Three strip domains of increasing aspect ratio, so graph diameter grows from 20 to 80
while everything else is held fixed: 49 training cases, 90 epochs, hidden width 64,
identical seeds. Relative `L2` error on the test split, interior nodes only.

Reproduce with `python examples/run_underreach.py --sweep 4 8 16 --samples 70 --epochs 90`.
Raw output is committed under [`benchmarks/`](benchmarks/).

| diameter | control (no comms) | MeshGraphNet L=4 | MeshGraphNet L=16 | **MGN-Transformer L=4** |
|---:|---:|---:|---:|---:|
| | 17,281 params | 130,113 params | 480,321 params | **130,981 params** |
| | reach 0 hops | reach 4 hops | reach 16 hops | **global** |
| 20 | 0.9750 | 0.1811 | 0.1202 | **0.1054** |
| 40 | 0.9666 | 0.1627 | 0.1136 | **0.0974** |
| 80 | 0.9046 | 0.1666 | 0.1329 | **0.0961** |

The parameter-matched comparison is the middle two columns against the last one:
130,113 versus 130,981 is a 0.7% difference in capacity.

**What holds up.** Against the parameter-matched baseline, interleaving global
attention reduces error by **42%, 40% and 42%** at the three diameters. That
consistency across a 4x change in diameter is the part worth trusting. Attention also
beats the deep baseline that has **3.7x more parameters and four times the depth**, at
every diameter. And the no-communication control sits at 0.90 to 0.98 throughout, which
confirms the task genuinely requires propagating boundary information: any architecture
scoring near 1.0 has learned nothing but the mean.

**What does not hold up, and I am not going to bury it.** The naive prediction is that
a reach-limited model should get *worse as diameter grows*. In the aggregate column it
does not. MeshGraphNet L=4 scores 0.181, 0.163, 0.167 as diameter goes 20, 40, 80.
That is flat. Taken alone, this table does not demonstrate under-reaching at all.

**Where the effect actually shows up.** It is only visible once error is resolved by
distance from the driven boundary. Splitting the interior into four bands, band 0
nearest a driven edge and band 3 at mid-strip:

| deepest band (mid-strip) | d=20 | d=40 | d=80 | change |
|---|---:|---:|---:|---:|
| MeshGraphNet L=4 | 0.195 | 0.247 | 0.288 | **+48%** |
| MeshGraphNet L=16 | 0.162 | 0.231 | 0.299 | **+85%** |
| MGN-Transformer L=4 | 0.222 | 0.232 | 0.219 | **-1%** |

Both purely local models degrade in the middle of the domain as the domain gets longer.
The global model does not: its deep-interior error is essentially independent of
diameter. That is the under-reaching signature, and it is invisible in the aggregate
metric that most comparisons report.

Note that L=16 degrades *more* in the deep interior than L=4 despite far greater depth
and capacity, which is consistent with 16 hops still being well short of an 80-hop
diameter.

**One more honest wrinkle.** The transformer is not uniformly better band by band. It is
clearly best nearest the boundary (0.045 to 0.048 against 0.070 to 0.113) but it is
consistently *worse* than the shallow baseline in band 2 at all three diameters. So the
aggregate win is real and reproducible, but it is not the case that attention improves
every region. I do not have a confident explanation for the band 2 behaviour and would
want more seeds before theorising about it.

**Caveats that belong next to these numbers.** One seed per configuration, so the
42/40/42% agreement is suggestive rather than an error bar. 49 training cases is small.
90 epochs is short enough that all four models are still improving. This is 2D scalar
steady diffusion. The result is about architecture ranking under identical budgets, not
about absolute accuracy.


## The solver is verified, not assumed

Every learned number above is measured against a P1 finite element solver in
`neuralmesh/fem/`. That solver is checked against answers known in advance, and you
can reproduce the check in one command:

```bash
neuralmesh verify
```

```
finite element verification
--------------------------------------------------------------
  n=9    nodes=81     h=0.1250   L2 error 1.658e-02
  n=17   nodes=289    h=0.0625   L2 error 4.298e-03
  n=33   nodes=1089   h=0.0312   L2 error 1.084e-03
  observed convergence rates: [1.947, 1.987]
  PASS  second-order convergence (theory: 2)
  PASS  linear field reproduced exactly (max error 1.33e-15)
--------------------------------------------------------------
solver verified
```

Two independent checks, both with a known answer:

- **Convergence rate.** A manufactured solution `u = sin(pi x) sin(pi y)` with the
  matching source. Theory says the mass-weighted L2 error falls as `h^2`. The measured
  rates are 1.947 and 1.987. A bug in the stiffness assembly almost always breaks the
  rate even when it leaves the solution looking plausible.
- **Exactness on linear fields.** A P1 space contains every linear function, so
  `u = 2x - 3y + 1` must be reproduced to round-off rather than to discretisation
  error. Measured max error 1.3e-15.

The test suite adds the properties that catch sign and indexing errors: the stiffness
matrix is symmetric, it annihilates constants, the mass matrix sums exactly to the
domain area, the response scales linearly with the load, and a positive source with
zero boundary data lifts the whole interior (the maximum principle).

## Install

```bash
git clone https://github.com/Samarjithbiswas/neuralmesh.git
cd neuralmesh
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
pytest -q
```

CPU only, deliberately. Every number in this README is reproducible on a laptop, and
pinning a CUDA build would make that false for most readers.

## Use it

```bash
neuralmesh verify                              # check the solver
neuralmesh solve --nx 24 --ny 8 --width 6      # one case, with graph diameter
neuralmesh dataset --samples 200               # generate and cache training data
neuralmesh train --arch mesh_graph_transformer --epochs 120
neuralmesh study --aspect 8 --out results.json # the comparison
```

As a library:

```python
from neuralmesh import (
    strip_mesh, solve_poisson, graph_from_solution, graph_diameter,
    build_model, ModelConfig, count_parameters,
)

mesh = strip_mesh(length=8.0, height=1.0, ny=6)
sol = solve_poisson(mesh, source=1.0, dirichlet_value=0.0)
graph = graph_from_solution(sol)

print(graph.summary(), "diameter", graph_diameter(graph))

model = build_model("mesh_graph_transformer", ModelConfig(hidden_dim=64, n_blocks=4))
print(count_parameters(model), "parameters, reach", model.receptive_hops)
```

`receptive_hops` returns `0` for the control, `n_blocks` for pure message passing, and
`-1` meaning global once any attention layer is present.

## Reproduce the study

```bash
python examples/run_underreach.py --quick              # a few minutes, checks plumbing
python examples/run_underreach.py                      # the headline configuration
python examples/run_underreach.py --sweep 4 8 16       # error against graph diameter
```

## What is in here

```
neuralmesh/
  mesh/geometry.py      TriMesh, structured and jittered meshes, uniform refinement
  mesh/graph.py         mesh to graph, relative-geometry edge features, BFS diameter
  fem/poisson.py        P1 assembly, Dirichlet constraints, verified convergence
  models/blocks.py      MLP, residual message passing, physics attention with
                        a learned distance bias, chunked to bound O(N^2) memory
  models/architectures.py  the three architectures plus capacity matching
  data/dataset.py       Latin hypercube sampling, train-only normalisation, caching
  train/trainer.py      training loop, masked losses, opt-in physics residual
  evaluate/underreach.py   the experiment and its controls
  cli.py                command line
tests/                  72 tests, physics properties rather than stored fixtures
examples/               reproduce the study
```

Some design choices worth flagging, because they change what the model can learn:

- **Edges carry relative geometry only.** Edge features are the displacement vector
  between endpoints and its length. Nothing in the message-passing path sees an
  absolute coordinate, so translation invariance holds by construction rather than by
  hoping the training data covered every position. There is a test that permutes node
  labels and asserts the prediction is unchanged.
- **Sum aggregation, not mean.** The physical quantity being passed between nodes is a
  flux, and sums are what conserve it.
- **Residual message-passing updates.** Without them, stacking the fifteen-plus blocks
  needed to span a long strip drives activations to a fixed point and training stalls.
- **Boundary nodes are excluded from the loss.** Their values are inputs, not
  predictions. Scoring them inflates every metric, because the model gets credit for
  copying a feature it was handed, and on a fine mesh the boundary can be a third of
  all nodes.
- **Normalisation is fitted on the training split only.** Fitting on everything leaks
  distributional information and flatters the reported error. This is the most common
  silent mistake in surrogate work, so there is a test for it.
- **The physics residual is off by default.** It is implemented and tested, but
  weighting a PDE residual against a data term is a real hyperparameter with real
  failure modes, and reporting a number obtained with an untuned physics weight would
  be misleading.

## Honest limits

- **This is a 2D scalar diffusion problem.** Steady, linear, single field. It is chosen
  because under-reaching is cleanest to isolate there, not because it is hard. Nothing
  here demonstrates anything about Navier-Stokes or nonlinear solid mechanics.
- **Attention is O(N^2) in node count.** Fine at these mesh sizes, and the reason
  production systems reach for hierarchical or spectral alternatives instead.
- **The datasets are small.** Hundreds of solves, not hundreds of thousands. Absolute
  error levels here are not competitive with a well-resourced surrogate and are not
  meant to be. What is being compared is architectures under identical budgets.
- **Reported error is a test-split average.** Averages hide the extremes, which is
  exactly where design decisions usually get made. That is why the per-distance-band
  breakdown is reported alongside.
- **Attention is not the only fix.** Hierarchical pooling, multigrid-inspired
  transfers, and spectral operators all attack the same problem, and this repository
  does not compare against them.

## Background

The under-reaching framing follows the mesh-based learned-simulation literature:
MeshGraphNet (Pfaff et al., *Learning Mesh-Based Simulation with Graph Networks*),
the over-smoothing and over-squashing results from graph learning, and neural operator
work (FNO, DeepONet) where global coupling is obtained spectrally rather than by
attention.

## License

MIT. See [LICENSE](LICENSE).

---

Built by [Samarjith Biswas, Ph.D.](https://samarjithbiswas.com) &nbsp;·&nbsp;
wave physics, acoustic metamaterials, and machine learning for simulation
