# Roadmap

An honest statement of what this repository currently supports, what it does not, and
what would have to be true before the work belongs in a high-tier venue.

The purpose of this file is to stop the README from overclaiming by omission. Anything
listed as not done is not done, and the README should not read as though it were.

---

## Where it stands

**Done and verified.**

- 2D P1 triangular solver, verified by two manufactured solutions and against an
  independent closed-form series (agreement 0.113% of peak).
- 3D tetrahedral solver for nonlinear diffusion `-div(k(u) grad u) = f` with
  `k(u) = k0(1 + alpha u^2)`, Newton-Raphson with a consistent tangent. Tangent confirmed
  against finite differences to 2.3e-08, Newton order 1.96 to 2.06, spatial convergence
  1.729 to 1.91 against a theoretical 2.
- Three architectures with matched parameter counts, a no-communication control, and
  error resolved by distance from the driven boundary.
- The 2D headline reproduced at five seeds, paired within seed: +46.0% mean improvement,
  standard deviation 13.4%, sign agreeing on 5/5.
- 112 tests, 20 CLI verification checks, CI on Python 3.10 to 3.12.

**Not done.** These are the honest gaps, in the order they would need closing.

| Gap | Why it matters | Status |
|---|---|---|
| Learned models trained on the 3D nonlinear problem | The solver exists; the experiment on it does not. Until it runs, the under-reaching result is a 2D scalar result. | not started |
| Published operator baselines | The comparison is currently between three architectures written here. A reviewer will ask about FNO, GNO, DeepONet, Transolver, and a published MeshGraphNet configuration. | not started |
| Larger datasets | 49 training cases per configuration. Hundreds to thousands would be needed before absolute accuracy means anything. | not started |
| Training to convergence | 90 epochs with all four models still improving. Ranking under a shared short budget is defensible; absolute numbers are not. | not started |
| More than five seeds | Five gives a sign test at p ≈ 0.03 and a wide interval on the magnitude. Ten to twenty would tighten it. | partial |
| A second physics | Diffusion only. Linear elasticity on the same tetrahedral meshes would test whether the finding is about ellipticity or about this operator. | not started |

---

## What would make this publishable, and where

### The paper that exists now

The publishable contribution today is **methodological, and it is a negative result about
measurement**: aggregate error metrics systematically hide under-reaching, and here is a
controlled protocol that exposes it.

The evidence is already in hand and is genuinely counterintuitive. Across diameters 20,
40 and 80 the reach-limited baseline shows **no degradation in aggregate relative L2**
(0.181, 0.163, 0.167). Resolve the same runs by distance from the driven boundary and the
deep-interior error grows 48% and 85% for the two local models while staying flat for the
global one. A reader who only saw the aggregate column would conclude under-reaching was
not present.

That is a real finding about how the field reports results, it is supported by
parameter-matched controls, and it is reproducible from committed data.

**Realistic venues:** Journal of Computational Physics, Computer Methods in Applied
Mechanics and Engineering, Engineering Applications of Artificial Intelligence, or a
NeurIPS/ICLR workshop on machine learning for the physical sciences.

### The paper that would reach Nature Machine Intelligence

Two things have to change, and neither is a weekend.

**A problem with consequence.** Scalar steady diffusion, in any number of dimensions, is
not going to carry a high-tier paper. The 3D nonlinear solver in this repository is a step
up but still a model problem. What would count: nonlinear solid mechanics with contact,
or transient Navier-Stokes, on geometry someone actually designs, with a quantity someone
actually certifies against.

**Benchmarking against the field.** Comparing three architectures written by one author
establishes an internal ranking, not a contribution to the literature. FNO, geometry-aware
FNO, DeepONet, GNO and Transolver all attack global coupling by different means, and a
claim about attention only means something relative to them. This also risks the finding:
a spectral operator may match or beat the attention variant, in which case the paper
becomes a different and more interesting one about *which* form of global coupling is
worth its cost.

Realistically six to twelve months, most of it spent generating credible data and
implementing baselines faithfully enough that the comparison is fair.

---

## Ordered plan

**Phase 1, foundations.** Done. Verified solvers in 2D and 3D, controlled protocol,
multi-seed headline.

**Phase 2, move the experiment to 3D.** Port the graph construction and dataset
generation to `TetMesh`, run the diameter sweep on `bar_mesh`, and find out whether the
distance-resolved signature survives nonlinearity. This is the next thing to build, and
it is where the finding is most likely to break. If it does break, that is the result and
it gets reported.

**Phase 3, baselines.** Implement FNO on the structured box meshes (its FFT needs the
regular grid, which is a genuine limitation worth documenting rather than hiding),
DeepONet, and a graph neural operator. Match parameter budgets and training budgets as
carefully as the current comparison does.

**Phase 4, scale.** Thousands of solves, training to convergence, ten to twenty seeds,
proper confidence intervals rather than a sign test.

**Phase 5, a second physics.** Linear elasticity, then hyperelasticity, on the same
meshes. This is what distinguishes a claim about elliptic problems from a claim about one
operator.

**Phase 6, write it.** Which paper depends entirely on what phases 2 and 3 return.

---

## Standing rules for this repository

1. **Every number is measured.** If a figure appears in the README it came from a log, a
   solve or a test in this repository, and the raw output is committed under
   `benchmarks/`.
2. **Negative results get equal billing.** The aggregate metric showing no diameter trend
   is stated in bold in the README, next to the headline. The wall-shear-stress head being
   weak is stated next to the strong result. This is not modesty, it is the only way the
   strong claims stay credible.
3. **Caveats travel with numbers.** Validation rather than test, random rather than
   published split, seeds, and dataset size are named wherever a score is quoted.
4. **The solver is verified, not trusted.** Everything learned is measured against it, so
   it carries its own suite, and the suite carries a test that sabotages the solver to
   prove the suite would notice.
