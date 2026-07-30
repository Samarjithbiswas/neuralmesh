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
| Learned models trained on the 3D nonlinear problem | Done. The result is **negative**: error falls with diameter in 3D, so the 2D signature does not reproduce. See below. | done, negative |
| A diameter sweep that is not confounded | Both sweeps raise node count with diameter, so supervision and required reach move together. `examples/controlled_sweep.py` holds nodes fixed at ~420 while diameter spans 34 to 104. | running |
| Published operator baselines | The comparison is currently between three architectures written here. A reviewer will ask about FNO, GNO, DeepONet, Transolver, and a published MeshGraphNet configuration. | not started |
| Larger datasets | 49 training cases per configuration. Hundreds to thousands would be needed before absolute accuracy means anything. | not started |
| Training to convergence | 90 epochs with all four models still improving. Ranking under a shared short budget is defensible; absolute numbers are not. | not started |
| More than five seeds | Five gives a sign test at p ≈ 0.03 and a wide interval on the magnitude. Ten to twenty would tighten it. | partial |
| A second physics | Diffusion only. Linear elasticity on the same tetrahedral meshes would test whether the finding is about ellipticity or about this operator. | not started |

---

## What would make this publishable, and where

### The paper that exists now

**This section was written before the confound was found, and it no longer holds as
stated.** Kept here rather than deleted, because quietly rewriting a claim after it fails
is how a repository stops being trustworthy.

The intended contribution was methodological: aggregate error metrics hide under-reaching,
and a distance-resolved protocol exposes it. The 2D evidence looked strong, with the
aggregate flat across diameters 20, 40 and 80 while deep-interior error grew 48% and 85%
for the local models and stayed flat for the global one.

What is wrong with it: the sweep varies diameter by lengthening the strip, so node count
rises with diameter and supervision is confounded with required reach. The confound
happens to push against the deep-band finding, which is why it is suggestive rather than
dead, but it is not a controlled measurement. The 3D replication then produced the
opposite trend, consistent with the confound dominating.

**What is actually defensible today**, and it is less than the above: at any *fixed*
diameter, with parameter counts matched and all models seeing identical graphs, the
global-attention variant beats the reach-limited baseline. That holds in 2D at five seeds
(+46.0%, sign agreeing 5/5) and in 3D at all three diameters (+28%, +50%, +25%). That is a
ranking result, not a scaling law, and a ranking result is a workshop paper rather than a
journal one.

Whether the scaling claim comes back depends on the controlled sweep.

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

**Phase 2 outcome, recorded honestly.** The 3D port is done and the finding did not
survive: aggregate and deep-interior error both *fall* with diameter, for every
architecture. Diagnosis is that both sweeps confound diameter with node count, and in 3D
the extra supervision dominates. So the correct statement is not "under-reaching is
absent in 3D" but "this design cannot measure it in either dimension". The architecture
ranking does survive: the transformer wins at all three 3D diameters by 28%, 50% and 25%.

This is the outcome the plan said would be reported if it happened, and it changes what
Phase 3 is for: baselines now have to be compared under a design that actually isolates
reach.

**Phase 2b, the corrected sweep.** Hold node count fixed while varying diameter, by
trading strip width for length. This is the experiment that should have been run first.
Until it returns, the diameter *trend* is unresolved in both dimensions and the README
says so.

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
