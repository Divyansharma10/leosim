# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A reproduction of Ron et al., *Time-Dependent Network Topology Optimization for LEO
Satellite Constellations* (IEEE INFOCOM 2025), with an intended extension. Research
code: the deliverable is numbers that can be defended, so the validation suites carry
as much weight as the simulator. `leo-isl-topology-project.md` holds the research
context — the algorithm, the paper's ambiguities, and what the extension is meant to
add. Read it before changing scoring or selection.

## Commands

Run everything from the repo root (scripts rely on the cwd being on `sys.path`).

```bash
pip install -r requirements.txt

python run_sim.py                  # stage 1: build or load the link dataset
python validate.py                 # stage 1: 29 physics checks
python plot_visibility.py          # -> visible_pairs.png

python run_topology.py             # stage 2: all selectors, metrics CSV + plot
python validate_topology.py        # stage 2: 93 checks (--steps 300 for a fast pass)

python run_experiment1.py          # stage 3: the paper's city-to-city hop counts
python validate_routing.py         # stage 3: 36 checks (--steps 300 for a fast pass)

python run_traffic.py --loads 0.5 1 2 5   # stage 4: every selector under offered load
python validate_traffic.py         # stage 4: 27 checks (--steps 300 for a fast pass)

python mutants.py                  # meta-test: 69-mutation sweep over stages 2-4
python mutants.py T01 T02          # just the named mutants
```

There is no pytest and no test framework. `validate.py`, `validate_topology.py`,
`validate_routing.py` and `validate_traffic.py` **are** the tests; each check prints PASS/FAIL with the measured
value and the bound. None has a flag to run a single check — to isolate one, call its
`check_*` function from a scratch script. `--steps 300` is the fast path for the
stage-2/3/4 suites and is what `mutants.py` drives.

Each mutant names the suite meant to catch it (`suite=ROUTING_SUITE` for the `R*` ones),
and every named suite's baseline must be green before any mutant runs, so a stage-3
mutant cannot be credited to a stage-2 check.

`mutants.py` **rewrites source files in place**. Never run anything else against the
repo while it is running, and never leave it interrupted — check `git`-less working
files are restored before doing anything else. It takes ~10 minutes.

Useful flags:

```bash
python run_topology.py --objective min            # paper's Eq. 15 argmin
python run_topology.py --latency-formula c_over_d # paper's written c/D
python run_topology.py --planes 10 --per-plane 20 # constellation shape
python run_sim.py --inclination 53.2              # Gen1 shell instead of the 70 deg default

python run_experiment1.py --all-metrics           # all three OSPF cost metrics
python run_experiment1.py --pair Darwin Singapore # another city pair

# full 70-deg shell: 34 real planes, 601 sats. 5 s step because 1 s needs ~7.8 GB
python run_experiment1.py --planes 34 --step 5 --grid-gaps bridge --all-metrics
python validate_topology.py --planes 34 --step 5 --grid-gaps bridge
```

`--planes` used to be capped at 200 satellites: the drivers never passed `n_sats`, and
selection silently truncates to it. They now pass `planes * per_plane`.

## Three stages, and why only the first is inside `leosim/`

**Stage 1 (`leosim/`)** is a model of *possibility*: for every timestep, which satellite
pairs can see each other, how far apart, at what capacity and latency. It decides
nothing. `LinkDataset` holds `[timestep, sat_i, sat_j]` arrays (`distance`, `visible`,
`capacity`, `latency`) plus `positions` `[t, sat, 3]`.

**Stage 2 (`topology/`)** decides which four links each satellite actually forms.

**Stage 3 (`routing/`)** puts cities on the ground and finds paths across a chosen
topology. It exists because stages 1 and 2 produce only load-independent, topology-wide
quantities, while the paper's headline numbers are per-path. It is also the first part
of the project that is **not rotation-invariant**: stage 1 could ignore the TEME/GCRS
distinction because every quantity was a distance between two orbiting bodies, but a
ground station rotates with the Earth, so the frame and the sidereal rate now matter.
The suite pins both.

`topology/` and `routing/` sit outside the `leosim` package **on purpose**. The stage-1 cache key is
a SHA-256 over config + TLE bytes + the source of every `*.py` under `leosim/`
(`leosim.config.source_digest`). A module placed there would invalidate the ~800 MB
dataset on every edit to a scoring function. That source hashing is deliberate and
load-bearing — without it, editing `geometry.py` silently serves pre-edit arrays from
disk — so do not weaken it; move code out of `leosim/` instead.

Editing anything under `leosim/` forces a full recompute (a few minutes) and orphans
the previous `cache/<hash>/` directory.

## Stage-2 structure

- `base.py` — `LinkSet` (adjacency matrix is the single source of truth), `b_matching`,
  and `assert_valid`. Constraints are enforced **twice**, by construction in the matcher
  and by an independent check inside every `step()`, so a bug in the matcher cannot also
  hide itself.
- `scoring.py` — `w1*cap_norm + w2*(1 - lat_norm) + w3*churn`. Normalisation is by a
  **running** maximum that only increases, never a per-timestep max, so scores stay
  comparable across time (the lookahead accumulation depends on this). The clip in
  `normalise` matters only for future samples read against the causal max.
- `dotd.py` — `ScoredSelector` shared by `DoTD` and `Greedy`; the *only* difference is
  the accumulation term, so any gap between them is attributable to it. `Greedy` forces
  `lookahead="none"` whatever config it is handed.
- `grid.py` — `+Grid`, the structural baseline. Structure is fixed once at reset; the
  active topology is that structure intersected with visibility. Planes are re-derived
  here from orbit normals fitted to trajectories, independently of stage 1's labels.
- `metrics.py` — per-timestep rows; all load-independent, matching the paper's axes.

## Stage-3 structure

- `stations.py` — station GCRS positions per timestep, the **geodetic** vertical (taken
  from Skyfield via a 1 km probe, not assumed radial — the two differ by up to 0.19°,
  the same order as the mask's own effect), and access above the elevation mask.
- `paths.py` — one graph per timestep with N + 2 nodes, then Dijkstra. Satellite edges
  are the links the selector chose, **never** the visibility set. Edge costs are
  strictly positive by assertion, because a zero weight vanishes from a sparse matrix
  rather than being free.
- Unreachable timesteps return the `NO_PATH` sentinel and are excluded from means; a
  zero there would silently pull every average down.

The stage-3 hazard is not wrong physics, it is a path over the wrong graph. Routing over
visibility instead of the selection gives plausible hop counts that are identical for
all four algorithms and destroys the comparison without failing anything. `R05` mutates
exactly that, and `check_graph` exists for it.

## Conventions that are not negotiable here

**No bare numeric literal for a meaningful quantity** outside `leosim/config.py`,
`topology/config.py`, `routing/config.py` and `traffic/config.py`. Fields marked `ASSUMED` in those files were not pinned by the
paper or the spec; the value is a documented default, not a reading. Treat changing one
as changing a result.

**Paper ambiguities are exposed as flags, never resolved silently.** Four are live:
`objective` (prose says highest score, Eq. 15 and Algorithm 1 say `argmin`),
`latency_formula` (`c/D` as written is dimensionally inverted and rewards long links),
`route_metric` (the paper says OSPF and reports both hops and latency, without
saying which one OSPF minimises), and `grid_gaps` (whether +Grid links planes across an
empty slot in the shell -- the paper does not say; the 15 deg `strict` cutoff is ours). `variant_label` on both configs is printed by the
drivers, stamped on every result row, and written into output filenames. A result that
does not say which variant produced it is not interpretable — a hop count depends on
all of these flags and on the elevation mask alike.

**Every check drives production code.** Never recompute the quantity under test locally
in a validation check — that is how the FSPL check became unfalsifiable (a 60 dB
injected error passed the whole suite), and how the forward-discount check passed while
nothing pinned the discount direction. Where a constant must be asserted, hard-code the
pinned literal rather than deriving it from the same code each run.

**Pair each check with a discrimination check.** Several checks exist only to stop
another passing vacuously, and say so in their docstring. This is not ceremony: a
healthier system makes checks vacuous silently. Fixing the constellation made the
`connected` flag unfalsifiable (everything is connected now) and made the plane-gap
mutant a no-op. Both had been passing for the wrong reason.

**Add a mutant when you add behaviour** to scoring, selection, metrics or routing.
`mutants.py` is the meta-test, and its claim is `0 UNEXPECTED survivors`. A survivor is
either a missing check or a genuinely equivalent mutant — and equivalence must be
*verified and written down*, not assumed. Two are currently marked equivalent with
reasons; re-check those reasons when the code around them changes.

## Current configuration and state

187 satellites: 10 consecutive real orbital planes of ~17–27 on the 70° shell, 1800
timesteps of 1 s. `SelectionConfig.n_planes`/`sats_per_plane` mean **real orbital
planes**, identified by RAAN gap, with the plane count an output of the clustering.
Do not reintroduce a forced cluster count: it merges ~7 real planes into each label and
thins every one of them, which leaves `+Grid`'s in-plane neighbours a third of an orbit
apart and beyond the range cap. The suite now asserts stage 1 and stage 2 resolve planes
one-for-one, which is what catches that.

Stage 1 is validated (29/29), stage 2 (93/93), stage 3 (36/36) and stage 4 (27/27),
with 67/69 mutants killed and 0 unexpected survivors.

**At 34 planes (601 sats) two checks fail on purpose.** Under `grid_gaps=strict`,
"+Grid baseline is one connected component" fails: the shell's two empty plane slots
split +Grid in two, which inflated its hop count to 17. Under `bridge`, stage 3's
"+Grid needs more hops than every scored algorithm" fails because the paper's
ordering is false there: +Grid 5.47 mean hops, Greedy 5.76, DoTD-forward 6.38. Both
are findings; do not weaken either check. See `results/scale/NOTES.md`.

At global scale the scored selectors also form closed 5-cliques from counter-rotating
planes (Greedy connected at 82% of timesteps). A property of per-link selection, not
a bug.

Experiment 1 reproduces the paper's headline on medians — New York -> San Francisco,
DoTD 5 hops and Greedy 6 — and holds that ordering under all three OSPF cost metrics.
`+Grid` comes out at 7 hops against the paper's 22. The obvious explanation -- 187
satellites over 100° of RAAN leave the grid nearly direct -- was tested by scaling to
the full 34-plane shell, and failed: a connected (`bridge`) +Grid takes 5 hops there,
and DoTD-forward's lead over Greedy inverts. The gap stays open.

Two other open findings, both recorded rather than resolved:

- `DoTD-forward` churns ~5x more than Greedy (0.49 vs 0.10 per step) while
  `DoTD-backward` churns slightly *less* (0.093). The paper claims lower churn is a DoTD
  selling point. A `w_lookahead` sweep shows churn rising monotonically with it and
  meeting Greedy exactly at 0, so the forward number is set by that one undocumented
  weight — `config.py` flags it as the most arbitrary value in stage 2. Vary it before
  trusting any DoTD-vs-Greedy margin.
- The paper's +28% average capacity is measured along routing paths, not over the whole
  topology, so it is only now comparable at all. Stage-2 totals are not the same metric.

## Stage 4 (`traffic/`) — Phase 2, built; extension term not yet

- `demand.py` — 200 GeoNames cities (`data/cities.csv`, raw population, no market
  filtering; the China skew is a model property), diurnal raised cosine in local solar
  time, Poisson flow arrivals. Gateways: largest city of each of the 20 most populous
  countries (ASSUMED).
- `network.py` — shortest-path forest from all gateways at once over the SELECTED links;
  cities attach by their cheapest access link and are never graph nodes (a city seeing
  two satellites would otherwise be a free relay). Fluid queue per directed link with a
  50 ms buffer; a removed link loses its backlog, which is the only way churn costs
  anything. Loss is `1 - delivered/routed`; the summed link drops double-count.

Baseline on 187 sats: the gateway feeder links, not ISLs, are the bottleneck. Delivery
is 100% up to 2 Gbps peak load and collapses by 5 Gbps; selectors differ by ~2%.
Evaluate on 601 sats (bridge), build on 187. Phase 2 comparisons must include Greedy
as well as DoTD: beating DoTD alone could be the lookahead's doing.

`README.md` documents stage 1 only, and two of its sections are now stale: the satellite
selection (it describes 20 planes × 5) and the visibility-geometry formula (the clamped
`t*` form has been replaced by a cross-product form that is symmetric by construction,
because the old one disagreed with its own transpose at float32 precision).
