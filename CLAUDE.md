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
python validate_topology.py        # stage 2: 102 checks (--steps 300 for a fast pass)

python run_experiment1.py          # stage 3: the paper's city-to-city hop counts
python validate_routing.py         # stage 3: 44 checks (--steps 300 for a fast pass)

python run_traffic.py --loads 0.5 1 2 5   # stage 4: every selector under offered load
python validate_traffic.py         # stage 4: 34 checks (--steps 300 for a fast pass)

python validate_extension.py       # Phase 2 extension: 65 checks (61 at --steps 300)

python run_rlenv.py --loads 0.5 1 2   # Phase 3: baseline policies through the RL env
python validate_rlenv.py           # Phase 3 env: 95 checks, ~18 s (all G* mutants use it)

python mutants.py                  # meta-test: 129-mutation sweep, stages 2-4 + extension + rlenv
python mutants.py T01 T02          # just the named mutants
```

There is no pytest and no test framework. `validate.py`, `validate_topology.py`,
`validate_routing.py`, `validate_traffic.py`, `validate_extension.py` and `validate_rlenv.py` **are** the tests; each check prints PASS/FAIL with the measured
value and the bound. None has a flag to run a single check — to isolate one, call its
`check_*` function from a scratch script. `--steps 300` is the fast path for the stage-2/3/4 and
extension suites and is what `mutants.py` drives.

Each mutant names the suite meant to catch it (`suite=ROUTING_SUITE` for the `R*` ones),
and every named suite's baseline must be green before any mutant runs, so a stage-3
mutant cannot be credited to a stage-2 check.

`mutants.py` **rewrites source files in place**. Never run anything else against the
repo while it is running, and never leave it interrupted — check `git`-less working
files are restored before doing anything else. It takes about an hour; the 20 `X*` mutants
each re-run the ~2.5-minute extension suite.

Useful flags:

```bash
python run_topology.py --objective min            # paper's Eq. 15 argmin
python run_topology.py --latency-formula c_over_d # paper's written c/D
python run_topology.py --planes 10 --per-plane 20 # constellation shape
python run_sim.py --inclination 53.2              # Gen1 shell instead of the 70 deg default

python run_experiment1.py --all-metrics           # all three OSPF cost metrics
python run_experiment1.py --pair Darwin Singapore # another city pair

python run_traffic.py --w-egress 0.2 --w-congestion 0.2   # extension on: UNTUNED, see Phase 2

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
- **Stretch factor** (`path_km / geodesic_floor_km`) is reported beside hops and
  latency. The floor is the arc on a sphere of radius min(|a|, |b|), the stations'
  own geocentric radii, at the angle between their geocentric vectors. That makes
  stretch >= 1 a theorem: radial projection never lengthens a path lying outside the
  sphere, and every routed path does (ISLs clear Earth + 50 km; access links leave
  above the mask). The triangle inequality alone gives only the chord, and
  `great_circle_km` (6371 km, geodetic latitude) is not a bound: New York sits at
  6369 km. `path_stretch` raises below 1. Co-located endpoints (a city on a gateway's
  coordinates) have no arc, so their stretch is NaN, not an error.

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

## Cross-machine reproducibility (2026-10-26) -- a stated limitation

**Exact-byte reproducibility does not hold across machines. Numerical
reproducibility holds to ~1e-15 for every float quantity checked so far.**
Measured between this laptop (Python 3.11.5, numpy 2.0.0, scipy 1.16.1) and
the A100 cluster (numpy 2.5.4, scipy 1.18.1): the pinned backpressure
delivered fraction agrees to 7.87e-16, every tolerance-based check passes, and
each machine is deterministic run to run. Greedy, DoTD-forward and
DoTD-backward agree **bit for bit**, float64 priority matrices included, so
stage 1 and the stage-2 scoring arithmetic are identical. What differs is the
last bit of reductions (a sum's SIMD path changes with the numpy build).
`validate_rlenv.py` therefore compares the pinned queue trajectory to a stored
array (`data/pins/`, itself held to the original SHA-256) within
`RLValidationConfig.queue_rtol`, and episode totals within `sum_rtol`;
per-step vectors with no reduction are still compared exactly.

**+Grid previously had an order-dependent pairing (fixed 2026-10-26).** Its
cross-plane links came from whatever `linear_sum_assignment` returned, and
7 of the 9 plane pairs on the default shell have exactly tied optima: summed
|phase difference| (L1 matching) is degenerate whenever several members of
one plane lie on the same side of several members of the other. Which tied
pairing came out was decided by the last bit of the SVD-fitted phases.
Measured with `results/repro/diagnose_grid.py` against the cluster: stage-1
data, plane labels, plane adjacency and in-plane order all identical; normals
<= 5.6e-16 apart (OpenBLAS 0.3.27 on AVX2 vs 0.3.34 on AVX-512), phases <=
5.7e-14 deg; on planes 2-3 that bit swapped 60-82/72-76 for 60-76/72-82, so
the two machines' +Grids differed by 4 of ~331 links at every one of 1800
steps. Not scipy: the laptop's scipy on the cluster's phases gives the
cluster's pairing.

Now `deterministic_assignment` (topology/grid.py) keeps the minimum total
phase difference and breaks ties -- totals within `PAIRING_TIE_TOL_DEG` =
1e-9 -- by lowest satellite index. Measured decision margins: accepted ties
<= 5.7e-14 deg, nearest non-tie 0.22 deg, so the tolerance is ~13 orders of
magnitude from either side. `check_grid_pairing` pins the rule, checks
optimality and margins, and requires an identical grid under 1e-11 deg phase
noise (the untie-broken solver changes in 5/5 draws). Margins are a property
of the shell: on another shell, check them before trusting cross-machine
agreement.

**This changed +Grid's results.** The new grid differs from the old by 34
structural pairs (the index rule decides 19 rows, not just the one that
flipped). Experiment 1, New York -> San Francisco: +Grid median hops 7
(unchanged), mean 7.40 -> 7.41, mean latency 37.1 -> 36.0 ms (hop metric);
scored selectors identical (new run in `results/repro/`). Every committed
+Grid number elsewhere -- `results/*.csv`, `results/scale/`, the 34-plane
table above, Phase 2 and rlenv +Grid figures -- predates the fix. The
`GOLDEN` +Grid pins in `validate_extension.py` were re-recorded once, on
purpose, with the old values kept in a comment.

## Current configuration and state

187 satellites: 10 consecutive real orbital planes of ~17–27 on the 70° shell, 1800
timesteps of 1 s. `SelectionConfig.n_planes`/`sats_per_plane` mean **real orbital
planes**, identified by RAAN gap, with the plane count an output of the clustering.
Do not reintroduce a forced cluster count: it merges ~7 real planes into each label and
thins every one of them, which leaves `+Grid`'s in-plane neighbours a third of an orbit
apart and beyond the range cap. The suite now asserts stage 1 and stage 2 resolve planes
one-for-one, which is what catches that.

Stage 1 is validated (29/29), stage 2 (96/96; 102/102 after the 2026-10-26 +Grid tie-break), stage 3 (44/44), stage 4 (34/34) and
the Phase 2 extension (65/65), with 99/101 mutants killed, the 2 documented equivalents surviving and 0 unexpected,
in the last full sweep; X21 (added after it) was run alone and killed, as were M48-M50 (the +Grid tie-break, 2026-10-26).

**At 34 planes (601 sats) checks fail on purpose**, all of them findings; do not
weaken any. See `results/scale/NOTES.md`.

| suite | `strict` | `bridge` |
|---|---|---|
| stage 2 | 92/96 | 93/96 |
| stage 3 | 36/36 | 35/36 |

- Stage 2, both policies: "Greedy / DoTD-forward / DoTD-backward: one connected
  component at every timestep" (3 checks). At global scale the scored selectors form
  closed 5-cliques from counter-rotating planes (disconnected at 66, 46 and 99 of 360
  timesteps). A property of per-link selection, not a bug.
- Stage 2, `strict` only: "+Grid: one connected component at every timestep". The
  shell's two empty plane slots split +Grid in two at every timestep, which inflated
  its hop count to 17.
- Stage 3, `bridge` only: "+Grid needs more hops than every scored algorithm", because
  the paper's ordering is false there: +Grid 5.47 mean hops, Greedy 5.76,
  DoTD-forward 6.38.

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

## Stage 4 (`traffic/`) and the Phase 2 extension — built, untuned

- `demand.py` — 200 GeoNames cities (`data/cities.csv`, raw population, no market
  filtering; the China skew is a model property), diurnal raised cosine in local solar
  time, Poisson flow arrivals. Gateways are a flag, `gateway_sites` (`--gateway-sites`):
  `populous20` (default, ASSUMED) is the largest city of each of the 20 most populous
  countries, which puts gateways where the demand is; `krishnan8` is the eight sites
  named in Krishnan et al. Fig. 1, read from `data/gateways_krishnan.csv` (names only
  from the paper, coordinates from GeoNames; see `data/README.md`). Neither is a real
  operator's gateway map. The choice is in `variant_label` and the output filename.
- `network.py` — shortest-path forest from all gateways at once over the SELECTED links;
  cities attach by their cheapest access link and are never graph nodes (a city seeing
  two satellites would otherwise be a free relay). Fluid queue per directed link with a
  50 ms buffer; a removed link loses its backlog, which is the only way churn costs
  anything. Loss is `1 - delivered/routed`; the summed link drops double-count.
  `gateway_imbalance` per step = peak feeder utilisation / aggregate feeder
  utilisation over gateways with any capacity (1 = balanced). A topology cannot change
  feeder capacity -- ground links are not selected -- so moving load between gateways
  is the only gain available to it there. On 187 sats it is ~5.3-5.7 for every
  selector (range 2.5-12.5 per step): load is heavily skewed.
- `predict.py` -- the open-loop `GroundPrediction` the extension scores against, and
  `gateway_pair_share` (reported by `run_traffic.py`).

Baseline on 187 sats: the gateway feeder links, not ISLs, are the bottleneck. Delivery
is 100% up to 2 Gbps peak load and collapses by 5 Gbps; selectors differ by ~2%.
Evaluate on 601 sats (bridge), build on 187. Phase 2 comparisons must include Greedy
as well as DoTD: beating DoTD alone could be the lookahead's doing.

### The extension terms (`w_egress`, `w_congestion`; both default 0)

Both are per-satellite quantities in a `GroundPrediction` (`traffic/predict.py`),
handed to `Greedy`/`DoTD` as `ground=`; `topology/` never imports `traffic/`.

- **Egress** (reward): best feeder capacity from j to any gateway above the mask, 0 if
  none. Per link, the **max** of the two ends -- a mean would rank gateway-to-gateway
  pairs highest. Integrated over the forward window inside the same `_forward_parts`
  loop as link quality (same offsets, same `exp(-h/H)` weights), masked by the link's
  visibility at each future sample. Added to the forward term, not folded into its
  `(w1 + w2)` normaliser, so switching egress on does not shrink the geometric
  lookahead; the forward term is then bounded by `1 + w_egress`.
- **Congestion** (penalty), current timestep only: expected demand landing on j, each
  city's rate split equally over the satellites it sees (ASSUMED). Per link, the mean
  of the two ends. Written as headroom `w * (1 - D)`: a constant shift on every
  candidate, so the selection equals subtracting `w * D` -- the suite verifies it
  over all-zero predictions.
- Both normalised by their own running maxima (`GroundNormaliser`), causal and clipped.
- **DoTD-backward's accumulator remembers the Phase 1 score only.** Folding the new
  terms in rescales its memory whenever a weight is on, even with uninformative
  predictions. A deviation from the agreed plan, made for that reason.
- **Open loop is structural.** The predictor uses `expected_rate_bps`, never
  `offered_bits`, and never a `TrafficSim`. The suite checks seed invariance and scans
  `predict.py` for forbidden names. If queue feedback ever seems necessary, raise it
  rather than adding it.

Regression guard: SHA-256 of every selector's adjacency and priority sequence, recorded
on the Phase 1 code before the extension existed, is pinned in `validate_extension.py`
(`GOLDEN`). If stage 1 is legitimately recomputed, those pins move and must be re-recorded
from a commit known to be Phase 1, never from the code under test.

No weight is tuned. `ExtensionValidationConfig.probe_weight = 0.2` exists only to prove
each term is not inert. At that probe every timestep differs from Phase 1 for both
terms.

**REQUIRED before reporting any extension result: a demand-perturbation sensitivity run.**
The congestion term is fed the exact demand model stage 4 evaluates against -- a perfect
forecast, the best case for the extension and its main threat to validity. Any reported
gain must be re-measured with the predictor's demand perturbed away from the evaluator's
(e.g. a shifted diurnal peak, a different trough ratio, or population noise) and the
gain's dependence on that error stated. A result without it is not reportable.

## Phase 3, step 1 (`rlenv/`): Gymnasium environment, backpressure baseline -- no agent yet

Directed forwarding over a fixed selector's topology, following Krishnan et al.
(arXiv 2601.13662, Sec. II-III, read 2026-10-15): one aggregate queue per
satellite, user uplink anycast to any gateway, the downlink drained automatically
(not in the action), one bit per directed selected link (their Eq. 18), and
backpressure as "activate k->m iff (Q_k - Q_m) C_km > 0" (Eq. 16).

- **New physics, on purpose.** Stage 4's queues are per link with fixed routes:
  there is no forwarding decision in it for an action to make. `rlenv/queues.py`
  is a per-satellite queue model fed by stages 1-4 unchanged (selector link sets,
  `offered_bits`, access, `gateway_access`). Nothing in `traffic/` was modified.
- **There were no Phase 2 backpressure numbers** (Phase 2 is fixed OSPF), so
  "reproduces Phase 2" became three checks: inputs equal stages 2-4's exactly;
  backpressure through `env.step` equals a direct loop over the queue model,
  bit for bit; and golden pins (`PIN_BP_*`, recorded on this code, a regression
  guard only). Plus a curve-shape check: the env's `ShortestPathPolicy` against
  stage 4's `TrafficSim` (run `to_gateway`) on the same topology and bits; max
  delivered-fraction gap 0.072 at 3x8. Same routing, two queue models, same
  curve -- backpressure's lead (0.99 vs 0.55 delivered at load 1) is routing.
- **Action** `MultiBinary(N, D)`, D = `u_max`; slot (i, k) = send to i's k-th
  selected neighbour, ascending index, padded with -1 and masked. Padding is
  ignored, non-binary or misshapen actions are rejected. Legality is asserted
  inside every `step()`, independently of the decoder.
- **Observation** Dict, float64 (float32 made backpressure tie on nearly equal
  queues and diverge from the raw-queue loop by 1 Mb at load 0.5): `queue` as
  q/(q+q_ref) in [0, 1), `isl_capacity` and `downlink_capacity` over FIXED
  whole-window maxima, `neighbours`, `action_mask`.
- **Unbounded queues are load-bearing.** With a finite buffer the queue-only
  reward is gamed by dropping: at load 2, 5 s buffer, Random out-scored
  backpressure (-1.11 vs -1.30) while delivering 47% vs 62%. `buffer_s` stays
  available, default None. Reward is linear -(w_mean mean + w_max max) of q/q_ref.
- **Load is scaled to the subset** (`EnvConfig.load` = served demand / the
  subset's total downlink capacity), because the global 2 Gbps peak puts ~0.5
  Gbps on 24 satellites with ~3 Gbps of downlink. The action only matters once
  congested: at load 0.5 Random is close to backpressure (-0.32 vs -0.26).
- **Shadow reward (Eq. 20), `reward_kind="shadow"`: PARALLEL run** (chosen
  2026-10-16; the paper does not say). A second queue model, empty at reset,
  stepped with backpressure on its own queues through the env's own
  `observe`/`forward_for`, same arrivals. Checked bit for bit against a real
  backpressure rollout, and per step = queue reward(agent) - queue reward(BP).
  `RewardContext.counterfactual` (one-step) remains, for analysis only.
- `reward_scale` multiplies every reward the env returns. 1.0, NOT chosen.
- `start_mode="uniform"` draws each episode's start from the seeded RNG. Needed
  for training and for any selector comparison (see the scale section below).

**Evaluation scale (2026-10-16): 6x12 = 72 sats, smallest robust choice.**
Criterion: all four selectors connected at every step of the window, and every
pair of scored selectors giving backpressure episode rewards that differ by
more than the seed spread, on episodes spread over the window (starts 0, 400,
800, 1200, 1599, seeds 7-9, load 1). Measured |dR| in seed-sd units
(G = Greedy, Df/Db = DoTD-forward/backward):

| shape | N | connected | links G-Df / G-Db | BP G-Df | BP G-Db | SP G-Df |
|---|---|---|---|---|---|---|
| 4x10 | 40 | all | 5.5% / 0.1% | 11.3 | 0.1 | 0.0 |
| 4x12 | 48 | all | 5.2% / 0.0% | 0.1 | 0.0 | 0.0 |
| 4x15 | 60 | all | 8.2% / 3.2% | 5.8 | 0.1 | 0.0 |
| 5x12 | 60 | all | 23% / 6.5% | 1.9 | 2.1 | 13.2 |
| **6x12** | **72** | **all** | 25% / 7.5% | 3.8 | 2.0 | 2.2 |
| 5x15 | 75 | all | 25% / 5.9% | 8.2 | 2.0 | 1.1 |
| 8x10, 10x10 | 80, 100 | +Grid never | | | | |
| 6x15 | 90 | DoTD-b 29% disconnected | | | | |

- Smaller shapes all fail (3x8 ... 3x16: the scored selectors are identical;
  5x8, 6x8, 5x10, 6x10: someone disconnects).
- 4x10 is the smallest for Greedy vs DoTD-forward ALONE, but fragile: 4x12,
  bigger, has the same link gap (5%) and no outcome gap. DoTD-backward equals
  Greedy there. 5x12 is borderline (~2 sd).
- Link differences do not imply outcome differences. Backpressure delivers
  the same fraction on any connected topology (the downlink caps it), so the
  topology shows up in queue lengths, not delivery. At 5x8 and 4x15 the scored
  selectors agree for the first 342 / 477 steps, which is why fixed episodes at
  step 0 showed no difference.
- "Seed sd" here is 3 seeds x 5 starts at one load: a screening criterion, not
  a significance test. Re-measure at the chosen scale before reporting.

**Queue-model agreement is shape- and window-dependent.** Shortest-path
delivered fraction, env vs stage 4, max |d| over loads 0.25-2, by episode start:

| shape | start 0 | start 800 | start 1599 |
|---|---|---|---|
| 3x8 | 0.072 | | |
| 10x20 (187) | 0.041 | 0.081 | 0.079 |
| 4x6 | 0.113 | 0.031 | 0.073 |

Two mechanisms, with opposite signs:
1. Buffering (env higher): stage 4's 50 ms per-link buffers drop bursts that
   unbounded per-satellite queues carry over. Largest near the knee (4x6 start
   0, load 1: +0.113; 187: up to +0.081 at load 0.5).
2. Stranded demand (env lower, independent of load): a city attaches to its
   best-capacity satellite even if that satellite's component has no gateway.
   Stage 4 attaches through a satellite that has a path, or drops the bits from
   its denominator as no_path. 4x6 start 1599: 9.8% of arrivals stranded, giving
   -0.067 at every load. 0.0% at 3x8 and 187, and 0 on any topology connected
   at every step, so it does not arise at the 6x12 evaluation scale.

Default 3x8 = 24 sats, chosen from a measured sweep of 24-sat shapes (gateway
in view on 13.6% of satellite-steps, connected, selected degree 3.71). Wider
shapes (4x6, 8x3, 12x2) split into ~2 components. 3x8 is the CPU dev scale
only: its selectors are identical. `validate_rlenv.py` 95/95 at
the default (87 before the 2026-11-19 reward-direction checks, 84 before the 2026-10-26 tolerance change), 71/71 at 10x20 (before it; 72 expected, not re-run) (187 sats; golden pins skipped off-default, count before the per-step capacity checks), and
26/26 `G*` mutants killed (G25-G26 run alone, 2026-11-19). Recorded findings, checks NOT weakened:

- At 3x8 Greedy, DoTD-forward and DoTD-backward pick identical topologies at all
  1800 steps (the matching is nearly forced), so `selector` only distinguishes
  +Grid from the scored ones at this scale.
- 2x12 and 4x6 fail "backpressure delivers at least what shortest path does":
  at light load backpressure trails by up to 0.015 (no queue gradient, so bits
  wander). 4x6 also fails the curve-shape bound at the knee (0.113 at load 1):
  50 ms per-link buffers vs unbounded queues.

`README.md` documents stage 1 only, and two of its sections are now stale: the satellite
selection (it describes 20 planes × 5) and the visibility-geometry formula (the clamped
`t*` form has been replaced by a cross-product form that is symmetric by construction,
because the old one disagreed with its own transpose at float32 precision).
