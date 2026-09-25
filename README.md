# LEO Constellation Simulator

A reproduction of Ron et al., *Time-Dependent Network Topology Optimization for LEO
Satellite Constellations* (IEEE INFOCOM 2025), built in three stages from real
Starlink TLE data.

| Stage | Package | Produces |
|---|---|---|
| 1 — possibility | `leosim/` | every link that *could* exist, and its quality |
| 2 — selection | `topology/` | which four links each satellite actually forms |
| 3 — routing | `routing/` | city-to-city paths across a chosen topology |

```
pip install -r requirements.txt

python run_sim.py             # stage 1: build (or load) the link dataset
python validate.py            # stage 1: 29 physics checks

python run_topology.py        # stage 2: DoTD, Greedy, +Grid -> metrics + plot
python validate_topology.py   # stage 2: 86 checks

python run_experiment1.py     # stage 3: the paper's city-to-city hop counts
python validate_routing.py    # stage 3: 36 checks

python run_phase1_report.py   # every figure and table in results/
python mutants.py             # 53 deliberate bugs; 51 killed, 2 equivalent
```

Default run: **187 satellites** (10 orbital planes x 20), 1800 timesteps of 1 s.
Stage 1 is ~2 min cold and instant cached.

**Results and the reproduction report live in `results/REPRODUCTION.md`.**
Headline: New York -> San Francisco comes out at DoTD 5 hops / Greedy 6, matching
the paper. Three of the paper's numbers did not reproduce; the report says which
and why.

## Layout

| File | Responsibility |
|---|---|
| `leosim/config.py` | All physical constants and tunables. Nothing else holds a bare literal. |
| `leosim/tle.py` | TLE parsing, shell filtering, real-orbital-plane selection |
| `leosim/propagate.py` | SGP4 -> ECI positions and WGS84 geodetic altitudes |
| `leosim/geometry.py` | Distance matrices and Earth-occlusion line of sight |
| `leosim/linkbudget.py` | FSPL -> SNR -> Shannon capacity; latency |
| `leosim/dataset.py` | `LinkDataset` result object and disk cache |
| `topology/scoring.py` | Running normalisation (Eq. 12) and the weighted link score |
| `topology/dotd.py` | DoTD (forward and backward accumulation) and Greedy |
| `topology/grid.py` | `+Grid` structural baseline; orbital-plane recovery |
| `topology/base.py` | b-matching under U=4, and independent constraint enforcement |
| `routing/stations.py` | Ground station positions, geodetic vertical, access |
| `routing/paths.py` | Per-timestep graph and shortest path |
| `validate*.py` | The three validation suites |
| `mutants.py` | Mutation harness over stages 2 and 3 |

`topology/` and `routing/` sit *outside* the `leosim` package deliberately: the
stage-1 cache key hashes every `*.py` under `leosim/`, so a module placed there
would invalidate the whole link dataset on every edit to a scoring function.

## Output

`LinkDataset` holds arrays indexed `[timestep, sat_i, sat_j]`:

| Array | Shape | Units |
|---|---|---|
| `distance` | (T, N, N) | km, symmetric, zero diagonal |
| `visible` | (T, N, N) | bool |
| `capacity` | (T, N, N) | bits/s, zero where not visible |
| `latency` | (T, N, N) | seconds, zero where not visible |
| `positions` | (T, N, 3) | km, Earth-centred inertial (GCRS) |
| `altitudes` | (T, N) | km, WGS84 geodetic |

Helpers: `.degree()`, `.visible_pairs(t)`, `.summary()`.

Cached under `cache/<hash>/` as `.npy` files, memory-mapped on load. The hash
covers the whole config, the TLE file contents *and* a SHA-256 over the source
of every module in `leosim/`, so changing any of the three recomputes
automatically — a stale cache cannot be silently reused. A default run is
~226 MB on disk.

Hashing the source matters more than it sounds. Without it the key is a
function of config and data only, so editing `geometry.py` leaves the
fingerprint unchanged: the pre-edit arrays come straight back off disk and the
edit appears to have done nothing. That failure is silent, and it hides exactly
the changes most worth checking — including, until it was fixed, every attempt
to confirm that `validate.py` fails when the physics is wrong.

## Coordinate frame

SGP4 natively emits TEME; Skyfield applies the TEME→ICRF rotation, so
`positions` are GCRS. **Every stage-1 quantity is rotation-invariant** —
pairwise distance and distance from Earth's centre to a chord are unchanged by
any rotation about the origin — so the TEME/GCRS choice cannot affect any
output.

That stops being true at stage 3. A ground station rotates *with* the Earth, so
its position in the inertial frame depends on the Earth rotation angle at that
instant, and `routing/stations.py` asks Skyfield for GCRS station positions at
every timestep. The stage-3 suite pins it: a station sweeps 360 deg per **sidereal**
day (7.516356 deg over 1799 s, matched to 4e-8 relative). A solar day would give
7.4958 deg and a station treated as Earth-fixed would give 0.

Measuring that is itself subtle — about the GCRS z axis the same rotation reads
7.5263 deg at New York and 7.5161 deg at Singapore, because GCRS is referred to
J2000 and the true pole has precessed ~0.15 deg since, biasing the measurement by
an amount that grows with latitude. The check recovers the rotation axis from the
trajectory instead.

## Visibility geometry

Two satellites see each other iff the **segment** between them clears
6378 + 50 = 6428 km, and they are within 7000 km.

The quantity is the smallest distance from Earth's centre to the *segment*, not to
the infinite line through both satellites. The foot of that perpendicular can lie
outside the segment, in which case the line dives through Earth but the piece
between the two satellites never does; an unclamped test wrongly reports occlusion.
`validate.py` demonstrates this with a concrete failing case.

It is computed in a form that is **symmetric by construction**:

```
inside  <=>  p1 . d <= 0 <= p2 . d        (d = p2 - p1)
clearance = |p1 x p2| / |d|   if inside,  else  min(|p1|, |p2|)
```

The obvious parameterisation, `|p1 + clamp(t*)*(p2 - p1)|`, is mathematically
identical but *not* symmetric in floating point: evaluated as (i, j) and as (j, i)
it rounds differently, and pairs whose clearance sits within an ulp of the blocking
radius come out visible one way and occluded the other. At 187 satellites that was
6 asymmetric entries in 63 million — harmless in aggregate, but a visibility matrix
that disagrees with its own transpose breaks the bidirectional-link constraint every
selector is checked against. The expressions above are exactly symmetric or exactly
antisymmetric under the swap, so the two orderings agree bit for bit.

Within a single shell the segment clamp never changes an outcome: for two satellites
at equal radius the nearest point is the midpoint regardless of separation angle. It
becomes load-bearing once radii differ — a second shell, a ground station, a GEO relay.

## Two places the original spec was internally inconsistent

Both were found by the validation suite and are documented in code.

### 1. Altitude and period bands cannot both hold

Kepler's third law ties them together:

| Constraint | Implies |
|---|---|
| altitude 500–600 km | period **94.6–96.7 min** |
| period 90–95 min | altitude **274–519 km** |

They overlap only in 500–519 km. Real Starlink at 550 km orbits in 95.65 min, not
≤95 min. The suite therefore **derives** the asserted period band from the
configured altitude band, so the two cannot contradict each other, and adds a
much stronger check: the propagated period must satisfy Kepler's third law at
the propagated orbital radius (agreement 0.025%, tolerance 1%). That catches
unit errors, wrong time scales and frame-origin mistakes that any band wide
enough to be safe would let through.

To assert the literal 90–95 min instead, set
`SelectionConfig.altitude_band_km = (350, 519)` — the period band follows
automatically.

### 2. Starlink is no longer a 500–600 km constellation

Of 10,744 catalog objects (fetched 2026-08-20), only **1,194 (11%)** have a mean
altitude in 500–600 km. The median is **471 km**, with a decaying tail down to
194 km. The 550 km figure is the Gen1 shell.

Consequences, all handled:

- Selecting by inclination alone lands at ~470 km and fails the altitude check.
  `--method first` demonstrates this: it yields 247–515 km.
- Filtering on *mean* altitude is not enough either. J2 short-period radial
  oscillation (~10–15 km) plus the geoid being ~12 km smaller at 70° latitude
  than at the equator pushed instantaneous geodetic altitude to 609 km for a
  subset whose mean altitudes were all in band.

So `SimConfig.verify_altitude_band` (default on) propagates each candidate over
one orbital period and keeps only those whose **actual** geodetic altitude stays
inside the band. This costs ~1 s and makes `altitude_band_km` a guarantee the
suite can assert rather than an approximation.

## Selection

Default `"planes"`: filter to one coherent shell (altitude band + inclination), group
satellites into **real orbital planes** by RAAN gap, then take a run of `n_planes`
planes that are consecutive in RAAN and fill each with `sats_per_plane` members
spread by mean anomaly.

The number of planes is an **output** of the clustering, not an input. An earlier
version forced a requested number of RAAN *clusters* by cutting the shell at its
largest gaps, which is a different thing entirely: asking for 5 planes on the 70°
shell merged its 34 real planes (10° apart) into 5 blocks of ~7, and picking "20 per
plane" out of such a block spread the picks across all 7 real planes by mean anomaly.
A nominal 5 x 20 selection was really ~3 satellites in each of 28 planes.

That is invisible in the geometry — every pairwise distance is still correct — but it
wrecks `+Grid`, whose in-plane neighbours are then a third of an orbit apart and
mostly beyond the 7000 km range cap. The baseline lost half its links and fell into
26 components, making any comparison against it meaningless. The stage-2 suite now
asserts that stage 1's plane labels and stage 2's independently recovered planes agree
one-for-one.

The 70° shell holds 702 satellites in 34 real planes; the default window is the run of
10 consecutive planes that supplies the most satellites without an unsampled gap
inside it, giving **187**.

Inclination defaults to auto: the best-populated shell inside the altitude band, which
for the current catalog is **70.0°**. To use the Gen1 53.2° shell instead:

```
python run_sim.py --inclination 53.2
```

Also available: `--method first`, `--method random --seed N`.

## Link budget

```
FSPL = 20log10(d_m) + 20log10(f_Hz) + 20log10(4π/c)
Pr   = Pt + Gt + Gr − FSPL − L_pol − L_misalign
N    = 10log10(k · T · B)
C    = B · log2(1 + 10^((Pr−N)/10))
```

RF ISL at Ku band, no atmospheric attenuation term (ISLs are in vacuum). Real
Starlink ISLs are optical; the requested parameter set is RF.

Given: 12.2 GHz, 100 MHz, 40 dBi both ends, 4.5 dB polarization, 0.5 dB
misalignment.

**Two parameters the SNR needs were not in the spec** and are marked `ASSUMED`
in `config.py`:

| Parameter | Default | Note |
|---|---|---|
| `tx_power_dbw` | 10.0 (10 W) | assumed |
| `noise_temperature_k` | 354.81 | assumed; typical Ku receive chain |

These yield 1.79 Gbps at 100 km, 1.13 Gbps at 1000 km, 646 Mbps at 5300 km.
Change them in `config.py`; nothing else needs touching.

## A note on the 7000 km range cap

It never binds. The grazing-tangent range between two satellites at equal
altitude *h* over the 6428 km blocking sphere is `2·√((6378+h)² − 6428²)`:

| Altitude | Max geometric range |
|---|---|
| 500 km | 4894 km |
| 550 km | 5168 km |
| 600 km | 5431 km |

Occlusion is always the binding constraint, so the cap filters zero links in a
single LEO shell (observed longest link: 5314 km). It is implemented anyway —
it is a real hardware constraint and it matters if you later mix shells or add
higher orbits. If you intended it to bind, it needs to be ≤ ~4500 km.

## Stages 2 and 3

**Stage 2** selects each satellite's four ISLs. `DoTD` implements the paper's
DTEG lookahead in both readings the paper admits — forward (anticipating future
link quality) and backward (accumulating realised scores) — against `Greedy`
(same score, current instant only) and `+Grid` (structural, no scoring). The two
places the paper contradicts itself are exposed as flags rather than resolved
silently: `--objective min|max` and `--latency-formula c_over_d|d_over_c`. Every
output carries the variant that produced it.

**Stage 3** puts cities on the ground and routes across the chosen topology.
Satellite-satellite edges are exactly the links the selector chose, never the
visibility set — routing over visibility yields plausible hop counts that are
identical for all four algorithms and silently destroys the comparison, so it has
a dedicated check and a dedicated mutant. `--metric hop|latency|inv_capacity`
exposes the third ambiguity: the paper says OSPF and reports both hops and
latency without saying which one OSPF minimises.

## Validation

Three suites, 151 checks, each printing the measured value and the bound: 29 for
stage 1, 86 for stage 2, 36 for stage 3.

On top of them, `mutants.py` injects 53 deliberate bugs one at a time, runs the
suite that should catch each, and restores and hash-verifies the file afterwards.
**51 killed, 2 verified equivalent, 0 unexpected survivors.** A survivor means a
check is missing; an equivalent mutant means the change provably cannot alter
behaviour on any input the suite can reach, and every such claim is written down
and re-checked rather than assumed.

Two bugs this caught that inspection would not have are described above: the
plane-selection failure that silently crippled `+Grid`, and the visibility matrix
that disagreed with its own transpose.

### Stage-1 checks
Beyond the six you asked for, the suite adds checks that exist to stop the
others passing vacuously:

- **A grazing pair 1 km above the atmosphere must be visible.** Without it, a
  visibility function that returned `False` unconditionally would pass the
  antipodal test.
- **A pair 1 km *below* the blocking radius must not be visible**, and across a
  401-point sweep from 200 km below the threshold to 200 km above it, the
  verdict must flip exactly where the configured blocking radius says. The
  antipodal check only probes a clearance of zero and the grazing check only
  probes one point above the threshold, which leaves the threshold's actual
  position unconstrained between them: loosening `blocking_radius_km` by 200 km
  admits 1,174 links straight through the solid body of the Earth, and both of
  those checks still pass. These two pin it.
- **A cross-plane pair must be measurably less stable than the in-plane pair**
  (CV 57.9% vs 0.47%). Confirms the stability metric discriminates rather than
  passing on everything.
- **Mean geocentric radius must be consistent with a LEO shell.** Catches
  metre/kilometre unit errors and wrong frame origins.
- **Kepler's third law**, as above.
- **Latency must equal d/c for every visible link**, not just at a spot value.
- **FSPL, SNR, capacity and latency at 1000 km must match pinned absolute
  values** — 174.175 dB, 33.924 dB, 1127.0 Mbps, 3.33564 ms — to a relative
  tolerance of 1e-9. Every other link-budget check is *relative*: a monotone
  sweep, or one array agreeing with another. Relative checks are blind to any
  error that moves every value by the same factor. The literals are hard-coded
  rather than recomputed each run, because a reference the suite derives from
  the same inputs would drift along with the code it is meant to be checking.
  The SNR and capacity pins depend on the two `ASSUMED` parameters, so they are
  asserted only while those hold their documented defaults and are skipped with
  a printed note otherwise; the FSPL and latency pins have no such dependency.

Every check drives the production code rather than recomputing the quantity
locally. The FSPL check used to violate that: it evaluated the closed form
inline and asserted on its own arithmetic, never calling
`leosim.linkbudget.free_space_path_loss_db` at all. It was therefore
unfalsifiable — replacing `20log10` with `10log10` in the production function, a
60 dB error at 1000 km, left the suite reporting every check passed. An audit of
all the checks found it to be the only instance.
