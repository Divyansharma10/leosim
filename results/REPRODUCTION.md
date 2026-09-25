# Reproduction report — DoTD (Ron et al., INFOCOM 2025)

Phase 1: reproduce *Time-Dependent Network Topology Optimization for LEO Satellite
Constellations* (DOI 10.1109/INFOCOM55648.2025.11044725) and its two baselines, and
report where the reproduction agrees, where it does not, and why.

Every number here comes from a CSV in this directory; every figure has the CSV that
produced it beside it. The run that produced them is recorded in `run_metadata.csv`.

---

## 1. Headline

**Experiment 1 reproduces.** New York → San Francisco, median hop count:

| | paper | ours |
|---|---|---|
| DoTD | 5 | **5** |
| Greedy | 6 | **6** |
| +Grid | 22 | **7** |

The DoTD/Greedy result holds under all three OSPF cost metrics we tried, so it does not
depend on how we resolved the paper's unstated routing cost. **+Grid does not
reproduce**, and §5 argues that is a property of our constellation rather than of the
algorithm.

**Three things in the paper did not reproduce, and one of them is a claim about DoTD's
main selling point.** They are §5, and they are the substance of this report.

---

## 2. What we ran, and how it differs from the paper

| | paper | ours | closeable? |
|---|---|---|---|
| Satellites | 907 | **187** | yes, at ~8x the memory |
| Orbital planes | full constellation | **10 consecutive**, ~10° apart in RAAN | yes |
| Shell | Starlink | 70.0° inclination, 500–600 km altitude band | — |
| Epoch | 2024-06-10 08:23 UTC | **2026-08-20 15:15 UTC** | **no** |
| Window | not stated | 1800 steps of 1 s | yes |
| Ground stations | not stated | 25° elevation mask (ASSUMED) | — |

The **epoch cannot be matched**. Ours is the median TLE epoch of the selected
satellites from a catalogue fetched 2026-08-20; reproducing the paper's 2024-06-10
would need a historical TLE file we do not have. Orbital geometry at a different epoch
is a different constellation, so no result here is expected to match theirs to the
digit — only in magnitude and ordering.

The **187 satellites in 10 planes spanning ~100° of RAAN** is the single most important
difference, and it is what §5.1 turns on.

Two parameters the paper's link budget needs but does not give are marked `ASSUMED` in
`leosim/config.py`: transmit power (10 dBW) and receive-chain noise temperature
(354.81 K). The elevation mask (25°) is likewise ours, not theirs — see §5.4.

---

## 3. Experiment 2 — all five Table II city pairs

`experiment2_pairs.csv`, `fig_hops_median.png`, `fig_latency_mean_ms.png`,
`fig_path_cap_mean_gbps.png`.

Mean hop count, and mean end-to-end latency in ms, over timesteps with a path:

| pair | route | reachable | +Grid | Greedy | DoTD-fwd | DoTD-back |
|---|---|---|---|---|---|---|
| S1 | Sydney → Darwin | **0.0%** | — | — | — | — |
| S2 | Miami → Calgary | 98.6% | **4.05** / 21.2 | 6.41 / 32.3 | 5.99 / 26.7 | 6.20 / 30.4 |
| S3 | New York → Miami | 92.1% | 3.65 / 16.9 | 3.40 / 15.3 | **3.20** / 12.3 | 3.34 / 14.7 |
| S4 | New York → San Francisco | 93.5% | 7.40 / 37.1 | 5.45 / 25.2 | **5.11** / 23.8 | 5.41 / 25.6 |
| S5 | Phnom Penh → Kathmandu | 84.9% | 4.70 / 23.8 | 4.02 / 17.8 | **4.00** / 17.7 | 4.08 / 18.1 |

DoTD-forward is best or tied-best on four of the five pairs. Two exceptions worth
stating rather than burying:

- **S2 Miami → Calgary: +Grid wins outright**, 4.05 hops against DoTD-forward's 5.99,
  and on latency too. This pair runs roughly across our plane structure, so the rigid
  grid happens to point the right way. It is the one pair where the paper's ordering
  inverts completely.
- **S5 is a tie.** DoTD-forward 3.998 vs Greedy 4.015 — a 0.4% difference that we would
  not claim as a result.

**Reachability is a property of the ground segment, not of the topology**: all four
algorithms have identical reachable fractions on every pair, which the validation suite
asserts.

---

## 4. Paired comparison — the honest view

`paired_hop_comparison.csv`, `fig_paired_hops.png`.

Hop counts are integers, so a difference in means of 0.3 can mean anything. At the same
timestep, how often does DoTD actually beat Greedy?

| pair | | fewer hops | equal | more hops |
|---|---|---|---|---|
| S2 | DoTD-forward | 39.2% | 44.7% | 16.1% |
| S3 | DoTD-forward | 32.6% | 54.8% | 12.5% |
| S4 | DoTD-forward | **43.0%** | 43.0% | 14.0% |
| S5 | DoTD-forward | 9.2% | 83.3% | 7.5% |
| S2 | DoTD-backward | 18.5% | 77.2% | 4.3% |
| S3 | DoTD-backward | 7.7% | 90.7% | 1.7% |
| S4 | DoTD-backward | 19.7% | 64.9% | 15.3% |
| S5 | DoTD-backward | 0.0% | 94.0% | 6.0% |

DoTD-forward's advantage is real but **not uniform**: it wins 43% of timesteps on S4 and
loses 14% of them. On S5 it is indistinguishable from Greedy at 83% of timesteps. A
reader given only the means would not know that.

**DoTD-backward is barely a different algorithm from Greedy.** It selects an identical
link set at 14.1% of timesteps and has a mean Jaccard overlap of **0.884** with Greedy's
topology (`topology_overlap.csv`); DoTD-forward's overlap is 0.618 and +Grid's is 0.196.

---

## 5. Where the reproduction disagrees with the paper

### 5.1 +Grid: 7 hops, not 22

The paper's +Grid takes 22 hops on New York → San Francisco. Ours takes 7.

Our best explanation is the constellation, not the baseline. +Grid's cost is that it
cannot route around its own structure: with global coverage, a path must follow the
grid's rows and columns and pays a large detour. Our 187 satellites span ~100° of RAAN,
so between two cities 4100 km apart the grid is already pointed roughly the right way
and the detour never materialises. S2 (§3), where +Grid *wins*, is the same effect
seen from the other side.

**This is testable and we have not tested it**: re-running on the full 34-plane shell
(~680 satellites, at a coarser time step to fit memory) should push +Grid's hop count up
while leaving DoTD and Greedy roughly where they are. Until that is run, we report the
discrepancy as open.

### 5.2 Capacity gains are ~2%, not 28%

The paper reports **+28.09%** average capacity over Greedy and up to +70.47% over +Grid.
Ours, on the same quantity (mean link capacity along the routed path):

| pair | DoTD-fwd vs Greedy (mean link) | (bottleneck) |
|---|---|---|
| S2 | +1.95% | +4.86% |
| S3 | +2.48% | +4.21% |
| S4 | +0.56% | +1.15% |
| S5 | +0.01% | +0.28% |

An order of magnitude short. Two candidate causes, neither confirmed:

1. **Which quantity Fig. 5c means.** We read "average capacity" as the mean capacity of
   the links along the path. If it is instead the path's bottleneck, our numbers roughly
   double — still nowhere near 28%. Both are reported in `experiment2_pairs.csv`.
2. **Capacity is a monotone function of distance in this link budget**, every other term
   being a fixed constant. So the spread of achievable capacities is set by the spread of
   achievable link *distances*, and in a dense 10-plane window almost every candidate
   link is short. A sparser or more global constellation offers a wider range of
   distances and therefore more room for a scoring function to win.

If (2) is the explanation, the capacity claim is also a casualty of the constellation
size, and the same scale-up that tests §5.1 would test this.

### 5.3 DoTD-forward churns *more* than Greedy — the paper claims the opposite

`experiment3_churn.csv`, `fig_churn.png`, `fig_w_lookahead.png`.

Lower link churn is one of DoTD's advertised benefits (their Fig. 6). We measure the
opposite for the forward variant:

| algorithm | churn / step | ISLs kept |
|---|---|---|
| +Grid | 0.000 | 100.00% |
| Greedy | 0.096 | 99.99% |
| **DoTD-forward** | **0.491** | 99.93% |
| DoTD-backward | 0.093 | 99.99% |

This is not an implementation error in the discount direction — the validation suite now
reads the discount curve out of the production code one offset at a time and asserts it
falls with offset. It is the **`w_lookahead` weight**, which the paper does not pin. The
churn reward is w3 = 0.2 inside a base score in [0,1]; the forward term adds up to
`w_lookahead` more priority containing no churn reward at all, so churn's share of the
ranking is roughly `w3 / (1 + w_lookahead)`.

Sweeping it (`w_lookahead_tradeoff.csv`) shows a clean monotone trade:

| w_lookahead | topology capacity (Gbps) | hops (S4) | churn / step |
|---|---|---|---|
| 0.0 (= Greedy) | 398.7 | 5.453 | 0.096 |
| 0.125 | 401.7 | 5.354 | 0.166 |
| 0.25 | 403.3 | 5.227 | 0.188 |
| 0.5 | 406.3 | 5.225 | 0.276 |
| 1.0 | 409.1 | 5.111 | 0.491 |
| 2.0 | 412.3 | 5.159 | 0.836 |

Capacity and hop count improve with `w_lookahead`; churn worsens 8.7×. **The paper
reports gains on the first two and a gain on the third, which this trade-off says cannot
happen simultaneously** unless their accumulation is normalised differently from ours.
That normalisation is not stated in the paper. It is the single most consequential
undocumented parameter we encountered.

**Caveat on our churn measurement**: the paper's Fig. 6 counts ISLs *kept* between
consecutive steps, and at our 1-second step that metric saturates — every algorithm sits
between 99.93% and 100%, because ~370 links change by less than one per step. The "ISLs
kept" column above does not discriminate and should not be used for comparison; churn
per step does. A coarser step would make the paper's metric meaningful and is worth
doing before claiming a Fig. 6 comparison.

### 5.4 S1 Sydney → Darwin cannot be evaluated

`experiment2_S1_mask_sensitivity.csv`, `fig_S1_mask_sensitivity.png`.

The paper's most dramatic latency numbers are on this pair — DoTD 4.2 ms, Greedy 9.9 ms,
+Grid 203.6 ms. **We have no path at any of the 1800 timesteps.**

Sydney's best satellite over the whole window reaches **23.22°** elevation, 1.8° below
our 25° mask. That mask is ASSUMED — it is not in the paper. So the existence of S1 in
our results is decided entirely by a parameter we chose:

| mask | Sydney sees | reachable | DoTD-fwd hops | DoTD-fwd latency |
|---|---|---|---|---|
| 25° (default) | 0.00 sats | 0.0% | — | — |
| 20° | 0.08 | 8.2% | — | — |
| 15° | 0.32 | 32.0% | — | — |
| 10° | 0.79 | 78.1% | 2.80 | 14.36 ms |
| 5° | 1.41 | 100.0% | — | — |

At a 10° mask (defensible for a gateway rather than a user terminal) S1 becomes
measurable: DoTD-forward 14.36 ms against Greedy's 13.85 ms and +Grid's 20.09 ms.
**That is not the paper's 4.2 / 9.9 / 203.6 ms in any respect** — our +Grid is 10×
faster than theirs and our DoTD 3× slower.

We report S1 as unevaluated at the stated configuration. The 10° row is a sensitivity,
not a result.

---

## 6. The paper's three internal contradictions

Each is implemented as a flag rather than silently resolved, so both readings can be run.

### 6.1 `argmin` vs "highest score"

The text says select the highest-scoring links; Eq. 15 and Algorithm 1 line 19 both say
`argmin`. `variant_matrix.csv`, `fig_variant_matrix.png`, New York → San Francisco:

| objective | latency formula | DoTD-fwd median | Greedy median | latency (DoTD-fwd) | matches paper's 5/6 |
|---|---|---|---|---|---|
| **max** | **d_over_c** | **5** | **6** | **23.8 ms** | **yes — exactly** |
| max | c_over_d | 5 | 5 | 26.5 ms | ordering lost |
| min | d_over_c | 3 | 4 | 40.6 ms | no |
| min | c_over_d | 3 | 4 | 36.7 ms | no |

**Only `objective=max` with `latency=d_over_c` reproduces the paper's 5/6 ordering, and
it reproduces it exactly.** Both of the paper's *written* forms fail.

The `argmin` result is interesting rather than merely wrong: it gives **fewer** hops (3
vs 5) because selecting the lowest-scoring links selects the *longest* ones, and long
ISLs cover more ground per hop — at the cost of nearly double the end-to-end latency
(40.6 ms vs 23.8 ms). So `argmin` is not nonsense, it optimises a different thing. But
the paper reports low hop counts *and* low latency together, which only `max` delivers.

**Conclusion: the prose is right and Eq. 15 / Algorithm 1 line 19 contain an error.**

### 6.2 Latency written as `c/D`

Eq. 5 and Eq. 9 define latency as `c/D` rather than `D/c`. This is dimensionally
inverted, and fed into the `w2·(1 − L̄)` term it rewards *longer* links where it means to
reward shorter ones. The table above shows the dimensionally correct `D/c` is also the
better performer — 5.11 hops and 23.8 ms against 5.44 and 26.5 ms.

**Conclusion: a typo. `D/c` is intended.** The validation suite separately asserts that
this flag changes only which topology is selected and never how delay is *reported*.

### 6.3 Churn direction

§5.3. Unresolved, and the only one of the three that is not obviously a typo.

---

## 7. Physical sanity of the latency numbers

Added as a validation check this session (`validate_routing.py`). For every pair and
every timestep with a path, reported end-to-end latency must exceed the straight-line
(chord) distance between the cities divided by c. **0 violations over 1194 paths**;
tightest margin +2.64 ms on S5.

Note the bound is the **chord**, not the great-circle arc. Any route from A to B is at
least |A − B| long by the triangle inequality, while the arc is *longer* than the chord
and so cannot bound a path from below. For these five short pairs the arc happens to
hold as well and is asserted separately, explicitly scoped — but it is not a general law,
and a long enough space path legitimately beats the ground arc traversed at c. That is
the mechanism behind the paper's "faster than fibre" premise, so asserting the arc
globally would amount to asserting that premise is impossible.

**Which raises an uncomfortable comparison.** Against idealised great-circle fibre at
2c/3, our paths are *slower*:

| pair | DoTD-forward | great-circle fibre | |
|---|---|---|---|
| S2 | 26.70 ms | 20.08 ms | +33.0% |
| S3 | 12.33 ms | 8.80 ms | +40.2% |
| S4 | 23.84 ms | 20.66 ms | +15.4% |
| S5 | 17.71 ms | 13.61 ms | +30.1% |

Two honest qualifiers: real fibre does not run along great circles (routes are typically
1.3–1.5× longer, which would put our paths ahead), and our sparse 187-satellite window
forces detours that a full constellation would not. But as measured, **this constellation
does not demonstrate the paper's motivating claim**, and we should not imply it does.

---

## 8. How far to trust these numbers

The validation is the part we would defend hardest.

| stage | checks | status |
|---|---|---|
| 1 — orbits, geometry, link budget | 29 | all pass |
| 2 — topology selection | 86 | all pass |
| 3 — ground stations and routing | 36 | all pass |

Plus a **mutation sweep**: 53 deliberate bugs injected one at a time into the production
code, each reverted and hash-verified afterwards. **51 killed, 2 verified equivalent, 0
unexpected survivors** (`mutation_report.txt`). A surviving mutant means a check is
missing; an equivalent mutant means the change provably cannot alter behaviour, and each
such claim is written down and re-checked rather than assumed.

Two bugs this methodology caught that testing-by-inspection would not have:

- **The satellite selection was silently wrong.** A nominal "5 planes × 20 satellites"
  was really ~3 satellites in each of 28 real planes, because the selector forced a RAAN
  cluster count rather than resolving actual orbital planes. Every distance was still
  correct, so nothing failed — but +Grid's in-plane neighbours were a third of an orbit
  apart and mostly beyond the range cap, the baseline lost half its links and fell into
  26 components, and any comparison against it was meaningless. Fixed; the suite now
  asserts stage 1 and stage 2 resolve planes one-for-one.
- **Visibility disagreed with its own transpose.** At 187 satellites, 6 pairs in 63
  million came out visible in one direction only — float32 rounding at the occlusion
  threshold. Rewritten to a formulation that is symmetric by construction.

**Known limitations**, stated plainly: one constellation, one epoch, one 30-minute
window, one shell; no traffic, no queueing, no congestion; capacity is Shannon (an upper
bound, as in the literature); routing is a static shortest path recomputed per timestep,
not a converged OSPF with timers; and the ground segment has one station per city with an
assumed mask.

---

## 9. Open questions

1. **Does the +Grid discrepancy close at scale?** The 34-plane, ~680-satellite run is the
   experiment that would settle §5.1 and probably §5.2. It needs a coarser time step to
   fit in memory. Worth the compute?
2. **How should `w_lookahead` be set?** §5.3 shows capacity, hops and churn cannot all
   improve together under our normalisation. If the paper's accumulation is normalised
   differently, that is the missing piece — is it worth writing to the authors?
3. **Is the S1 elevation mask worth pinning from a source?** It decides whether the
   paper's most dramatic result is measurable at all.
4. **Does Fig. 5c mean mean-link or bottleneck capacity?** Our reading changes the
   comparison by roughly 2×.
5. **Scope**: the traffic and queueing layer for Phase 2. DoTD's metrics are all
   load-independent, so a traffic-aware extension cannot be measured on them at all — it
   is real extra scope, and without it the extension has no axis to win on.

---

## Files

| file | contents |
|---|---|
| `experiment2_pairs.csv` | §3, all five pairs × four algorithms |
| `experiment2_S1_mask_sensitivity.csv` | §5.4 |
| `paired_hop_comparison.csv` | §4 |
| `topology_overlap.csv` | §4, Jaccard overlap with Greedy |
| `experiment3_churn.csv` | §5.3 |
| `variant_matrix.csv`, `variant_matrix_verdict.csv` | §6 |
| `w_lookahead_tradeoff.csv` | §5.3 |
| `run_metadata.csv` | constellation, epoch, variant labels |
| `mutation_report.txt` | §8 |
| `fig_*.png` | figures for the above |
| `phase1_run_log.txt` | console log of the run |
