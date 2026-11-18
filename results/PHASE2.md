# Phase 2 report — ground-aware terms for DoTD, evaluated under load

Phase 2 adds two terms to the DoTD/Greedy link score that look at the ground (where the
gateways are, where the demand is) and asks whether they move delivered throughput,
delay, loss, stretch or gateway load balance under offered load.

**This is a minimum-viable evaluation: one constellation slice, one time window, one
seed, untuned weights.** It is not a study. The demand-perturbation run CLAUDE.md
requires before any extension result is reported is now done (§4.6), and so is a 1 s
check of the 5 s step at a saturated load (§4.5). The numbers say whether there is
anything worth studying; with one traffic seed and one window they are still not a
claim of improvement.

Every number comes from `results/phase2/`; `report.txt` is the full output of
`report_phase2.py`, which only reads what `TrafficSim` wrote.

---

## 1. Headline

**The terms do not produce an effect that can be told apart from noise, and they do
not touch gateway load imbalance — the one place a topology could gain at 187
satellites.**

- **Gateway imbalance: no change.** Mean imbalance moves by at most ±0.05 on ~5.4 in any
  configuration; the per-gateway load shares at 5 Gbps are the same to within 1.5
  points. Seoul and Taipei carry ~47% of all feeder load on ~17% of feeder capacity with
  the terms off, and still do with them on. The median imbalance gets slightly *worse*
  with both terms on (Greedy +0.19 to +0.27, DoTD-forward +0.05 to +0.11).
- **Greedy + both terms delivers 0.4–0.7 points more** than Greedy from 3 Gbps up. But
  **neither term alone reproduces that**: congestion alone *loses* 0.3–0.4 points at
  every load from 3 Gbps, and egress alone is mixed (+0.2 at 3 Gbps, −0.5 at 5 Gbps). A
  gain present only in the combination, with both components negative, is what
  perturbing the topology at random would also look like. With one seed it cannot be
  attributed to the terms.
- **The Greedy gain needs a near-perfect forecast** (§4.6). With the predictor's
  per-city demand off by a lognormal factor of sd 0.5 (×/÷1.65) it is intact; at sd 1
  (×/÷2.7) its mean roughly halves and goes negative on some noise seeds; at sd 2 it is
  negative from 3.5 Gbps. Changing only the noise seed moves delivery by up to
  2 points, three times the gain itself.
- **The 5 s step did not create the Greedy gain** (§4.5): at 3.5 Gbps and 1 s it is
  +0.72 points against +0.69 at 5 s.
- **DoTD-forward + both terms is slightly worse** than DoTD-forward from 3.5 Gbps up
  (−0.1 to −0.4 points). At 1 s and 3.5 Gbps the sign flips (+0.25), so the DoTD-forward
  change is inside step-size noise either way.
- **Stretch and path length: no meaningful change** (−0.2% stretch, +0.1% path km).
- The largest difference on the page is not the extension at all: **Greedy (terms off)
  beats DoTD-forward (terms off) by 2.4 points at 5 Gbps** (0.904 vs 0.880).

## 2. Configuration

| | |
|---|---|
| constellation | 187 satellites, 10 real planes of the 70° shell (Phase 1 default) |
| window | the default 30 min (360 steps at 5 s), start 2026-08-20 (median TLE epoch) |
| step | **5 s** (1 s would take ~5× longer); one 1 s check at 3.5 Gbps, §4.5 |
| seed | 7 |
| feeder | stage-1 budget, 12.2 GHz / 100 MHz (the default, where feeders bind) |
| gateways | `populous20`; 7 of the 20 never see a satellite in this slice (Cairo, Moscow, Kinshasa, Istanbul, Lagos, Johannesburg, Sydney) |
| routing | fixed shortest path, hop metric, 25° mask, anycast to the cheapest gateway |
| demand | 200 cities, population × diurnal, Poisson, direction gateway → user |
| loads | peak aggregate 2, 2.5, 3, 3.5, 4, 4.5, 5 Gbps |
| topology flags | `objective=max, latency=d_over_c, grid_gaps=strict`, `w_lookahead=1` |

Configurations: Greedy and DoTD-forward, each with terms off (0 / 0) and on
(`w_egress = 0.2`, `w_congestion = 0.2`); plus two Greedy-only ablations, egress only
(0.2 / 0) and congestion only (0 / 0.2).

**Weights: 0.2 each, UNTUNED.** Picked, not searched. 0.2 is the base score's churn
weight `w3` and half of the capacity and latency weights (0.4 each), so the terms can
reorder near-tied candidates without overriding geometry; and it is
`ExtensionValidationConfig.probe_weight`, the value at which the extension suite shows
every timestep's selection differs from Phase 1 — the terms are demonstrably not
inert there. No other value was tried.

## 3. What the two terms are

Both come from an open-loop `GroundPrediction` (`traffic/predict.py`) built from
geometry and population before any traffic is simulated. Neither reads queue state.

- **Egress (reward), `w_egress`.** Per satellite: the best feeder capacity to any
  gateway above the mask, 0 if none. Per candidate link: the **max** of its two ends.
  Integrated over DoTD's forward window with the same weights as the geometric
  lookahead (Greedy sees only the current step). Intent: give gateway-visible
  satellites ISLs so traffic can reach a feeder in fewer hops.
- **Congestion (penalty), `w_congestion`.** Per satellite: expected user demand
  landing on it, each city's rate split equally over the satellites it sees. Per link:
  the mean of its two ends, current step only. Intent: steer ISLs away from satellites
  already loaded by users.

Both normalised by causal running maxima; both 0 by default, which reproduces Phase 1
bit-for-bit (pinned by SHA-256 in `validate_extension.py`).

## 4. The numbers

### 4.1 Delivered fraction of routed demand

| load (Gbps) | Greedy off | Greedy both | Greedy egress | Greedy cong | DoTD-f off | DoTD-f both |
|---|---|---|---|---|---|---|
| 2.0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2.5 | 0.9995 | 1.0000 | 0.9996 | 0.9991 | 0.9994 | 0.9995 |
| 3.0 | 0.9922 | 0.9980 | 0.9945 | 0.9894 | 0.9902 | 0.9927 |
| 3.5 | 0.9787 | 0.9856 | 0.9785 | 0.9751 | 0.9696 | 0.9685 |
| 4.0 | 0.9566 | 0.9633 | 0.9556 | 0.9536 | 0.9419 | 0.9392 |
| 4.5 | 0.9308 | 0.9359 | 0.9272 | 0.9272 | 0.9108 | 0.9065 |
| 5.0 | 0.9043 | 0.9085 | 0.8989 | 0.9007 | 0.8803 | 0.8762 |

Loss is 1 − delivered (end-to-end, from per-city delivery). The configurations
diverge from 2.5–3 Gbps, where the first gateway feeders saturate; below that every
configuration delivers everything. Demand no satellite can see (24.5% of offered, a
ground-segment property, identical for every configuration) is excluded.

Terms on minus off, delivered (points):

| load | Greedy both | Greedy egress | Greedy cong | DoTD-f both |
|---|---|---|---|---|
| 3.0 | +0.58 | +0.22 | −0.29 | +0.26 |
| 3.5 | +0.69 | −0.02 | −0.36 | −0.11 |
| 4.0 | +0.67 | −0.10 | −0.30 | −0.28 |
| 4.5 | +0.51 | −0.36 | −0.36 | −0.43 |
| 5.0 | +0.43 | −0.53 | −0.36 | −0.41 |

Timestep-paired, Greedy both vs off at 5 Gbps: more delivered on 18% of steps, less on
25% — the aggregate gain comes from a minority of steps. Congestion-only is the one
consistent within-run signal: worse on 13% of steps, better on 1.7%.

### 4.2 Delay under load (mean, ms)

| load | Greedy off | Greedy both | Greedy egress | Greedy cong | DoTD-f off | DoTD-f both |
|---|---|---|---|---|---|---|
| 2.0 | 7.79 | 7.81 | 7.82 | 7.75 | 7.60 | 7.61 |
| 2.5 | 8.81 | 7.81 | 8.24 | 9.39 | 8.84 | 8.21 |
| 3.0 | 11.07 | 9.62 | 11.29 | 11.57 | 12.45 | 12.35 |
| 3.5 | 15.66 | 15.48 | 16.20 | 15.70 | 18.18 | 19.31 |
| 4.0 | 20.15 | 19.74 | 20.71 | 20.18 | 22.76 | 23.38 |
| 4.5 | 23.30 | 23.56 | 24.52 | 23.56 | 26.10 | 26.54 |
| 5.0 | 26.28 | 26.51 | 27.02 | 26.41 | 28.84 | 28.34 |

Greedy + both is 1.0–1.4 ms faster at 2.5–3 Gbps, at the knee; from 4.5 Gbps the
sign flips. p95 delay is set by the 50 ms buffer once any feeder saturates (~57 ms
for every configuration from 4 Gbps) and does not discriminate.

### 4.3 Gateway load imbalance (peak / aggregate feeder utilisation; 1 = balanced)

| load | Greedy off | Greedy both | Greedy egress | Greedy cong | DoTD-f off | DoTD-f both |
|---|---|---|---|---|---|---|
| mean, 3.0 | 5.39 | 5.39 | 5.43 | 5.41 | 5.50 | 5.50 |
| mean, 5.0 | 5.41 | 5.41 | 5.45 | 5.43 | 5.53 | 5.52 |
| median, 3.0 | 4.90 | 5.12 | 4.90 | 5.03 | 4.83 | 4.90 |
| median, 5.0 | 4.88 | 5.15 | 4.87 | 5.04 | 4.77 | 4.83 |

Imbalance barely depends on load, because which gateway a city reaches is decided by
hop distance, not load. Per gateway at 5 Gbps, share of feeder load vs share of feeder
capacity (same in every configuration to within 1.5 points; full table in `report.txt`):

| gateway | capacity share | load share (Greedy off) | load share (Greedy both) | steps over capacity (off / both) |
|---|---|---|---|---|
| Seoul | 0.088 | 0.224 | 0.224 | 69 / 72 |
| Taipei | 0.078 | 0.235 | 0.245 | 42 / 49 |
| Shanghai | 0.074 | 0.103 | 0.100 | 36 / 36 |
| New York | 0.094 | 0.007 | 0.007 | 0 / 0 |
| Mexico City | 0.075 | 0.005 | 0.005 | 0 / 0 |

Loss at these loads is three East Asian feeders overflowing while the Americas'
feeders idle. The terms do not move load between them.

### 4.4 Stretch factor and path length

Demand-weighted over routed city→gateway paths. Stretch: Greedy 7.21 → 7.19 with both
terms, DoTD-forward 7.19 → 7.18 (5 Gbps); path length +2 km on ~2340 km. Neither
changes.

Stretch is a poor metric for city→gateway paths and is reported with that warning: a
city a few km from its gateway still climbs ~1100 km to a satellite and back, so its
stretch runs to the hundreds (max 363 seen in validation) and dominates a mean. Path
length is reported beside it for that reason. 12.5% of routed demand has no stretch
at all (a city on its gateway's coordinates; `populous20` puts every gateway on a
city) and is counted apart, not as 1.

### 4.5 Step size at saturation: 1 s vs 5 s, 3.5 Gbps

3.5 Gbps is where the terms' effect peaks (Greedy +0.69 in §4.1). The spread between
all configurations keeps growing to 5 Gbps, but that spread is mostly Greedy vs
DoTD-forward, not the terms. Greedy and DoTD-forward, terms off and both on, seed 7,
same window (1800 steps at 1 s). Output in `results/phase2/step1/`.

| | off, 1 s | both, 1 s | Δ at 1 s | Δ at 5 s |
|---|---|---|---|---|
| Greedy | 0.9781 | 0.9853 | +0.72 | +0.69 |
| DoTD-forward | 0.9695 | 0.9720 | +0.25 | −0.11 |

The Greedy gain is the same at 1 s, so the coarser step did not create it. The
DoTD-forward change flips sign between steps, so it is inside step-size noise. The
terms-off levels agree with 5 s to within 0.06 points (Greedy 0.9781 vs 0.9787,
DoTD-forward 0.9695 vs 0.9696).

### 4.6 Demand perturbation: the gain against forecast error

The congestion term's predictor normally uses exactly the evaluator's demand model,
which is the best case for the extension. Here each city's *predicted* rate is
multiplied by a fixed, mean-one lognormal factor `exp(sd·z − sd²/2)`, z ~ N(0, 1) per
city and constant in time: a market-size mis-estimate
(`TrafficConfig.predict_rate_noise_sd`). The evaluator's offered load does not change
(checked in `validate_extension.py`). Egress is geometry and does not change either, so
this tests the congestion term's information. Both terms on (0.2 / 0.2), 5 s step,
traffic seed 7, noise seeds 1–3. Output in `results/phase2/robust/` (`summarise.py`,
`gain_vs_noise.csv`).

Delivered, both terms minus terms off (points), mean [min, max] over noise seeds:

| Greedy | 3.0 | 3.5 | 4.0 | 4.5 | 5.0 |
|---|---|---|---|---|---|
| sd 0 (perfect) | +0.58 | +0.69 | +0.67 | +0.51 | +0.43 |
| sd 0.5 (×/÷1.65) | +0.58 [0.57, 0.58] | +0.68 [0.66, 0.70] | +0.66 [0.64, 0.69] | +0.50 [0.47, 0.54] | +0.41 [0.36, 0.46] |
| sd 1 (×/÷2.7) | +0.50 [0.35, 0.58] | +0.36 [−0.23, 0.66] | +0.21 [−0.62, 0.64] | −0.06 [−1.08, 0.49] | −0.26 [−1.50, 0.41] |
| sd 2 (×/÷7.4) | +0.21 [0.07, 0.34] | −0.05 [−0.52, 0.39] | −0.19 [−0.94, 0.46] | −0.37 [−1.28, 0.50] | −0.46 [−1.54, 0.65] |

| DoTD-forward | 3.0 | 3.5 | 4.0 | 4.5 | 5.0 |
|---|---|---|---|---|---|
| sd 0 (perfect) | +0.26 | −0.11 | −0.28 | −0.43 | −0.41 |
| sd 0.5 | +0.21 [0.10, 0.26] | −0.18 [−0.31, −0.11] | −0.34 [−0.46, −0.28] | −0.50 [−0.63, −0.43] | −0.48 [−0.62, −0.41] |
| sd 1 | +0.25 [0.25, 0.26] | −0.13 [−0.15, −0.11] | −0.30 [−0.33, −0.26] | −0.50 [−0.59, −0.42] | −0.54 [−0.71, −0.41] |
| sd 2 | +0.10 [−0.13, 0.25] | −0.26 [−0.60, −0.02] | −0.45 [−0.85, −0.17] | −0.74 [−1.28, −0.46] | −0.87 [−1.52, −0.49] |

- **Dependence on forecast error:** the Greedy gain holds at sd 0.5, roughly halves
  and can turn negative at sd 1, and is gone or negative from 3.5 Gbps at sd 2.
  DoTD-forward only gets worse as the error grows.
- **Spread between noise seeds:** at sd ≥ 1, changing only the noise seed moves
  delivery by up to 2 points, about three times the gain. That spread mixes the
  forecast being wrong with the selection simply being disturbed; the two are not
  separated here. Either way it shows that disturbing the topology at this scale moves
  delivery by more than the 0.7 points in §4.1.
- **What can be stated:** Greedy + both terms gains 0.4–0.7 points with a forecast
  correct to within ×/÷1.65 per city, and nothing at ×/÷2.7, on one traffic seed, one
  window and untuned weights. DoTD-forward does not gain at any error level.

### 4.7 Diagnostic: did routing use the links the terms changed?

Hypothesis tested: the terms changed which links exist, but shortest-path routing
ignored the new ones, so traffic and gateway imbalance stayed put. One run: Greedy,
terms off vs on (0.2 / 0.2), 3.5 Gbps, seed 7, 5 s step. It reproduces §4.1 exactly
(0.9787 / 0.9856). `TrafficSim` is wrapped, not modified. Each per-link and per-gateway
quantity is asserted equal to its own per-step totals. Output, script and full report
are in `results/phase2/diagnostic/`. "Bits on a link" = bits routed onto it (as in
`TrafficSim`, not thinned by upstream drops).

**The hypothesis is mostly wrong.** The added links were used, and per-gateway load
did move. The imbalance summary stayed flat because the load moved between East and
Southeast Asian gateways, not away from them.

- **Most traffic never touches an ISL.** 76.2% of routed bits go city → satellite →
  gateway on one satellite (0 ISL hops), 21.6% take 1 hop, 2.2% take more. Off and
  on are identical, and the mean is 0.27 hops in both. The topology can only affect
  about a quarter of the traffic.
- **1. Utilisation.** 94.4% of ISL link-steps carry nothing (94.5% with the terms on).
  Over the run, 44.0% of the 452 distinct link pairs carry nothing at all (46.6% of 468
  on). The top 10% of pairs carry 85% of ISL bits. Mean ISL utilisation is 0.001. One
  ISL briefly reached 1.07 with the terms off (0.9 Gb dropped); there were no ISL drops
  with them on. Loss is at the feeders.
- **2. Added links were not idle.** 10.8% of the terms-on link-steps are absent from
  terms-off at the same step (14,451 added, 13,859 removed, 73 pairs never present in
  off). They carry **23.0%** of terms-on ISL bits, about 2× the average link (mean
  0.02 vs 0.01 Gb per link-step; idle 93.4% vs 94.6%). The removed links carried
  15.1% of terms-off ISL bits. Routing did use the new links, so that part of the
  hypothesis fails.
- **3. Gateway-facing links: the egress term did not create more of them.** Share of
  link-steps by how many ends see a gateway: 0 ends 85.0 → 84.9%, 1 end 12.0 → 12.0%,
  2 ends 3.1 → 3.1%. Added links lean slightly toward 1 end (15.5%), but so do removed
  links (15.1%), so the two cancel. 1-end links carry 88% of ISL bits in both runs,
  because they are the first hop out of a gateway's satellite. **2-end links carry
  exactly 0 bits in both runs.** That is structural under the hop metric: both ends are
  already 1 hop from a gateway, so the link between them is never on a shortest path.
  Egress's max-combination is right to avoid these links.
- **4. Per gateway (Tb delivered over the run, off → on).** 5.2% of routed bits change
  gateway, a gross shift of 0.17 Tb against a net gain of 0.028 Tb.

  | gateway | routed off → on | delivered off → on | Δ delivered | delivered/routed off → on |
  |---|---|---|---|---|
  | Taipei | 0.939 → 0.983 | 0.920 → 0.965 | +0.045 | 0.980 → 0.981 |
  | Seoul | 0.897 → 0.901 | 0.868 → 0.885 | +0.017 | 0.968 → 0.983 |
  | Shanghai | 0.415 → 0.401 | 0.377 → 0.377 | +0.000 | 0.910 → 0.940 |
  | Ho Chi Minh City | 0.365 → 0.306 | 0.365 → 0.306 | −0.059 | 1 → 1 |
  | Dhaka | 0.329 → 0.321 | same | −0.008 | 1 → 1 |
  | Lahore | 0.305 → 0.336 | same | +0.031 | 1 → 1 |
  | Tokyo | 0.233 → 0.237 | same | +0.004 | 1 → 1 |
  | Mumbai | 0.227 → 0.226 | same | −0.002 | 1 → 1 |
  | Jakarta | 0.143 → 0.143 | same | −0.000 | 1 → 1 |
  | São Paulo | 0.068 → 0.070 | same | +0.002 | 1 → 1 |
  | Bogotá | 0.040 → 0.039 | same | −0.002 | 1 → 1 |
  | New York | 0.029 → 0.030 | same | +0.001 | 1 → 1 |
  | Mexico City | 0.021 → 0.020 | same | −0.001 | 1 → 1 |
  | Cairo, Moscow, Kinshasa, Istanbul, Lagos, Johannesburg, Sydney | 0 | 0 | 0 | — |

  The largest reassignment is Ho Chi Minh City → Taipei (0.057 Tb). That moves load
  **onto** one of the two overloaded gateways. Load also shuffled among Seoul,
  Taipei, Lahore and Dhaka in both directions. The Americas gateways moved by under
  0.002 Tb. Individual gateways shifted by up to 16% (Ho Chi Minh City) while mean
  imbalance went 5.40 → 5.41, because the shifts stayed within Asia and partly fed
  the hottest feeder.

**What this points to instead.** The +0.69-point gain is fewer drops at the three
saturated feeders: their losses fall from 0.085 to 0.058 Tb while Taipei receives
*more* routed load. Bits routed over at least one ISL lose 2.4% off and 0.86% on.
Bits with 0 ISL hops, whose routes the topology cannot change, also lose less
(2.04% → 1.63%). Both point to *when* load reaches a hot feeder, not *which* feeder it
goes to: pooled per-gateway totals cannot show that, and this run does not resolve
it. The diagnostic agrees with §5.1 (the terms do not target per-gateway feeder load)
and §5.3 (most demand cannot reach a distant gateway). It contradicts the premise of
§5.2: routing does use the added links, and changed the gateway for 5% of traffic.
Same caveat as everything above: one seed, one window.

## 5. What didn't work, and why

1. **Neither term targets the binding constraint.** At 187 satellites loss comes from
   per-gateway feeder overflow (Seoul, Taipei, Shanghai). Egress rewards links into
   *any* gateway-visible satellite, weighted by feeder *capacity*, blind to how
   loaded that gateway is — so it does not prefer an idle New York feeder over a
   saturated Seoul one. Congestion penalises satellites loaded by *users*, not
   satellites feeding an overloaded *gateway*. A term that could move load between
   gateways would need predicted per-gateway load against capacity, which the current
   `GroundPrediction` does not carry.
2. **Routing undoes the topology's choice.** Cities anycast to the hop-nearest gateway
   with fixed shortest-path routing, as the project requires ("keep OSPF fixed"). An
   extra ISL toward an idle gateway is only used if it makes that gateway hop-nearer
   for some city. Re-scoring a handful of links rarely does. (§4.7 qualifies this: added
   links carry 2× the average link's bits and 5% of traffic changes gateway, but 76% of
   bits use no ISL at all.)
3. **Most of the gap is geographic and unreachable.** Seven gateways never see a
   satellite in this 100°-RAAN slice, and demand is skewed to East and South Asia
   (the China-weighted city list). No ISL selection over these 187 satellites can
   carry Seoul's overflow to Mexico City.
4. **Congestion alone is consistently a small net loss** (−0.3 to −0.4 points from
   3 Gbps; worse on 13% of paired steps, better on 1.7%). Steering ISLs away from
   user-loaded satellites appears to cost path quality without relieving anything.
5. **No sign that DoTD's lookahead combines well with egress**, even though egress is
   integrated over the forward window: DoTD-forward + terms is worse than DoTD-forward
   from 3.5 Gbps.

## 6. Limitations

Stated plainly; each one bounds what §4 can mean.

- **187 satellites, not 601.** The evaluation constellation was meant to be the full
  shell (601 satellites, bridged). On this machine (16 GB) the 601-satellite traffic
  run was killed for low memory, and at ~45 s per run the sweep would have taken
  hours, so it was dropped. That is a machine limit, not a finding. 601 changes the problem materially — islands of disconnected satellites,
  gateway imbalance ranging 3.6–20.5 per step (vs ~5.4 here), and Greedy beating DoTD
  on hops — so nothing here transfers to it. The stage-4 suite *does* pass at 601
  (34/34, before the stretch checks were added; `results/scale/validation/traffic_p34_step5_bridge.log`).
- **One window, one seed: no variance estimate.** Every difference above is a single
  draw. Seeds were 3 for the partial terms-off baseline below 2.5 Gbps, where the
  seed-to-seed spread in mean delay was ~0.02 ms; above saturation it is unmeasured.
  Window-to-window variance at 601 moved per-selector hop means by up to 1.7 hops,
  the same order as between-selector gaps (`results/scale/NOTES.md`); at 187 it was not
  measured. The timestep-paired counts in §4.1 are within-run evidence only.
- **Weights untuned.** One value, 0.2 / 0.2, chosen for the reasons in §2. A different
  weight could do better or worse; nothing here says which.
- **Forecast error tested one way only.** §4.6 perturbs population (a fixed per-city
  factor). A shifted diurnal peak or a different trough ratio, which would move the
  error in time rather than between cities, was not run. Three noise seeds per level
  is enough to show the spread, not to estimate it.
- **5 s step.** Checked against 1 s at 1 and 2 Gbps, 3 seeds, terms off: scored
  selectors agree within 0.03 ms mean delay and ~0.1 imbalance, delivery identical
  (`results/traffic/p10` vs `p10_step5`). At saturation it was checked at one load
  (3.5 Gbps, §4.5): the Greedy gain holds, the DoTD-forward change flips sign. Other
  loads above saturation are 5 s only.
- **One feeder configuration.** The sourced 2.1 GHz feeder was not run; with it,
  feeders probably stop binding below 5 Gbps and the ISLs would become the
  constraint, which is where a topology term has more to work with.

## 7. What would be worth doing next

In order of how much each would change the conclusion:

1. A gateway-load-aware term (predicted per-gateway demand against feeder capacity),
   since §5.1 is the mechanism by which these terms cannot help.
2. Seeds and windows on the existing configurations, to put an error bar on the
   ±0.7-point differences in §4.1 before believing any of them. §4.6 already shows
   that a small change in the selection moves delivery by up to 2 points.
3. The 2.1 GHz feeder, where the ISLs bind instead of the feeders.
4. 601 satellites on a machine with more memory, or a sparse-backlog `TrafficSim`
   (the dense M×M arrays are ~5 MB each at 821 nodes and are not the memory
   problem; the ~2 GB stage-1 dataset is).

## Code changes made for this evaluation

- `traffic/network.py`: `TrafficSim` reports demand-weighted stretch (stage 3's
  definition, far end = the gateway actually reached), path length, and per-gateway
  feeder load. `SparseAccess` carries slant range.
- `validate_traffic.py`: 3 checks — traffic stretch equals stage 3's `shortest_path`
  stretch on identical paths (412 matched, worst error 3e-16), stretch varies and
  exceeds 1, co-located pairs are undefined rather than 1. Suite 37/37 at
  `--steps 300`.
- `mutants.py`: T17 (stretch omits ground links), T18 (stretch against the wrong
  gateway). Both killed. The full mutation sweep was not re-run.
- `run_traffic.py`: `--selectors`; rows written as each run finishes, with resume;
  non-default step in the filename; per-gateway shares to `*_gateways.csv`.
- `report_phase2.py`: the tables in §4.1–4.4.
- `traffic/config.py`, `traffic/predict.py`, `run_traffic.py`: predictor-only demand
  error (`predict_rate_noise_sd`, `predict_noise_seed`; `--predict-noise-sd`,
  `--predict-noise-seed`), for §4.6. When the error is on, it is stamped in the ground
  label and the output filename.
- `validate_extension.py`: 2 checks. The noise perturbs demand but not egress and
  repeats per seed; it never reaches the evaluator's `expected_rate_bps`. Suite 65/65
  (61/61 at `--steps 300`), `results/phase2/validate_extension.log`.
- `mutants.py`: X21 (noise silently dropped). Killed by the first check above,
  `results/phase2/mutant_X21.log`. The full sweep was not re-run.

## Reproduce

```bash
O=results/phase2
C="python run_traffic.py --step 5 --no-all-visible --loads 2 2.5 3 3.5 4 4.5 5 --seeds 7 --out $O"
$C --selectors Greedy DoTD-forward
$C --selectors Greedy DoTD-forward --w-egress 0.2 --w-congestion 0.2
$C --selectors Greedy --w-egress 0.2
$C --selectors Greedy --w-congestion 0.2
python report_phase2.py > $O/report.txt

# §4.5: 1 s at 3.5 Gbps (the terms-on run also wrote the All-visible reference)
S="python run_traffic.py --step 1 --loads 3.5 --seeds 7 --selectors Greedy DoTD-forward --out $O/step1"
$S
$S --w-egress 0.2 --w-congestion 0.2

# §4.6: demand perturbation
for sd in 0.5 1 2; do for s in 1 2 3; do
  python run_traffic.py --step 5 --loads 3 3.5 4 4.5 5 --seeds 7 --selectors Greedy DoTD-forward \
    --w-egress 0.2 --w-congestion 0.2 --predict-noise-sd $sd --predict-noise-seed $s --out $O/robust
done; done
python $O/robust/summarise.py

# §4.7: routing diagnostic (~30 s)
python $O/diagnostic/diagnose_routing.py > $O/diagnostic/report.txt
```

About 15 minutes for the §4.1 sweep and 3–4 minutes for each 1 s run. The nine
perturbation runs were run in parallel and took a few minutes in total.
