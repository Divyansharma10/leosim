# ISL Topology Design for LEO Satellite Constellations

Project reference — base paper, algorithm, simulator, and current status.

**Base paper:** Ron et al., *Time-Dependent Network Topology Optimization for LEO Satellite Constellations*, IEEE INFOCOM 2025. DOI: 10.1109/INFOCOM55648.2025.11044725

**Secondary paper (traffic model only):** Krishnan et al., *Reinforcement Learning for Opportunistic Routing in Software-Defined LEO–Terrestrial Systems*, arXiv 2601.13662

---

## 1. The problem

Satellites in a constellation like Starlink connect to each other with laser links, so traffic moves through space instead of coming down to the ground at every hop. This matters for two reasons: you can serve regions with no ground station nearby, and light travels ~47% faster in vacuum than in optical fibre, so long-distance paths through space can beat terrestrial fibre on latency.

But each satellite carries only about **four optical terminals**, so it can hold four inter-satellite links (ISLs) at a time. The constellation moves at ~7.7 km/s, so neighbours leave range constantly and each satellite must keep re-choosing its four partners.

That choice defines the network's capacity, end-to-end latency, and how often paths break. It is called **ISL topology design**.

---

## 2. DoTD — the base algorithm

**DoTD** = DTEG-based Optimal Topology Design, where DTEG = Dynamic Time-Expanded Graph.

### Scoring

Every candidate link gets one score, built from three normalised quantities:

| Term | Meaning |
|---|---|
| Capacity | How much data the link can carry (from link budget) |
| Latency | Propagation delay, distance ÷ c |
| Link churn | Did this link already exist at the previous step? If so, no reconnection cost |

The three are on incomparable scales (Gbps, milliseconds, a 0/1 flag), so each is normalised to 0–1 and combined as a weighted sum:

```
score = w1·(capacity) + w2·(1 − latency) + w3·(was connected before)
```

Paper uses w1 = w2 = 0.4, remainder on churn.

### Lookahead — the actual contribution

The score is not only about the present instant. Orbits are deterministic, so the entire network's future is computable. DoTD builds a **layered graph** — one layer per time slot, out to a horizon of a few minutes — and each satellite carries a **running score forward**: when satellite *i* evaluates *j*, it adds *j*'s accumulated score from earlier steps to the current link score.

Consequence: a link that looks excellent right now but leaves the satellite badly positioned a minute later scores lower than one that stays useful across the window. It selects partners for the next few minutes, not the next second.

The paper also proves the running total stays **finite** as time accumulates, so scores remain comparable over long runs.

### Selection

Each satellite takes its top four, subject to:
- Neither satellite may already hold four links
- Links are bidirectional — if *i* selects *j*, then *j* selects *i*

Dynamic programming, roughly **O(M²)** per time step. CPU-only, no learning.

### Baselines

| Baseline | Description |
|---|---|
| **+Grid** | Industry default. Two neighbours in own orbital plane, two in adjacent planes. No scoring, purely structural. |
| **Greedy** | Same score as DoTD, but current instant only — no lookahead. |

Greedy is the important comparison: the gap between Greedy and DoTD isolates exactly what the time-expansion buys.

### Reported results

New York → San Francisco: **DoTD 5 hops, Greedy 6 hops, +Grid 22 hops.** Average capacity +28.09% over Greedy, up to +70.47% over +Grid. Latency 4.2 ms vs 9.9 ms (Greedy) vs 203.6 ms (+Grid) on Sydney → Darwin.

---

## 3. The gap this project attacks

Every input to DoTD's score is **geometry and link physics**. Nothing in it knows:

1. **Where the traffic is.** A satellite over the empty Pacific at 3 a.m. selects its neighbours exactly as one over a dense region at peak hour does. Congestion is invisible.
2. **Who can reach the ground.** Data must eventually exit through a ground gateway. Whether a candidate neighbour currently has a gateway in view plays no part in link selection.

The topology is optimised for a network with no users in it.

**Extension:** add a demand-and-egress term to the score, so link selection also weighs predicted traffic load and gateway visibility. Same DP structure, same complexity — changing what goes into the score, not the algorithm's skeleton.

### The evaluation problem (critical)

DoTD's metrics — hop count, propagation latency, per-link capacity, churn — are **all load-independent**. Traffic appears in none of them.

So reproducing their metric set and adding a traffic-aware term will measure **nothing**. A traffic-aware method cannot win on axes that cannot see traffic.

Phase 2 therefore requires a **traffic and queueing layer**: offered load from the ground, queues at satellites, and a delay-or-throughput-under-load metric. The secondary paper supplies a usable model — population density modulated by time of day, feeding Poisson arrivals, with satellite queue dynamics.

This decision shapes what the simulator must track, so it is made early.

---

## 4. The simulator (`leosim`) — built

Five stages, each feeding the next.

### Stage 1 — Orbits

Download TLE (Two-Line Element) data — two lines of text per satellite encoding orbital parameters at a given epoch: inclination, altitude, position in orbit. Sources: CelesTrak (no account needed, used by the base paper) or Space-Track (registration required).

~8,000 Starlink satellites listed; filter to **100 spread across distinct orbital planes**, so the constellation is representative rather than bunched in one ring.

**Epoch matters:** the paper used June 10 2024, 08:23 UTC. Matching it keeps numbers comparable.

### Stage 2 — Propagate

Feed TLEs into **SGP4**, the standard orbit propagation model. Given a TLE and a time, it returns x, y, z position. Run for all 100 satellites across the window (e.g. every second for 30 minutes). Everything after this is geometry on those coordinates.

### Stage 3 — Distances

At each time step, compute distance between every pair: a 100×100 matrix, done 1,800 times for a 30-minute run.

### Stage 4 — Visibility

Proximity is not enough — the Earth may block the line. For each pair, take the straight line between them and find its closest approach to Earth's centre. If that is less than Earth's radius (6378 km) plus 50 km of atmosphere, the line passes through the planet and no link is possible. Also apply a **7000 km maximum range**.

Result: a true/false matrix of which pairs *could* link.

### Stage 5 — Link quality

For each visible pair, run a link budget (see §5) to get capacity, and compute latency as distance ÷ c.

### Output

For every time step: who can see whom, how far, how fast, how much bandwidth. Cached to disk, keyed on a fingerprint of config + TLE bytes + source code hash, so editing the geometry correctly invalidates the cache.

### What this is and is not

This is a **model of possibility** — every link that *could* exist, and its quality. It does **not** decide which four each satellite forms. That is DoTD's job, and it is not written yet. There is no traffic in it at all.

---

## 5. How capacity is calculated

Four steps, from distance to bits per second.

**1. Free-space path loss.** Power spreads out as it travels, falling off with the square of distance:

```
FSPL = 20·log10(d_km) + 20·log10(f_GHz) + 92.45
```

At 1000 km and 12.2 GHz: **~174 dB** — the signal arrives roughly 10^17 times weaker than it left.

**2. Received power.** Work in dB so everything adds and subtracts:

```
P_rx = P_tx + G_tx + G_rx − FSPL − losses
```

Antenna gains are 40 dBi each end (antennas concentrate energy into a beam rather than radiating in all directions). Losses: 4.5 dB polarisation, 0.5 dB misalignment.

**3. SNR.** Divide received power by thermal noise power across the bandwidth (100 MHz). In dB, a subtraction.

**4. Shannon capacity.**

```
C = B · log2(1 + SNR)
```

with SNR as a linear ratio, not dB. At 1000 km with these parameters: **~1.1 Gbps**.

### Two caveats

Shannon is an **upper bound** — real modulation and coding come in below it. The literature uses it anyway, because it is a fair basis for *comparing* topologies rather than predicting a product spec.

The whole chain reduces to **distance** — every other term is a fixed constant. So capacity is a monotonically decreasing function of distance, meaning capacity and latency in DoTD's score are both really measuring "closer is better," on different curves.

### Parameters (Table I of base paper)

| Parameter | Value |
|---|---|
| Carrier frequency | 12.2 GHz |
| Bandwidth | 100 MHz |
| Polarisation loss | 4.5 dB |
| Misalignment loss | 0.5 dB |
| Satellite antenna gain | 40 dBi |
| Ground station antenna gain | 33.2 dBi |
| Max communication range | 7000 km |
| Link selections per satellite (U) | 4 |
| Earth radius | 6378 km |
| Atmospheric layer height | 50 km |
| Objective weights | w1 = w2 = 0.4 |

---

## 6. Validation work

Errors in the geometry layer are **silent** — wrong physics produces plausible-looking numbers and plausible-looking plots, not crashes. Everything downstream depends on this layer, so it was validated before moving on.

An audit found three holes:

1. **The path-loss test could not fail.** It recomputed the answer with its own local arithmetic instead of calling the production function. A deliberately injected 60 dB error passed the whole suite.
2. **The occlusion threshold was untested.** Checks only probed the extremes (antipodal, grazing), never the band between. Loosening the blocking radius by 200 km admitted 1,616 new links — 1,174 passing through the solid Earth — with the suite still green.
3. **The cache ignored source code.** Fingerprint was config + TLE bytes only. Editing the geometry and re-running served stale results silently.

All three fixed. Suite went 23 → 29 checks with source-file hashing and hard-pinned absolute values. A 48-mutation sweep now catches 30, with 5 misses individually verified as provably equivalent mutants rather than assumed so.

### Still open

- **Time axis not tied to orbital motion.** Scaling the offset denominator makes a run span 12 hours while reporting 30 minutes, and all checks pass. Fix: assert that angular displacement between first and last step equals (elapsed ÷ orbital period) × 360°. This is the one real correctness hazard — DoTD's churn term depends on Δt meaning what it appears to mean.
- Pin Earth radius and atmosphere height as constants; assert the 6428 km blocking radius directly.
- Pin carrier frequency and bandwidth unconditionally (spec-given, part of the comparability claim).
- One geodetic altitude assertion.
- Move the mutation harness in-repo; prune orphaned cache directories.

---

## 7. What comes next

**Phase 1 — reproduce.** Implement DoTD plus both baselines (+Grid, Greedy) on top of the existing simulator. Build them together, not sequentially — DoTD alone produces numbers with nothing to compare against.

Target: **Experiment 1** of the paper — New York → San Francisco, 5 / 6 / 22 hops. This needs ground stations and OSPF routing on top of the topology. It is the concrete number that says whether the reproduction is genuine rather than merely running.

**Phase 2 — extend.** Traffic and queueing layer, then the demand-and-egress score term, then evaluation under load.

### Two implementation decisions

**argmin vs argmax.** The paper's text says select the *highest* scores, but Eq. 15 and Algorithm 1 line 19 both say `argmin`. Implement highest-score (matching the text's intent), but run both and see which reproduces their numbers. Either outcome is a finding.

**Latency formula.** The paper defines latency as `c/D` (Eq. 5, Eq. 9) rather than `D/c`. This is dimensionally inverted — fed into the `w2·(1 − L̄)` term, it rewards *longer* links where it means to reward shorter ones. Implement `D/c` correctly, additionally run their version, and report the difference rather than silently picking one.

*(Verify both of these directly in the PDF before citing them.)*

---

## 8. Open questions for the supervisor

1. Does the traffic/queueing layer belong in this project? It is real extra scope, and without it the extension cannot be measured at all.
2. Is a reproduction-and-extension report the expected deliverable, or is this aimed at a paper?
3. How closely must the reproduction match? If numbers come out in the same direction but different in magnitude, is that success or a discrepancy to chase?
4. Any existing satellite simulation code from past students?

---

## 9. Risks

- **The geometry layer is the real cost**, not the algorithm. Budget more than expected; it has already run over.
- **The reproduction may not reproduce.** Gains over +Grid (81% hop reduction, 96% latency reduction) are large enough that a discrepancy is plausible. That is a finding, not a failure — but flag it to the supervisor in advance.
- **Scope creep into routing.** Routing sits on top of topology and is a separate research area. Keep OSPF fixed, as the paper did.
- **Phase 1 is the fallback deliverable.** If the extension underperforms, the reproduction-and-correction study stands on its own.
