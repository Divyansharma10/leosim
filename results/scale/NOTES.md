# Scale-up run: 34 planes (601 satellites) vs 10 planes (187), 5 s step

Tests REPRODUCTION.md §5.1: does +Grid's 7-hop New York -> San Francisco count rise
toward the paper's 22 when the constellation covers the whole shell?

Run 2026-09-25. `run_experiment1.py --all-metrics --step 5`, 30 min / 360 steps,
`objective=max, latency=d_over_c`, mask 25 deg. The drivers now pass
`n_sats = planes * per_plane`; before that, `--planes 34` would have been silently
truncated to SelectionConfig's default of 200 satellites.

The 5 s step is a RAM constraint: 601 satellites at 1 s need ~7.8 GB. The 10-plane
control at 5 s reproduces the Phase 1 result at 1 s, so the step does not confound.

Output: `p34x20_step5/`, `p10x20_step5/`, `run.log`, `run_bridge.log`, `validation/`.

## Median hops, New York -> San Francisco, hop metric

| | paper | 187 sats (5 s) | 601, grid_gaps=strict | 601, grid_gaps=bridge |
|---|---|---|---|---|
| +Grid | 22 | 7 | 17 (partition artefact) | **5** |
| Greedy | 6 | 6 | 5 | 5 |
| DoTD-forward | 5 | 5 | 6 | 6 |
| DoTD-backward | - | 5 | 6 | 6 |

Means (bridge, hop metric): +Grid 5.47, Greedy 5.76, DoTD-backward 5.94,
DoTD-forward 6.38. DoTD-forward still has the lowest latency (25.9 vs Greedy 26.9 ms)
and highest path capacity (1.021 vs 0.974 Gbps).

**At full-shell scale the paper's ordering does not hold on hops.** +Grid, bridged,
takes the fewest hops of the four; DoTD-forward the most. The scale-up does not
reproduce the paper's 22-hop +Grid; §5.1 stays open.

## The +Grid gap policy is our choice, not the paper's

The 70 deg shell has 34 populated planes and two empty 10 deg slots (RAAN ~160 and
~0 deg), leaving two 20 deg gaps. The paper does not say how its grid treats a
missing plane. `TopologyConfig.grid_gaps` exposes both readings:

- `strict` (default, Phase 1 behaviour): planes are adjacent only across gaps up to
  1.5 x the median (15 deg). Right for a partial window, where a doubled gap means an
  unsampled plane. On the full shell it splits +Grid into two disconnected halves
  (planes 0-14 and 15-33) at every timestep.
- `bridge`: gaps up to 2.5 x the median (25 deg) are adjacent. Joins the full shell
  into one wrapped grid; still rejects a tripled gap.

The 15 deg cutoff is ours. Both values are pinned in `validate_topology.py`, and the
variant label and output filenames carry the policy.

## Validation at 34 planes (5 s step)

Current suite (96 stage-2 checks), logs in `validation/`, re-run 2026-09-26:

| suite | strict | bridge | 10-plane control, 5 s |
|---|---|---|---|
| stage 2 (`validate_topology`) | 92/96 | 93/96 | 86/86 (older 86-check suite) |
| stage 3 (`validate_routing`) | 36/36 | 35/36 | 36/36 |

Every failure is intended; none is a bug.

- **stage 2, both policies**: "Greedy / DoTD-forward / DoTD-backward: one connected
  component at every timestep" -- 66, 46 and 99 of 360 timesteps disconnected. The
  scored selectors form closed 5-cliques at global scale; see the next section.
- **strict, stage 2**: "+Grid: one connected component at every timestep" -- 360 of
  360 timesteps disconnected. That check exists to fail on exactly this
  configuration.
- **bridge, stage 3**: "+Grid needs more hops than every scored algorithm", which
  asserts the paper's result. It fails because the result is false here, not
  because anything is broken. Left failing on purpose.
- An earlier bridge stage-2 run also failed "+Grid output invariant to every scoring
  parameter": the check built its comparison +Grid from a fresh config, resetting
  grid_gaps to strict. Fixed to derive from the active config; it passes in the
  re-run above.

The DoTD-vs-Greedy hop inversion is therefore validated: apart from the connectivity
findings, every check that bears on selection, routing and the graph passes at 34
planes under both policies.

## Scored topologies are not connected at global scale

Stage 2's old check "the real runs are all connected" failed at 34 planes, and not
only because of +Grid:

| | connected fraction of timesteps | max components |
|---|---|---|
| Greedy | 0.817 | 4 |
| DoTD-forward | 0.872 | 4 |
| DoTD-backward | 0.725 | 4 |

The strays are closed 5-cliques: five satellites, ten internal links, all at degree 4,
drawn from plane pairs ~140 deg apart in RAAN -- counter-rotating planes crossing at
high latitude, where satellites bunch and become each other's four best candidates.
The 10-plane window spans 100 deg of RAAN and never contains such pairs. This is a
property of per-link scored selection at global scale, not a bug; it matters for
Phase 2, since demand landing on an island has no path to a gateway.

The old check asserted a fact about the 10-plane window ("every real run is
connected"). It now asserts what it protects: the connected flag is exercised with
both values.

## Window variance (4 x 6-minute windows at 1 s, bridge)

`run_windows.py`, output in `windows/`. Windows start 0, 24, 48 and 72 min after the
601-satellite selection's default start (2026-08-20T20:29:07Z), sampling about one
orbit. Every headline above came from ONE 30-minute slice; seeds vary demand, not
geometry, so the seed rule cannot see this.

Mean hops, New York -> San Francisco, hop metric:

| window | +Grid | Greedy | DoTD-forward | DoTD-backward | DoTD-f minus Greedy |
|---|---|---|---|---|---|
| +0 | 6.150 | 6.061 | 7.358 | 6.106 | +1.297 |
| +24 | 4.601 | 5.471 | 6.491 | 5.430 | +1.021 |
| +48 | 6.089 | 6.164 | 6.083 | 5.617 | -0.081 |
| +72 | 6.320 | 5.322 | 6.385 | 5.988 | +1.062 |

Fraction of timesteps connected:

| window | +Grid | Greedy | DoTD-forward | DoTD-backward |
|---|---|---|---|---|
| +0 | 1.000 | 0.333 | 0.792 | 0.172 |
| +24 | 1.000 | 0.744 | 1.000 | 0.681 |
| +48 | 1.000 | 0.406 | 0.839 | 0.322 |
| +72 | 1.000 | 0.781 | 0.783 | 0.764 |

What survives, and what does not:

- **DoTD-forward takes more hops than Greedy in 3 of 4 windows, by ~1 hop; in the
  fourth they tie** (-0.08 on the hop metric; on the latency route metric DoTD-forward
  is 0.3 hops better there). Report it as "in 3 of 4 windows", not as a property.
- **"+Grid (bridged) takes the fewest hops" does not survive.** It is best in 1 window
  of 4 (+24 min, 4.60); elsewhere it ties or trails. The 30-minute figure (5.47) was
  one slice.
- **Island frequency varies 2-4x between windows** (Greedy connected 33-78%,
  DoTD-backward 17-76%). A single-window disconnection rate is not a property of a
  selector.
- **Robust across all four: DoTD-forward is the most-connected scored selector**
  (78-100%), and bridged +Grid is always connected.
- Per-selector mean hops move by up to ~1.7 hops between windows -- the same order as
  the between-selector gaps. Any hop comparison needs its window spread beside it.
