# ISL-crossing fraction at the RL evaluation scale (6x12 = 72 sats), 2026-11-08

`measure_isl_crossing.py`; raw output `report.txt`; tables `context.csv`,
`env_attach.csv`, `stage4.csv`, `env_rollouts.csv`. 1800 x 1 s, hop metric,
load = `EnvConfig.load`. Seeds 7-9 for Greedy, 7 for the other selectors.
All four selectors are connected at every step under both gateway sets.

## Headline (Greedy; the other scored selectors are within ~1 point)

| | populous20 | krishnan8 |
|---|---|---|
| satellite-steps with a gateway in view | 14.2% | 5.6% |
| routed bits crossing >= 1 ISL (stage 4 OSPF) | **27%** | **89%** |
| routed hops 0 / 1 / 2 / 3+ | 73 / 24 / 2.3 / 0.3% | 11 / 41 / 29 / 19% |
| mean ISL hops per routed bit | 0.30 | 1.61 |
| RL-env attachment, fewest ISL hops needed: share >= 1 | 30% | 90% |
| stranded demand (no reachable gateway) | 0.0% | 0.2% |
| scaled peak demand at load 1 | 23.4 Gbps | 9.1 Gbps |

- populous20 at 72 sats looks like the 187-sat diagnostic: 73% zero-hop there
  vs 76%.
- Load does not change the routed-bit distribution (fixed routes; seeds agree
  to 0.1%). It changes which bits get *delivered*: multi-hop bits are dropped
  first. At load 2 the ISL share of delivered bits falls to 18% (populous20)
  and 67% (krishnan8), and ISL-crossing bits are delivered at 13-18% against
  30-55% for zero-hop bits.

## What it implies for learned routing

- **krishnan8: most traffic is multi-hop.** 9 bits in 10 need an ISL and
  ~half of those need 2+. The forwarding decision touches almost all traffic.
- **populous20: most traffic is single-hop in a fixed-route sense, but that
  does not cap the env the way it caps Phase 2.** Even at load 0.25, stage 4
  delivers only 85% of *zero-hop* bits (44% at load 1): the gateway-seeing
  satellite's own downlink is the bottleneck. In the env a zero-hop bit can
  still be forwarded to another gateway-seeing satellite, and that is how
  backpressure earns its lead (load 1: 0.65 vs shortest-path 0.43
  delivered). So the room at populous20 is downlink load-spreading, not path
  choice. That is a narrower problem than krishnan8's.
- **Backpressure wanders.** It makes 4-9 ISL transmissions per arrived bit
  under populous20 and 18-23 under krishnan8, against shortest-path's
  0.2-0.4 and 0.8-1.7. A policy that spreads load with fewer transmissions
  has room between the two baselines.

## Findings, not resolved

- **Backpressure trails shortest-path at light load under krishnan8**
  (load 0.25: 0.845 vs 0.970 delivered, Greedy; similar for all selectors).
  It is backlog, not loss: the queues are unbounded, so delivered + backlog
  = 1, and BP ends the 200-step episodes with 15.5% still queued. It needs a
  queue gradient before it routes. This is far beyond the 0.015 gap recorded
  at 2x12/4x6. `validate_rlenv`'s "backpressure delivers at least what
  shortest path does" would fail here; not run at this configuration.
- The same `load` is ~2.6x less absolute demand under krishnan8, because it
  is scaled to the subset's downlink capacity, which is smaller with fewer
  gateways. Comparisons across gateway sets are at equal *relative* load.
- Two attachment rules give similar zero-hop shares (73% stage 4 vs 70% env
  under populous20; 11% vs 10% under krishnan8). Stage 4 picks the access
  satellite that minimises path cost; the env picks the best-capacity one.
