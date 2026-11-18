"""Per-slot inputs to the shared link network, built from the observation only.

The agent sees exactly what backpressure sees (``rlenv`` observation), plus
quantities derived from it. Nothing here reads the environment's internals.

Per directed slot (i, k) with neighbour m = neighbours[i, k]:

    0  q_i                queue observation, q / (q + q_ref)
    1  q_m
    2  q_i - q_m          backpressure's queue differential
    3  c_ik               slot ISL capacity / isl_ref
    4  (q_i - q_m) c_ik   backpressure's weight (Eq. 16); its sign is BP's bit
    5  dl_i               best downlink / downlink_ref, 0 if no gateway in view
    6  dl_m
    7  h_i                hops from i to the nearest gateway-seeing satellite
    8  h_m                over the selected links, min(h, H) / H, unreachable 1
    9  deg_i / D          live slots of i
   10  mean_k q_k         global context: the reward is mean + max of queues
   11  max_k q_k

h is computed by BFS over the observation's own neighbour slots, as
``ShortestPathPolicy`` does; with only one-hop queues a link cannot tell
whether m leads anywhere, and under krishnan8 nine bits in ten need an ISL.
ASSUMED feature set.

The observation stays float64 (see rlenv.env); features are cast to float32
for the network only. Backpressure's bit, which the regression guard depends
on, is never taken from these features but from ``BackpressurePolicy`` on the
float64 observation.
"""

from __future__ import annotations

import numpy as np
import torch

N_FEATURES = 12


def hop_distance(obs: dict, max_hops: int) -> np.ndarray:
    """(N,) min(h, H) / H, h = hops to the nearest satellite with a gateway in
    view over the selected links; 1 where none is reachable."""
    nbr, mask = obs["neighbours"], obs["action_mask"].astype(bool)
    n = nbr.shape[0]
    unreached = np.iinfo(np.int64).max
    dist = np.where(obs["downlink_capacity"] > 0, 0, unreached)
    frontier = np.nonzero(dist == 0)[0]
    level = 0
    while frontier.size and level < max_hops:
        level += 1
        nxt = np.unique(nbr[frontier][mask[frontier]])
        nxt = nxt[dist[nxt] > level]
        dist[nxt] = level
        frontier = nxt
    out = np.ones(n, dtype=np.float64)
    ok = dist != unreached
    out[ok] = dist[ok] / max_hops
    return out


def link_features(queue: torch.Tensor, isl: torch.Tensor, downlink: torch.Tensor,
                  neighbours: torch.Tensor, mask: torch.Tensor, hops: torch.Tensor) -> torch.Tensor:
    """(B, N, D, F) float32 features. Inputs batched: queue/downlink/hops (B, N),
    isl/neighbours/mask (B, N, D); neighbours int64 with -1 padding. Padding
    slots get finite garbage (neighbour 0); the mask removes them downstream."""
    b, n, d = isl.shape
    idx = neighbours.clamp(min=0).reshape(b, n * d)
    q_m = torch.gather(queue, 1, idx).reshape(b, n, d)
    dl_m = torch.gather(downlink, 1, idx).reshape(b, n, d)
    h_m = torch.gather(hops, 1, idx).reshape(b, n, d)
    q_i = queue[:, :, None].expand(b, n, d)
    dl_i = downlink[:, :, None].expand(b, n, d)
    h_i = hops[:, :, None].expand(b, n, d)
    deg = (mask.float().sum(-1, keepdim=True) / d).expand(b, n, d)
    q_mean = queue.mean(1)[:, None, None].expand(b, n, d)
    q_max = queue.max(1).values[:, None, None].expand(b, n, d)
    diff = q_i - q_m
    feats = torch.stack([q_i, q_m, diff, isl, diff * isl, dl_i, dl_m, h_i, h_m, deg, q_mean, q_max], dim=-1)
    return feats.float()
