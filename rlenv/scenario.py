"""Everything about an episode that no action can change, built once.

The stage-1 dataset, the selector's topology for the whole window, ground
access, and demand all come from the existing stages unchanged; this module
only reshapes them into per-satellite slots. Precomputing the selector over
the whole window and slicing episodes out of it is what stage 4 does too, so
DoTD-backward's memory at an episode's first step is the Phase 2 one.

Neighbour slots: satellite i's selected neighbours at t, in ascending
satellite index, fill slots 0..deg-1 of D; the rest are padding (-1, masked).
Index order is ASSUMED: it is stable while the link set is, and a slot's
meaning changes when links churn. ``isl_slot_capacity`` is stored as (T, N, D),
never (T, N, N), so memory grows as N, not N^2, per timestep.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from leosim.config import SelectionConfig, SimConfig
from leosim.dataset import LinkDataset
from leosim.simulation import run as run_stage1
from routing import RoutingConfig, compute_access, dataset_times
from run_traffic import utc_hours
from topology import DoTD, Greedy, PlusGrid, TopologyConfig
from topology.base import LinkSet
from traffic import SparseAccess, TrafficConfig, expected_rate_bps, gateway_access, load_sites, offered_bits

from .config import EnvConfig

ROOT = Path(__file__).resolve().parent.parent


def make_selector(name: str, dataset: LinkDataset, tcfg: TopologyConfig):
    if name == "+Grid":
        return PlusGrid(dataset, tcfg)
    if name == "Greedy":
        return Greedy(dataset, tcfg)
    if name == "DoTD-forward":
        return DoTD(dataset, dataclasses.replace(tcfg, lookahead="forward"))
    if name == "DoTD-backward":
        return DoTD(dataset, dataclasses.replace(tcfg, lookahead="backward"))
    raise ValueError(f"Unknown selector: {name!r}")


def neighbour_slots(adjacency: np.ndarray, d: int) -> np.ndarray:
    """(N, D) neighbour index per slot, ascending, -1 for padding."""
    n = adjacency.shape[0]
    deg = adjacency.sum(axis=1)
    if deg.max(initial=0) > d:
        raise ValueError(f"degree {deg.max()} exceeds the {d} slots")
    out = np.full((n, d), -1, dtype=np.int64)
    ii, jj = np.nonzero(adjacency)  # row-major: jj ascending within each i
    rank = np.arange(len(ii)) - np.searchsorted(ii, ii)
    out[ii, rank] = jj
    return out


def best_downlink(gw_access: list[SparseAccess], n_steps: int, n_sats: int) -> np.ndarray:
    """(T, N) capacity of each satellite's best gateway in view, 0 if none.
    Best = max capacity, which with a distance-only budget is the closest
    gateway, as in Krishnan et al. Eq. 5."""
    out = np.zeros((n_steps, n_sats))
    for acc in gw_access:
        for t in range(n_steps):
            sats, cap, _ = acc.at(t)
            np.maximum.at(out[t], sats, cap)
    return out


def attach_cities(city_access: list[SparseAccess], n_steps: int) -> np.ndarray:
    """(T, C) satellite each city's bits join: its highest-capacity access
    link. -1 where the city sees none. ASSUMED (see rlenv.config)."""
    out = np.full((n_steps, len(city_access)), -1, dtype=np.int64)
    for c, acc in enumerate(city_access):
        for t in range(n_steps):
            sats, cap, _ = acc.at(t)
            if len(sats):
                out[t, c] = sats[int(np.argmax(cap))]
    return out


@dataclass
class Scenario:
    cfg: EnvConfig
    dataset: LinkDataset
    linksets: list[LinkSet]
    neighbours: np.ndarray  # (T, N, D) int, -1 padding
    isl_slot_capacity: np.ndarray  # (T, N, D) bits/s, 0 on padding
    downlink: np.ndarray  # (T, N) bits/s
    city_sat: np.ndarray  # (T, C) int, -1 = no access
    rate_bps: np.ndarray  # (T, C) expected demand at the scaled load
    traffic: TrafficConfig  # with the scaled peak_total_gbps
    isl_ref_bps: float  # largest ISL capacity among VISIBLE pairs, whole window
    downlink_ref_bps: float  # largest downlink capacity, whole window
    queue_ref_bits: float
    buffer_bits: float  # inf = unbounded

    @property
    def n_steps(self) -> int:
        return len(self.linksets)

    @property
    def n_sats(self) -> int:
        return self.dataset.n_sats

    @property
    def n_slots(self) -> int:
        return self.neighbours.shape[2]

    def offered(self, seed: int) -> np.ndarray:
        """(T, C) Poisson bits per step: stage 4's ``offered_bits``, unchanged."""
        return offered_bits(self.rate_bps, self.dataset.config.step_s,
                            dataclasses.replace(self.traffic, seed=int(seed)))

    def arrivals(self, offered_t: np.ndarray, t: int) -> np.ndarray:
        """(N,) bits joining each satellite's queue at t."""
        sat = self.city_sat[t]
        ok = sat >= 0
        return np.bincount(sat[ok], weights=offered_t[ok], minlength=self.n_sats).astype(np.float64)

    def isl_capacity(self, t: int) -> np.ndarray:
        """(N, N) bits/s on selected links at t, 0 elsewhere."""
        return np.where(self.linksets[t].adjacency, np.asarray(self.dataset.capacity[t], dtype=np.float64), 0.0)


def build_dataset(cfg: EnvConfig, root: Path = ROOT, verbose: bool = False) -> LinkDataset:
    sim = SimConfig(
        duration_s=cfg.duration_s,
        step_s=cfg.step_s,
        selection=SelectionConfig(n_sats=cfg.n_sats_max, n_planes=cfg.n_planes,
                                  sats_per_plane=cfg.sats_per_plane),
    ).resolve(root)
    return run_stage1(sim, verbose=verbose)


def build_scenario(cfg: EnvConfig, root: Path = ROOT, dataset: LinkDataset | None = None,
                   verbose: bool = False) -> Scenario:
    ds = dataset if dataset is not None else build_dataset(cfg, root, verbose)
    n_steps, n = ds.n_steps, ds.n_sats
    if cfg.start_step + cfg.episode_steps > n_steps - 1:
        raise ValueError(
            f"episode {cfg.start_step}+{cfg.episode_steps} needs timestep "
            f"{cfg.start_step + cfg.episode_steps}; the dataset has 0..{n_steps - 1}")

    topo = TopologyConfig()
    linksets = make_selector(cfg.selector, ds, topo).run(n_steps)
    nbr = np.stack([neighbour_slots(ls.adjacency, topo.u_max) for ls in linksets])
    cap = np.asarray(ds.capacity, dtype=np.float64)
    t_idx = np.arange(n_steps)[:, None, None]
    i_idx = np.arange(n)[None, :, None]
    slot_cap = np.where(nbr >= 0, cap[t_idx, i_idx, np.maximum(nbr, 0)], 0.0)

    rcfg = RoutingConfig()
    unit = TrafficConfig(peak_total_gbps=1.0, gateway_sites=cfg.gateway_sites, seed=cfg.seed)
    sites = load_sites(unit, root)
    times = dataset_times(ds)
    city_acc = [SparseAccess.from_access(compute_access(ds, s, rcfg, times)) for s in sites.cities]
    gw_acc = [gateway_access(ds, s, rcfg, unit, times) for s in sites.gateways]
    downlink = best_downlink(gw_acc, n_steps, n)
    city_sat = attach_cities(city_acc, n_steps)

    # Scale demand to the subset (see EnvConfig.load). Expected rate is linear
    # in peak_total_gbps, so one unit evaluation fixes the scale exactly.
    rate_unit = expected_rate_bps(sites, utc_hours(ds), unit)[:n_steps]
    served_unit = float(np.where(city_sat >= 0, rate_unit, 0.0).sum(axis=1).mean())
    dl_mean = float(downlink.sum(axis=1).mean())
    if served_unit <= 0 or dl_mean <= 0:
        raise ValueError("this subset serves no demand or sees no gateway: load cannot be scaled")
    peak = cfg.load * dl_mean / served_unit
    traffic = dataclasses.replace(unit, peak_total_gbps=peak)
    rate = expected_rate_bps(sites, utc_hours(ds), traffic)[:n_steps]

    vis = np.asarray(ds.visible)
    isl_ref = float(cap[vis].max())
    dl_ref = float(downlink.max())
    buffer = np.inf if cfg.buffer_s is None else cfg.buffer_s * dl_ref
    return Scenario(cfg, ds, linksets, nbr, slot_cap, downlink, city_sat, rate, traffic,
                    isl_ref, dl_ref, cfg.queue_scale_s * dl_ref, buffer)
