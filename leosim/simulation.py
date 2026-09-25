"""Orchestration: TLE -> propagation -> geometry -> link metrics -> cache."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .config import SimConfig
from .dataset import LinkDataset
from .geometry import pairwise_geometry
from .linkbudget import link_metrics
from .propagate import propagate, refine_by_geodetic_altitude, resolve_start_time
from .tle import load_tles, select_satellites


def run(cfg: SimConfig, use_cache: bool = True, verbose: bool = True) -> LinkDataset:
    """Build (or load) the full link dataset for ``cfg``."""
    tle_path = Path(cfg.tle_path)
    if not tle_path.exists():
        raise FileNotFoundError(f"TLE file not found: {tle_path}")
    tle_bytes = tle_path.read_bytes()

    key = cfg.fingerprint(tle_bytes)
    cache_path = Path(cfg.cache_dir) / key

    if use_cache and LinkDataset.is_cached(cache_path):
        if verbose:
            print(f"[cache] hit {cache_path}")
        return LinkDataset.load(cache_path, cfg)

    if verbose:
        print(f"[cache] miss {key} -- computing")

    t_start = time.perf_counter()
    records = load_tles(tle_path)

    shell_filter = None
    if cfg.verify_altitude_band and cfg.selection.method == "planes":
        start_time = resolve_start_time(records, cfg)
        shell_filter = lambda pool: refine_by_geodetic_altitude(
            pool, start_time, cfg.selection.altitude_band_km
        )

    chosen, info = select_satellites(records, cfg.selection, shell_filter=shell_filter)
    if verbose:
        print(
            f"[select] {info['n_selected']} sats from {info['n_catalog']} catalog"
            + (
                f", shell i={info['shell_inclination_deg']:.1f} deg"
                f" (pool {info['shell_pool_raw']} -> {info['shell_pool_size']}"
                f" altitude-verified, {info['n_planes_found']} planes)"
                if "shell_inclination_deg" in info
                else ""
            )
        )
        a = info["altitude_km"]
        print(
            f"[select] mean altitude {a['min']:.1f} .. {a['max']:.1f} km "
            f"(mean {a['mean']:.1f})"
        )

    positions, altitudes, times = propagate(chosen, cfg)
    if verbose:
        print(
            f"[propagate] {positions.shape[0]} steps x {positions.shape[1]} sats "
            f"in {time.perf_counter() - t_start:.1f}s"
        )

    n_steps, n_sats = positions.shape[0], positions.shape[1]
    dtype = np.dtype(cfg.float_dtype)

    distance = np.empty((n_steps, n_sats, n_sats), dtype=dtype)
    visible = np.empty((n_steps, n_sats, n_sats), dtype=bool)
    capacity = np.empty((n_steps, n_sats, n_sats), dtype=dtype)
    latency = np.empty((n_steps, n_sats, n_sats), dtype=dtype)

    t_geom = time.perf_counter()
    for lo in range(0, n_steps, cfg.time_chunk):
        hi = min(lo + cfg.time_chunk, n_steps)
        d, v = pairwise_geometry(positions[lo:hi], cfg.geometry)
        c, lat = link_metrics(d, v, cfg.link_budget, dtype=dtype)
        distance[lo:hi], visible[lo:hi], capacity[lo:hi], latency[lo:hi] = d, v, c, lat
    if verbose:
        print(f"[geometry] link matrices in {time.perf_counter() - t_geom:.1f}s")

    dataset = LinkDataset(
        distance=distance,
        visible=visible,
        capacity=capacity,
        latency=latency,
        positions=positions,
        altitudes=altitudes,
        sat_names=[r.name for r in chosen],
        norad_ids=[int(r.norad_id) for r in chosen],
        plane_index=[int(p) for p in info.get("plane_of_sat", [-1] * len(chosen))],
        times_utc_iso=[times[0].utc_iso(), times[-1].utc_iso()],
        time_offsets_s=np.arange(n_steps) * cfg.step_s,
        config=cfg,
        selection_info=info,
    )

    if use_cache:
        dataset.save(cache_path)
        size_mb = sum(f.stat().st_size for f in cache_path.iterdir()) / 1e6
        if verbose:
            print(f"[cache] wrote {cache_path} ({size_mb:.0f} MB)")

    if verbose:
        print(f"[done] total {time.perf_counter() - t_start:.1f}s")
    return dataset
