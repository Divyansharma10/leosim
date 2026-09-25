"""TLE loading and satellite subset selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sgp4.api import Satrec

from .config import SelectionConfig

WGS84_EQUATORIAL_KM = 6378.137


@dataclass
class TleRecord:
    """One catalog entry plus the mean orbital elements we select on."""

    name: str
    line1: str
    line2: str
    norad_id: int
    inclination_deg: float
    raan_deg: float
    mean_anomaly_deg: float
    eccentricity: float
    mean_altitude_km: float
    epoch_jd: float


def load_tles(path: str | Path) -> list[TleRecord]:
    """Parse a 3-line-per-satellite (name/line1/line2) TLE file.

    Altitudes come from SGP4's Brouwer-corrected semi-major axis
    (``satrec.altp``/``alta``), not from a naive Kepler inversion of the TLE
    mean motion, which ignores the J2 correction and is off by several km.
    """
    raw = Path(path).read_text(errors="replace").splitlines()
    lines = [ln.rstrip() for ln in raw if ln.strip()]

    records: list[TleRecord] = []
    i = 0
    while i < len(lines) - 2:
        name, l1, l2 = lines[i], lines[i + 1], lines[i + 2]
        if not (l1.startswith("1 ") and l2.startswith("2 ")):
            i += 1
            continue
        try:
            sat = Satrec.twoline2rv(l1, l2)
        except Exception:
            i += 3
            continue

        mean_alt = 0.5 * (sat.altp + sat.alta) * WGS84_EQUATORIAL_KM
        records.append(
            TleRecord(
                name=name.strip(),
                line1=l1,
                line2=l2,
                norad_id=sat.satnum,
                inclination_deg=np.degrees(sat.inclo),
                raan_deg=np.degrees(sat.nodeo) % 360.0,
                mean_anomaly_deg=np.degrees(sat.mo) % 360.0,
                eccentricity=sat.ecco,
                mean_altitude_km=mean_alt,
                epoch_jd=sat.jdsatepoch + sat.jdsatepochF,
            )
        )
        i += 3

    if not records:
        raise ValueError(f"No valid TLE triples parsed from {path}")
    return records


def _filter_shell(
    records: list[TleRecord], cfg: SelectionConfig
) -> tuple[list[TleRecord], float]:
    """Restrict to one coherent orbital shell (altitude band + inclination)."""
    lo, hi = cfg.altitude_band_km
    pool = [
        r
        for r in records
        if lo <= r.mean_altitude_km <= hi and r.eccentricity <= cfg.max_eccentricity
    ]
    if not pool:
        alts = np.array([r.mean_altitude_km for r in records])
        raise ValueError(
            f"No satellites with mean altitude in [{lo}, {hi}] km. "
            f"Catalog altitude percentiles (5/50/95): "
            f"{np.percentile(alts, [5, 50, 95]).round(1).tolist()} km. "
            "Widen SelectionConfig.altitude_band_km."
        )

    if cfg.inclination_deg is None:
        # Auto: the shell with the most satellites inside the altitude band.
        binned = np.round([r.inclination_deg for r in pool], 1)
        values, counts = np.unique(binned, return_counts=True)
        target = float(values[np.argmax(counts)])
    else:
        target = float(cfg.inclination_deg)

    shell = [r for r in pool if abs(r.inclination_deg - target) <= cfg.inclination_tol_deg]
    if not shell:
        raise ValueError(
            f"No satellites at inclination {target} +/- {cfg.inclination_tol_deg} deg "
            f"within altitude band [{lo}, {hi}] km."
        )
    return shell, target


def group_into_planes(
    shell: list[TleRecord], gap_deg: float
) -> list[list[TleRecord]]:
    """Group satellites into real orbital planes by RAAN gap.

    The number of planes is an OUTPUT, not an input. The previous version took
    a requested count and cut the shell at its largest gaps, which forces
    exactly that many groups whatever the shell contains: asking for 5 on the
    70 degree shell merged its 34 real planes (10 degrees apart) into 5 blocks
    of ~7 planes each. Picking "20 per plane" out of such a block then spread
    the picks across all 7 real planes by mean anomaly, so a nominal 5x20
    selection was really ~3 satellites in each of 28 planes.

    That is invisible in the geometry -- every pairwise distance is still
    correct -- but it wrecks +Grid, whose in-plane neighbours are then a third
    of an orbit apart rather than adjacent, mostly beyond the range cap. The
    baseline loses roughly half its links and the network falls into dozens of
    components, so any comparison against it is meaningless. Hence: cluster on
    the gap, and let the shell say how many planes it has.

    ``gap_deg`` separates planes that are ~10 degrees apart while tolerating
    the ~0.4 degree spread inside one; :data:`topology.grid.PLANE_GAP_DEG`
    applies the same rule to recovered normals, independently, as a cross-check.
    """
    raan = np.array([r.raan_deg for r in shell])
    order = np.argsort(raan)
    sorted_raan = raan[order]

    # Cut wherever consecutive RAANs are more than gap_deg apart. The wrap gap
    # is included so a plane straddling 0/360 is not split in two.
    gaps = np.diff(np.concatenate([sorted_raan, [sorted_raan[0] + 360.0]]))
    cuts = np.nonzero(gaps > gap_deg)[0]
    if len(cuts) == 0:
        return [[shell[j] for j in order]]

    planes: list[list[TleRecord]] = []
    start = (cuts[0] + 1) % len(order)
    current: list[TleRecord] = []
    for step in range(len(order)):
        k = (start + step) % len(order)
        current.append(shell[order[k]])
        if gaps[k] > gap_deg:
            planes.append(current)
            current = []
    if current:
        planes.append(current)
    return planes


def _circular_raan(planes: list[list[TleRecord]]) -> np.ndarray:
    return np.array([float(np.mean([r.raan_deg for r in p])) for p in planes])


def select_plane_window(
    planes: list[list[TleRecord]], n_planes: int, sats_per_plane: int
) -> list[list[TleRecord]]:
    """The best run of ``n_planes`` planes that are consecutive in RAAN.

    Consecutive matters as much as populated: +Grid's cross-plane links go to
    the planes on either side, so a selection of planes scattered around the
    shell leaves most of them with no sampled neighbour within range. Windows
    are taken circularly and scored on

      1. how many satellites they can actually supply,
         ``sum(min(|plane|, sats_per_plane))``, largest first, then
      2. the widest internal RAAN gap, smallest first,

    so a window containing an unsampled slot in the shell (the 70 degree shell
    has one, a 20 degree step where a plane is missing) loses to one without.

    Counting satellites rather than planes-that-fill matters: a window with one
    nearly empty plane can tie on full planes yet leave the selection short,
    which used to be papered over by topping up from wherever in the shell had
    spares -- putting satellites outside the window and ruining the very grid
    the window was chosen to give.
    """
    if n_planes >= len(planes):
        return planes

    order = np.argsort(_circular_raan(planes))
    ring = [planes[i] for i in order]
    raan = _circular_raan(ring)

    best = None
    for start in range(len(ring)):
        idx = [(start + k) % len(ring) for k in range(n_planes)]
        window = [ring[i] for i in idx]
        supply = sum(min(len(p), sats_per_plane) for p in window)
        steps = np.diff(np.unwrap(np.radians(raan[idx])))
        widest = float(np.degrees(np.abs(steps)).max()) if len(idx) > 1 else 0.0
        key = (-supply, widest)
        if best is None or key < best[0]:
            best = (key, window)
    return best[1]


def select_satellites(
    records: list[TleRecord],
    cfg: SelectionConfig,
    shell_filter=None,
) -> tuple[list[TleRecord], dict]:
    """Choose the working subset. Returns (satellites, provenance info).

    ``shell_filter`` is an optional callable applied to the shell pool after the
    mean-altitude and inclination cuts and before plane grouping. The simulator
    passes one that verifies propagated geodetic altitude, which makes
    ``altitude_band_km`` a guarantee rather than an approximation.
    """
    rng = np.random.default_rng(cfg.seed)
    info: dict = {"method": cfg.method, "n_catalog": len(records)}

    if cfg.method == "first":
        chosen = records[: cfg.n_sats]

    elif cfg.method == "random":
        idx = rng.choice(len(records), size=min(cfg.n_sats, len(records)), replace=False)
        chosen = [records[i] for i in sorted(idx)]

    elif cfg.method == "planes":
        shell, inclination = _filter_shell(records, cfg)
        info["shell_inclination_deg"] = inclination
        info["shell_pool_raw"] = len(shell)

        if shell_filter is not None:
            shell = shell_filter(shell)
            if len(shell) < cfg.n_sats:
                lo, hi = cfg.altitude_band_km
                raise ValueError(
                    f"Only {len(shell)} satellites at inclination {inclination:.1f} deg "
                    f"keep geodetic altitude inside [{lo}, {hi}] km for a full orbit, "
                    f"need {cfg.n_sats}. Widen SelectionConfig.altitude_band_km or "
                    "pin a different inclination_deg."
                )
        info["shell_pool_size"] = len(shell)

        planes = group_into_planes(shell, cfg.plane_gap_deg)
        info["n_planes_found"] = len(planes)
        info["plane_populations"] = sorted((len(p) for p in planes), reverse=True)

        planes = select_plane_window(planes, cfg.n_planes, cfg.sats_per_plane)
        planes.sort(key=lambda p: float(np.mean([r.raan_deg for r in p])))
        info["planes_used_raan_deg"] = [round(float(x), 1) for x in _circular_raan(planes)]
        info["plane_sizes_used"] = [len(p) for p in planes]

        chosen = []
        for plane in planes:
            # Spread within the plane by mean anomaly so the picks are phase-
            # separated rather than adjacent duplicates.
            plane = sorted(plane, key=lambda r: r.mean_anomaly_deg)
            k = min(cfg.sats_per_plane, len(plane))
            pick = np.linspace(0, len(plane) - 1, k).round().astype(int)
            chosen.extend(plane[j] for j in dict.fromkeys(pick.tolist()))

        # Deliberately no top-up. A plane short of sats_per_plane contributes
        # what it has; pulling the shortfall from elsewhere in the shell would
        # add satellites outside the chosen window, with no in-plane partners
        # and no adjacent sampled plane -- exactly the structure the window was
        # selected to avoid. For method='planes', n_sats is an upper bound and
        # the realised count is an output.

        chosen = chosen[: cfg.n_sats]
        info["plane_of_sat"] = _plane_labels(chosen, planes)

    else:
        raise ValueError(f"Unknown selection method: {cfg.method!r}")

    if cfg.method != "planes" and len(chosen) < cfg.n_sats:
        raise ValueError(
            f"Selection produced only {len(chosen)} satellites, need {cfg.n_sats}."
        )
    if not chosen:
        raise ValueError("Selection produced no satellites.")

    info["n_selected"] = len(chosen)
    alts = np.array([r.mean_altitude_km for r in chosen])
    info["altitude_km"] = {
        "min": float(alts.min()),
        "mean": float(alts.mean()),
        "max": float(alts.max()),
    }
    return chosen, info


def _plane_labels(chosen: list[TleRecord], planes: list[list[TleRecord]]) -> list[int]:
    """Plane index for each chosen satellite, for the in-plane validation test."""
    lookup = {}
    for pi, plane in enumerate(planes):
        for r in plane:
            lookup[r.norad_id] = pi
    return [lookup.get(r.norad_id, -1) for r in chosen]
